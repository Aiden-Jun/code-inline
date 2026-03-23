from tokenizers import Tokenizer, models, decoders, trainers
from tokenizers.pre_tokenizers import ByteLevel
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import psutil
import time

SEQ_LEN = 256
N_LAYERS = 6
N_HEADS = 8
D_MODEL = 256
D_FF = 1024
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

DATA_PATH = "data/python_corpus.txt"
DATA_LIMIT_MB = 750

VOCAB_SIZE = 12000
TOKENIZER_PATH = "tokenizers/python_bpe_tokenizer.json"
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
MODEL_PATH = "models/python_model.pt"

GPU = ""

if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU = "Nvidia"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    GPU = "Apple Silicon"
else:
    DEVICE = "cpu"
    GPU = "something"

print(f"Using device {DEVICE} on {GPU}")

USE_FP16 = (DEVICE == "cuda")
USE_COMPILE = False


class PythonBPETokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, path, limit_mb):
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )
        max_bytes = limit_mb * 1024 * 1024
        read = 0

        def iterator():
            nonlocal read
            with open(path, "rb") as f:
                for line in f:
                    read += len(line)
                    if read > max_bytes:
                        break
                    yield line.decode("utf-8", errors="ignore")

        self.tokenizer.train_from_iterator(iterator(), trainer=trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)


class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=SEQ_LEN, data_path=DATA_PATH, data_limit_mb=DATA_LIMIT_MB):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_limit_bytes = data_limit_mb * 1024 * 1024
        self.line_offsets = []
        self.total_bytes = 0

        print("Scanning file to initialize lazy dataset...")
        read_bytes = 0

        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            offset = 0
            for i, line in enumerate(f):
                line_bytes = len(line.encode("utf-8"))
                if self.total_bytes + line_bytes > self.data_limit_bytes:
                    break

                self.line_offsets.append((offset, line_bytes))
                offset += line_bytes
                self.total_bytes += line_bytes
                read_bytes += line_bytes

                if i % 1000 == 0 and i > 0:
                    percent = min(100, read_bytes / self.data_limit_bytes * 100)
                    print(f"Scanned {i} lines | {percent:.2f}% of limit")

        if len(self.line_offsets) == 0:
            raise ValueError(f"No lines read from {data_path}. Check file and DATA_LIMIT_MB.")

        print(f"Lazy dataset ready: {len(self.line_offsets)} lines (~{self.total_bytes / (1024 ** 2):.2f} MB)")

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        start, size = self.line_offsets[idx]
        with open(self.data_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start)
            line = f.read(size)

        tokens = self.tokenizer.encode(line)

        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [self.tokenizer.tokenizer.token_to_id("<PAD>")] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]

        seq = torch.tensor(tokens, dtype=torch.long)
        return seq[:-1], seq[1:]


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.tok(x) + self.pos[:, :x.size(1)]
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)


def load_model(model, path, device):
    state = torch.load(path, map_location=device)
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state[k] = v
    model.load_state_dict(new_state)


class Model:
    def __init__(self, tokenizer_path=TOKENIZER_PATH, model_path=MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path

        self.tokenizer = PythonBPETokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        self.model = GPT().to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model.eval()

    def complete(self, text, max_tokens=20, temperature=0.7):
        ids = self.tokenizer.encode(text)[-SEQ_LEN:]
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        tokens_out = []

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(x[:, -SEQ_LEN:])
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)  # changed
                next_id = torch.multinomial(probs, num_samples=1).item()  # changed
                tokens_out.append(self.tokenizer.decode([next_id]))
                x = torch.cat([x, torch.tensor([[next_id]], device=self.device)], dim=1)

        return tokens_out

    def model_memory(self):
        ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        vram_mb = 0
        if self.device == "cuda":
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        return {"ram_mb": ram_mb, "vram_mb": vram_mb}


if __name__ == "__main__":
    start_time = time.time()
    tokenizer = PythonBPETokenizer()

    if not os.path.exists(TOKENIZER_PATH):
        print("Training tokenizer...")
        os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
        tokenizer.train(DATA_PATH, limit_mb=DATA_LIMIT_MB)
        tokenizer.save(TOKENIZER_PATH)
    else:
        tokenizer.load(TOKENIZER_PATH)

    dataset = CodeDataset(tokenizer, seq_len=SEQ_LEN, data_limit_mb=DATA_LIMIT_MB)
    model = GPT().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing checkpoint: {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        print("Model state resumed successfully.")

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_FP16)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.tokenizer.token_to_id("<PAD>"))

    steps_per_epoch = math.ceil(len(dataset) / BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    if USE_COMPILE:
        model = torch.compile(model)

    print(f"Starting Training | Total Steps: {total_steps} | Device: {DEVICE}")

    for epoch in range(EPOCHS):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            dtype = "cuda" if "cuda" in DEVICE else "cpu"
            with autocast(device_type=dtype, enabled=USE_FP16):
                logits = model(x)
                loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                epoch_pct = ((step + 1) / steps_per_epoch) * 100
                print(f"Epoch {epoch + 1}/{EPOCHS} | Progress: {epoch_pct:.2f}% | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | Checkpoint Saved")

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"Training finished in {elapsed_hours:.2f} hours.")
    with open("time.txt", "w") as f:
        f.write(f"{elapsed_hours:.2f}")

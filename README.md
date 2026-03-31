# krait

A lightweight GPT style code completion model

## Features

- **Custom BPE Tokenizer**: Byte level BPE tokenizer trained specifically on Python code
- **Transformer Architecture**: GPT like model with causal self attention
- **Multi-Device Support**: CUDA (Nvidia), MPS (Apple Silicon), and CPU
- **Lazy Dataset Loading**: Memory efficient training on large code corpus
- **Code Completion**: Generate Python code completions

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 6 |
| Heads | 8 |
| D Model | 256 |
| D FF | 1024 |
| Vocab Size | 12000 |
| Seq Length | 256 |
| Dropout | 0.1 |

## Requirements

```
torch
tokenizers
psutil
```

I have trained a Python model and a C++ model on my friend's GPU with this, and it was fast and good enough for finishing obvious code snippets.

This is for a hackathon thing.
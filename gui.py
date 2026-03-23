import sys
import re
from training import Model
from pathlib import Path
from PySide6.QtCore import Qt, QDir, QRect, QSize, QThread, Signal, QObject
from PySide6.QtGui import QFont, QAction, QColor, QTextCharFormat, QSyntaxHighlighter, QPainter
from PySide6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QFileDialog, QWidget, QMessageBox, QLabel, QVBoxLayout, QHBoxLayout, QTreeView, QFileSystemModel, QSplitter, QInputDialog, QPushButton, QDialog, QTextEdit


EDITOR_BG = "#1e1e1e"
EDITOR_FG = "#ffffff"
SELECTION_BG = "#555555"
INFO_BAR_BG = "#2e2e2e"
INFO_BAR_FG = "#dcdcdc"
TREE_SELECTION_BG = "#444444"
TREE_SELECTION_FG = "#ffffff"
TREE_HANDLE_BG = "#4d4d4d"
KEYWORD_COLOR = "#569CD6"
STRING_COLOR = "#D69D85"
COMMENT_COLOR = "#6A9955"
NUMBER_COLOR = "#B5CEA8"
LINE_NUMBER_BG = "#3d3d3d"
BUILTIN_COLOR = "#C586C0"
CURRENT_LINE_BG = "#323232"
DIALOG_LOGO_COLOR = "#dcdcdc"
BUTTON_BG = "#3a3a3a"
BUTTON_HOVER_BG = "#505050"
BUTTON_PRESSED_BG = "#2e2e2e"
BUTTON_FG = "#dcdcdc"


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []

        keywords = r'\b(False|None|True|and|as|assert|break|class|continue|' \
                   r'def|del|elif|else|except|finally|for|from|global|if|' \
                   r'import|in|is|lambda|nonlocal|not|or|pass|raise|return|' \
                   r'try|while|with|yield)\b'
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(KEYWORD_COLOR))
        self.rules.append((re.compile(keywords), fmt))

        builtins = r'\b(len|print|range|str|int|float|list|dict|set|tuple|open|input)\b'
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(BUILTIN_COLOR))
        self.rules.append((re.compile(builtins), fmt))

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(STRING_COLOR))
        self.rules.append((re.compile(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', re.DOTALL), fmt))

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(COMMENT_COLOR))
        self.rules.append((re.compile(r'#.*'), fmt))

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(NUMBER_COLOR))
        self.rules.append((re.compile(r'\b\d+(\.\d+)?\b'), fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, fmt)


class CompletionWorker(QObject):
    completions_ready = Signal(list)

    def __init__(self, model_model):
        super().__init__()
        self.model = model_model
        self.text = ""
        self.running = True

    def run(self):
        last_processed = ""
        while self.running:
            if self.text and self.text != last_processed:
                last_processed = self.text
                try:
                    result = self.model.complete(self.text)
                    result = self._truncate_at_newline(result)
                except Exception:
                    result = []
                self.completions_ready.emit(result)
            QThread.msleep(50)

    def _truncate_at_newline(self, tokens: list[str]) -> list[str]:
        truncated = []
        for token in tokens:
            if "\n" in token:
                truncated.append(token.split("\n", 1)[0])
                break
            truncated.append(token)
        return truncated

    def update_text(self, new_text):
        self.text = new_text

    def stop(self):
        self.running = False


class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.code_editor = editor

    def sizeHint(self):
        return QSize(self.code_editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self, model_model):
        super().__init__()
        self.model = model_model
        self.ghost_text = ""
        self.ghost_tokens = []
        self.ghost_index = 0

        font = QFont("Menlo")
        font.setPointSize(16)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {EDITOR_BG};
                color: {EDITOR_FG};
                selection-background-color: {SELECTION_BG};
                border: none;
            }}
        """)

        self.highlighter = PythonHighlighter(self.document())

        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.on_cursor_moved)
        self.update_line_number_area_width(0)
        self.highlight_current_line()

        self.thread = QThread()
        self.worker = CompletionWorker(model_model)
        self.worker.moveToThread(self.thread)
        self.worker.completions_ready.connect(self.on_completions_ready)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        self.last_cursor_pos = self.textCursor().position()

    def line_number_area_width(self):
        digits = max(3, len(str(max(1, self.blockCount()))))
        space = 10 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def line_number_area_paint_event(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(LINE_NUMBER_BG))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor(INFO_BAR_FG))
                painter.drawText(0, top, self.line_number_area.width()-2, self.fontMetrics().height(),
                                 Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QRect(cr.left(), cr.top(),
                                                self.line_number_area_width(), cr.height()))

    def highlight_current_line(self):
        extraSelections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor(CURRENT_LINE_BG))
            selection.format.setProperty(QTextCharFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)
        self.setExtraSelections(extraSelections)

    def accept_single_token(self):
        if not self.ghost_tokens:
            return
        token = self.ghost_tokens.pop(0)
        self.insertPlainText(token)
        self.ghost_text = "".join(self.ghost_tokens)
        self.ghost_index = 0
        self.viewport().update()
        self.update_ghost_text()

    def accept_full_token(self):
        if self.ghost_text:
            self.insertPlainText(self.ghost_text[self.ghost_index:])
            self.ghost_text = ""
            self.ghost_index = 0
            self.update_ghost_text()

    def on_cursor_moved(self):
        cursor = self.textCursor()
        if cursor.position() != self.last_cursor_pos:
            self.ghost_text = ""
            self.ghost_index = 0
            self.viewport().update()
            text_up_to_cursor = self.toPlainText()[:cursor.position()]
            self.worker.update_text(text_up_to_cursor)
            self.last_cursor_pos = cursor.position()

    def on_completions_ready(self, completions):
        if not completions:
            return
        joined = "".join(completions)
        if joined == self.ghost_text:
            return
        self.ghost_tokens = completions
        self.ghost_text = joined
        self.ghost_index = 0
        self.viewport().update()

    def update_ghost_text(self):
        cursor = self.textCursor()
        text_up_to_cursor = self.toPlainText()[:cursor.position()]
        self.worker.update_text(text_up_to_cursor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.insertPlainText(" " * 4)
            self.update_ghost_text()
            return
        super().keyPressEvent(event)
        self.update_ghost_text()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.ghost_text or self.ghost_index >= len(self.ghost_text):
            return
        cursor = self.textCursor()
        cursor_rect = self.cursorRect(cursor)
        ghost = self.ghost_text[self.ghost_index:].split("\n", 1)[0]
        if not ghost:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor(180, 180, 180, 110))
        painter.drawText(
            cursor_rect.left(),
            cursor_rect.bottom() - self.fontMetrics().descent(),
            ghost
        )
        painter.end()

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


class StartupDialog(QDialog):
    def __init__(self, width=1200, height=700):
        super().__init__()
        self.setWindowTitle("Editor")
        self.choice = None
        self.selected_folder = None
        self.setFixedSize(width, height)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        logo = QLabel("Editor")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color: {DIALOG_LOGO_COLOR}; font-size: 52px;")
        main_layout.addWidget(logo)

        main_layout.addStretch()

        btn_open_folder = QPushButton("Open Folder")
        btn_new_file = QPushButton("New File")
        btn_cancel = QPushButton("Cancel")

        for btn in [btn_open_folder, btn_new_file, btn_cancel]:
            btn.setFixedHeight(40)
            btn.setMinimumWidth(200)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BUTTON_BG};
                    color: {BUTTON_FG};
                    border-radius: 10px;
                    font-size: 16px;
                }}
                QPushButton:hover {{
                    background-color: {BUTTON_HOVER_BG};
                }}
                QPushButton:pressed {{
                    background-color: {BUTTON_PRESSED_BG};
                }}
            """)

        btn_open_folder.clicked.connect(self.open_folder)
        btn_new_file.clicked.connect(lambda: self.select("file"))
        btn_cancel.clicked.connect(lambda: self.select("cancel"))

        main_layout.addWidget(btn_open_folder)
        main_layout.addWidget(btn_new_file)
        main_layout.addWidget(btn_cancel)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_folder = folder
            self.select("folder")

    def select(self, choice):
        self.choice = choice
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, folder_mode=True, initial_folder=None):
        super().__init__()
        self.setWindowTitle("Model")
        self.current_file: Path | None = None
        self.is_modified = False
        self.current_folder: Path | None = None if not folder_mode else initial_folder
        self.folder_mode = folder_mode

        self.model = Model()
        self.editor = CodeEditor(self.model)
        self.editor.textChanged.connect(self.on_text_changed)

        self.info_bar = QWidget()
        self.info_bar.setStyleSheet(f"background-color: {INFO_BAR_BG}; border: none;")
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(6, 4, 6, 4)
        info_layout.setSpacing(0)
        self.info_bar.setLayout(info_layout)
        self.file_label = QLabel("Untitled")
        self.file_label.setStyleSheet(f"color: {INFO_BAR_FG};")
        self.file_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_layout.addWidget(self.file_label)
        info_layout.addStretch()

        self.tree = QTreeView()
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.tree.setModel(self.model)
        self.tree.setIconSize(QSize(0, 0))
        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)
        self.tree.header().hide()
        self.tree.clicked.connect(self.on_tree_clicked)
        tree_font = QFont("Menlo")
        tree_font.setPointSize(14)
        self.tree.setFont(tree_font)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.setStyleSheet(f"""
            QTreeView {{
                border: none;
                outline: 0;
            }}
            QTreeView::item {{
                height: 28px;
                border: none;
            }}
            QTreeView::item:selected {{
                background-color: {TREE_SELECTION_BG};
                color: {TREE_SELECTION_FG};
            }}
        """)

        splitter = QSplitter()
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {TREE_HANDLE_BG}; }}")
        if folder_mode:
            splitter.addWidget(self.tree)

        editor_container = QWidget()
        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        editor_container.setLayout(editor_layout)
        editor_layout.addWidget(self.info_bar)
        editor_layout.addWidget(self.editor)
        splitter.addWidget(editor_container)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.init_menu()
        self.update_info_bar()
        self.resize(1200, 700)

        if folder_mode and initial_folder:
            self.current_folder = initial_folder
            self.tree.setRootIndex(self.model.index(str(initial_folder)))

    def init_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        new_file_action = QAction("&New File", self)
        new_file_action.setShortcut("Ctrl+N")
        new_file_action.triggered.connect(self.new_file)
        file_menu.addAction(new_file_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        completion_menu = self.menuBar().addMenu("&Completion")

        accept_full_action = QAction("Accept Next", self)
        accept_full_action.setShortcut("Ctrl+L")

        accept_token_action = QAction("Accept Token", self)
        accept_token_action.setShortcut("Ctrl+P")
        accept_token_action.triggered.connect(self.editor.accept_single_token)
        completion_menu.addAction(accept_token_action)

        accept_full_action.triggered.connect(self.editor.accept_full_token)
        completion_menu.addAction(accept_full_action)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.folder_mode = True
            self.current_folder = Path(folder)
            self.tree.setRootIndex(self.model.index(folder))
            self.tree.show()

    def new_file(self):
        if self.folder_mode:
            if self.current_folder is None:
                QMessageBox.warning(self, "No folder", "Please open a folder first.")
                return
            name, ok = QInputDialog.getText(self, "New File", "File name:")
            if ok and name:
                new_path = self.current_folder / name
                if new_path.exists():
                    QMessageBox.warning(self, "Exists", "File already exists!")
                    return
                new_path.write_text("", encoding="utf-8")
                self.current_file = new_path
                self.editor.clear()
                self.is_modified = False
                self.update_info_bar()
                self.tree.setRootIndex(self.model.index(str(self.current_folder)))
        else:
            self.current_file = None
            self.editor.clear()
            self.is_modified = False
            self.update_info_bar()
            self.tree.hide()

    def on_tree_clicked(self, index):
        path = Path(self.model.filePath(index))
        if path.is_file():
            if self.maybe_save():
                self.current_file = path
                self.editor.setPlainText(path.read_text(encoding="utf-8"))
                self.is_modified = False
                self.update_info_bar()

    def maybe_save(self):
        if not self.is_modified:
            return True
        reply = QMessageBox.question(self, "Unsaved changes", "Save changes before continuing?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            return self.save_file()
        if reply == QMessageBox.Cancel:
            return False
        return True

    def save_file(self):
        if self.current_file is None:
            return self.save_file_as()
        self.current_file.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.is_modified = False
        self.update_info_bar()
        return True

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file as")
        if not path:
            return False
        self.current_file = Path(path)
        return self.save_file()

    def on_text_changed(self):
        if not self.is_modified:
            self.is_modified = True
            self.update_info_bar()

    def update_info_bar(self):
        if self.current_file:
            name = self.current_file.name
        else:
            name = "Untitled"

        if self.is_modified:
            name += " ●"

        self.file_label.setText(name)


def main():
    app = QApplication(sys.argv)
    dialog = StartupDialog(width=600, height=400)
    if dialog.exec() != QDialog.Accepted:
        sys.exit()
    folder_mode = False
    initial_folder = None
    if dialog.choice == "folder":
        if dialog.selected_folder:
            folder_mode = True
            initial_folder = Path(dialog.selected_folder)
    elif dialog.choice == "file":
        folder_mode = False
    else:
        sys.exit()
    window = MainWindow(folder_mode=folder_mode, initial_folder=initial_folder)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

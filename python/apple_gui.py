# python/apple_gui.py
# Запуск: сначала "mvn compile" в корне проекта, затем "python python/apple_gui.py"
# Требуется Python 3.8+ (tkinter входит в стандартную поставку) и java в PATH.

import json
import os
import queue
import re
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# Java запускается в неинтерактивном режиме: аргументы передаются в Main.main()
import glob
import shutil


def find_java():
    """Ищет JDK 21: сначала JAVA_HOME, затем /usr/lib/jvm, затем java из PATH."""
    candidates = []
    home = os.environ.get("JAVA_HOME")
    if home:
        candidates.append(os.path.join(home, "bin", "java"))
    candidates += sorted(glob.glob("/usr/lib/jvm/*21*/bin/java"))
    path_java = shutil.which("java")
    if path_java:
        candidates.append(path_java)
    for exe in candidates:
        if not os.path.isfile(exe):
            continue
        try:
            out = subprocess.run([exe, "-version"], capture_output=True, text=True).stderr
        except OSError:
            continue
        if '"21' in out:
            return exe
    return "java"


JAVA_BIN = find_java()

JAVA_CMD = [
    JAVA_BIN,
    "-Dfile.encoding=UTF-8",
    "-Dstdout.encoding=UTF-8",
    "-Dstderr.encoding=UTF-8",
    "-cp", "target/classes",
    "volsu.Main",
]

EXIT_TEXT = {0: "успех", 1: "ошибка ввода", 2: "ошибка модели", -1: "сбой запуска"}
RESULT_RE = re.compile(r"(Не яблоко|Яблоко)(?:\s*\(уверенность\s*([0-9.]+)\))?")
METRIC_RE = re.compile(r"(accuracy|precision|recall)\s*=\s*([0-9.]+)")

COLOR_APPLE = "#2e7d32"
COLOR_OTHER = "#c62828"
COLOR_NONE = "#757575"


def run_detect(project_dir, args):
    """Запускает Java-программу и возвращает (код возврата, stdout, stderr)."""
    proc = subprocess.run(
        JAVA_CMD + args,
        cwd=project_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    return proc.returncode, proc.stdout, proc.stderr


class App:
    def __init__(self, root):
        self.root = root
        root.title("apple-detector")
        root.resizable(False, False)
        self.queue = queue.Queue()
        self.buttons = []

        self.vars = {
            "project": tk.StringVar(value=DEFAULT_PROJECT),
            "image": tk.StringVar(value="test_images/apple.jpg"),
            "model": tk.StringVar(value="models/model_A1.json"),
            "data": tk.StringVar(value="data"),
            "out": tk.StringVar(value="models/model_A1.json"),
            "results": tk.StringVar(value="results"),
            "partition": tk.StringVar(value="A1"),
        }

        self.build_ui()
        self.poll_queue()

    # ---------------------------- интерфейс ----------------------------

    def build_ui(self):
        form = ttk.Frame(self.root, padding=10)
        form.grid(row=0, column=0, sticky="ew")

        rows = [
            ("Папка проекта", "project", "dir"),
            ("Изображение", "image", "file"),
            ("Модель (--model)", "model", "file"),
            ("Данные (--train/--evaluate)", "data", "dir"),
            ("Сохранить модель (--out)", "out", "save"),
            ("Папка результатов", "results", "dir"),
        ]
        for i, (title, key, kind) in enumerate(rows):
            ttk.Label(form, text=title).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(form, textvariable=self.vars[key], width=48).grid(row=i, column=1, padx=6)
            ttk.Button(form, text="...", width=3,
                       command=lambda k=key, t=kind: self.browse(k, t)).grid(row=i, column=2)

        ttk.Label(form, text="Разбиение (--partition)").grid(row=len(rows), column=0, sticky="w", pady=2)
        ttk.Combobox(form, textvariable=self.vars["partition"], values=("A1", "A2", "ALL"),
                     width=8, state="readonly").grid(row=len(rows), column=1, sticky="w", padx=6)

        buttons = ttk.Frame(self.root, padding=(10, 0))
        buttons.grid(row=1, column=0, sticky="ew")
        actions = [
            ("Распознать", self.do_classify),
            ("Обучить", self.do_train),
            ("Оценить", self.do_evaluate),
            ("Критерий R", self.do_criterion),
            ("Справка", self.do_help),
        ]
        for i, (title, command) in enumerate(actions):
            button = ttk.Button(buttons, text=title, command=command)
            button.grid(row=0, column=i, padx=3, pady=6)
            self.buttons.append(button)

        # Поле 1: результат распознавания
        ttk.Label(self.root, text="Результат распознавания", padding=(10, 4, 10, 0)).grid(row=2, column=0, sticky="w")
        self.result_label = tk.Label(self.root, text="—", font=("Arial", 22, "bold"),
                                     fg=COLOR_NONE, width=28, relief="groove", pady=8)
        self.result_label.grid(row=3, column=0, padx=10, pady=4)

        # Поле 2: статистика из файла
        ttk.Label(self.root, text="Статистика (из файла результатов)", padding=(10, 8, 10, 0)).grid(row=4, column=0, sticky="w")
        self.stats_text = tk.Text(self.root, height=7, width=60, state="disabled", font=("Consolas", 11))
        self.stats_text.grid(row=5, column=0, padx=10, pady=4)

        self.status = ttk.Label(self.root, text="Готово", anchor="w", padding=(10, 4))
        self.status.grid(row=6, column=0, sticky="ew")

    def browse(self, key, kind):
        initial = self.vars["project"].get()
        if kind == "dir":
            path = filedialog.askdirectory(initialdir=initial)
        elif kind == "save":
            path = filedialog.asksaveasfilename(initialdir=initial, defaultextension=".json")
        else:
            path = filedialog.askopenfilename(initialdir=initial)
        if path:
            self.vars[key].set(path)

    def v(self, key):
        return self.vars[key].get().strip()

    def abs_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.v("project"), path)

    def set_result(self, text, color):
        self.result_label.config(text=text, fg=color)

    def set_stats(self, text):
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state="disabled")

    def set_status(self, code, message=""):
        lines = [x for x in message.strip().splitlines() if x.strip()]
        tail = lines[-1] if lines else ""
        self.status.config(text=f"Код возврата {code} ({EXIT_TEXT.get(code, 'неизвестно')}). {tail}")

    # ---------------------- запуск Java в фоновом потоке ----------------------

    def run(self, args, callback):
        for button in self.buttons:
            button.config(state="disabled")
        self.status.config(text="Выполняется...")

        def worker():
            try:
                result = run_detect(self.v("project"), args)
            except Exception as e:
                result = (-1, "", f"{type(e).__name__}: {e}")
            self.queue.put((result, callback))

        threading.Thread(target=worker, daemon=True).start()

    def poll_queue(self):
        try:
            while True:
                (code, out, err), callback = self.queue.get_nowait()
                for button in self.buttons:
                    button.config(state="normal")
                callback(code, out, err)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    # ------------------------------ действия detect ------------------------------

    def do_classify(self):
        args = ["--image", self.v("image")]
        if self.v("model"):
            args += ["--model", self.v("model")]
        self.run(args, self.on_classify)

    def do_train(self):
        args = ["--train", self.v("data"), "--out", self.v("out"), "--partition", self.v("partition")]
        self.run(args, lambda code, out, err: self.set_status(code, err or out))

    def do_evaluate(self):
        args = ["--evaluate", self.v("data"), "--model", self.v("model")]
        if self.v("partition") in ("A1", "A2"):
            args += ["--partition", self.v("partition")]
        self.run(args, self.on_evaluate)

    def do_criterion(self):
        self.run(["--criterion", self.v("results")], self.on_criterion)

    def do_help(self):
        self.run(["--help"], self.on_help)

    # ------------------------------ обработка ответов ------------------------------

    def on_classify(self, code, out, err):
        self.set_status(code, err or out)
        match = RESULT_RE.search(out) if code == 0 else None
        if not match:
            self.set_result("—", COLOR_NONE)
            return
        is_apple = match.group(1) == "Яблоко"
        text = match.group(1)
        if match.group(2):
            text += f" ({match.group(2)})"
        self.set_result(text, COLOR_APPLE if is_apple else COLOR_OTHER)

    def on_evaluate(self, code, out, err):
        self.set_status(code, err or out)
        if code != 0:
            return
        stats, partition = self.read_stats_file()
        if stats is None:  # запасной вариант: разбор консольного вывода
            stats = {k: float(v) for k, v in METRIC_RE.findall(out)}
        self.set_stats(self.format_stats(partition, stats))

    def on_criterion(self, code, out, err):
        self.set_status(code, err or out)
        if code != 0:
            return
        lines = [x for x in out.splitlines() if x.startswith(("R(", "Лучшее"))]
        self.set_stats("\n".join(lines) if lines else out.strip())

    def on_help(self, code, out, err):
        self.set_status(code, err)
        messagebox.showinfo("Справка", out.strip() or err.strip())

    # ------------------------------ чтение статистики ------------------------------

    def read_stats_file(self):
        """Читает метрики из results/<разбиение>_confusion_matrix.csv (строки 'metric,<имя>,<значение>')."""
        partition = "A1"
        try:
            with open(self.abs_path(self.v("model")), encoding="utf-8") as f:
                partition = json.load(f).get("partitionName", "A1")
        except (OSError, ValueError):
            pass
        path = os.path.join(self.abs_path(self.v("results")), f"{partition}_confusion_matrix.csv")
        stats = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 3 and parts[0] == "metric":
                        stats[parts[1]] = float(parts[2])
        except (OSError, ValueError):
            return None, partition
        return stats, partition

    @staticmethod
    def format_stats(partition, stats):
        lines = [f"Разбиение: {partition}"]
        for key in ("accuracy", "precision", "recall"):
            value = f"{stats[key]:.4f}" if key in stats else "нет данных"
            lines.append(f"{key:<10}= {value}")
        return "\n".join(lines)


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()

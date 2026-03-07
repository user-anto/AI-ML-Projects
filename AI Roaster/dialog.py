import threading

_dialog_lock = threading.Lock()


def _show_text_dialog(title: str, text: str) -> None:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        print(f"\n{title}\n{text}\n")
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("720x420")
    root.minsize(520, 300)
    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))

    body = tk.Frame(root)
    body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

    text_area = scrolledtext.ScrolledText(
        body,
        wrap=tk.WORD,
        font=("Consolas", 11),
    )
    text_area.pack(fill=tk.BOTH, expand=True)
    text_area.insert("1.0", text)
    text_area.configure(state=tk.DISABLED)

    actions = tk.Frame(root)
    actions.pack(fill=tk.X, padx=10, pady=10)

    def copy_text() -> None:
        root.clipboard_clear()
        root.clipboard_append(text)

    copy_btn = tk.Button(actions, text="Copy", command=copy_text)
    copy_btn.pack(side=tk.LEFT)

    close_btn = tk.Button(actions, text="Close", command=root.destroy)
    close_btn.pack(side=tk.RIGHT)

    root.mainloop()


def show_text_dialog_async(title: str, text: str) -> None:
    def worker() -> None:
        with _dialog_lock:
            _show_text_dialog(title, text)

    threading.Thread(target=worker, daemon=True).start()

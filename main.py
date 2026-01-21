# main.py
import tkinter as tk
from tkinter import ttk
import traceback

from config import AppConfig
from ui.app import App


def main():
    root = tk.Tk()
    ttk.Style().theme_use("clam")  # 更稳定的主题
    cfg = AppConfig()
    app = App(root, cfg)
    root.geometry("960x550")
    root.mainloop()

def _tk_ex_handler(widget, exc, val, tb):
    """处理 tkinter 未捕获的异常"""
    traceback.print_exception(exc, val, tb)
    # 可选：弹窗提示用户
    import tkinter.messagebox as mb
    mb.showerror("程序异常", f"发生未处理的错误：\n{val}\n\n详情已打印到控制台")

# 绑定
tk.Tk.report_callback_exception = _tk_ex_handler

if __name__ == "__main__":
    main()

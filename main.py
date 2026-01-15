# main.py
import tkinter as tk
from tkinter import ttk
from config import AppConfig
from ui.app import App

def main():
    root = tk.Tk()
    ttk.Style().theme_use("clam")  # 更稳定的主题
    cfg = AppConfig()
    app = App(root, cfg)
    root.geometry("1100x650")
    root.mainloop()

if __name__ == "__main__":
    main()

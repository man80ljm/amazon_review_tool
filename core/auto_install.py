# auto_install.py  (建议作为“安装器/引导器”使用，而不是打进 exe 里)
import subprocess
import sys
import os
import threading
import tkinter as tk
import tkinter.messagebox as mb

PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

REQUIRED = [
    "torch",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "openpyxl",
    "python-docx",
    "jieba",
]

def _try_imports():
    try:
        import torch  # noqa
        import transformers  # noqa
        import sentence_transformers  # noqa
        return True
    except Exception:
        return False

def _append(text_widget: tk.Text, msg: str):
    text_widget.insert("end", msg + "\n")
    text_widget.see("end")
    text_widget.update_idletasks()

def _run_cmd_stream(cmd, text_widget: tk.Text):
    """实时输出 stdout/stderr"""
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    for line in p.stdout:
        _append(text_widget, line.rstrip("\n"))
    return p.wait()

def check_and_install_dependencies():
    # 1) 如果已具备依赖，直接返回
    if _try_imports():
        return True

    # 2) 创建一个隐藏 root，保证 messagebox / Toplevel 稳
    root = tk._default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()

    resp = mb.askyesno(
        "首次运行设置",
        "检测到缺少运行依赖包。\n\n"
        "程序将安装必要依赖（可能较大，取决于 torch 版本）。\n"
        "是否现在安装？"
    )
    if not resp:
        mb.showwarning("提示", "缺少必要依赖，程序将退出。")
        return False

    # 3) 安装窗口
    win = tk.Toplevel(root)
    win.title("正在安装依赖...")
    win.geometry("700x420")

    text = tk.Text(win, wrap="word")
    text.pack(fill="both", expand=True, padx=10, pady=10)

    _append(text, "开始安装依赖 ...")
    _append(text, f"Python: {sys.executable}")
    _append(text, f"Mirror: {PIP_MIRROR}")
    _append(text, "-" * 60)

    def worker():
        try:
            # 先升级 pip（减少奇怪错误）
            _append(text, "升级 pip ...")
            code = _run_cmd_stream([sys.executable, "-m", "pip", "install", "-U", "pip", "-i", PIP_MIRROR], text)
            if code != 0:
                _append(text, "⚠️ pip 升级失败，但继续尝试安装依赖。")

            # 正式安装
            for pkg in REQUIRED:
                _append(text, "-" * 60)
                _append(text, f"安装 {pkg} ...")
                code = _run_cmd_stream([sys.executable, "-m", "pip", "install", pkg, "-i", PIP_MIRROR], text)
                if code == 0:
                    _append(text, f"✅ {pkg} OK")
                else:
                    _append(text, f"❌ {pkg} 安装失败（返回码={code}）")
                    _append(text, "建议：复制上面的报错信息给我，我帮你定位。")
                    break

            _append(text, "-" * 60)
            if _try_imports():
                _append(text, "🎉 依赖已就绪！请关闭后重新启动程序。")
                mb.showinfo("完成", "依赖安装成功，请重新启动程序！")
            else:
                _append(text, "❌ 依赖仍未就绪（可能 torch 未成功）。")
                mb.showerror("失败", "依赖安装未完成。请把窗口日志复制给我。")
        finally:
            try:
                win.focus_force()
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()
    return _try_imports()

if __name__ == "__main__":
    ok = check_and_install_dependencies()
    sys.exit(0 if ok else 1)

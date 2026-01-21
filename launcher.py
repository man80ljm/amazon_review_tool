import os
import subprocess
import sys


def main():
    base = os.path.dirname(sys.executable)
    app_dir = os.path.join(base, "app")
    py = os.path.join(base, "venv", "Scripts", "pythonw.exe")
    script = os.path.join(app_dir, "main.py")
    if not os.path.isfile(py):
        raise FileNotFoundError(f"python.exe not found: {py}")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"main.py not found: {script}")
    subprocess.Popen([py, script], cwd=app_dir)


if __name__ == "__main__":
    main()

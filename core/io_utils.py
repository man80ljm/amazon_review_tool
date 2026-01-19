# core/io_utils.py
import os
import pandas as pd
import sys

def load_file(path: str, required_columns=None, required_cols=None) -> pd.DataFrame:
    """
    读取 CSV / XLSX

    兼容参数名：
    - required_columns（新/标准）
    - required_cols（旧/一些地方用的名字）
    - 两者都传时，以 required_columns 优先

    required_columns/required_cols 为 None -> 不做列校验（用于自动识别列名）
    """
    if required_columns is None and required_cols is not None:
        required_columns = required_cols

    path_lower = path.lower()

    if path_lower.endswith(".xlsx"):
        df = pd.read_excel(path, engine="openpyxl")

    elif path_lower.endswith(".csv"):
        last_err = None
        df = None
        used_enc = None

        # 多编码兜底
        for enc in ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                used_enc = enc
                break
            except Exception as e:
                last_err = e
                df = None

        if df is None:
            raise ValueError(f"CSV读取失败（编码问题）。最后错误：{last_err}")

        # 分隔符兜底（防止只有1列）
        if df.shape[1] == 1:
            for sep in (",", ";", "\t"):
                try:
                    df2 = pd.read_csv(path, encoding=used_enc, sep=sep)
                    if df2.shape[1] > 1:
                        df = df2
                        break
                except Exception:
                    pass
    else:
        raise ValueError("仅支持 .csv 或 .xlsx 文件")

    # 列校验：只有 required_columns 不为 None 时才校验
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"缺少必要列：{missing}\n需要列：{list(required_columns)}\n当前列：{list(df.columns)}"
            )

    # 不在这里写死任何列（比如 review_text），避免列名不一致导致导入阶段崩溃
    return df.copy()

# ========== 输出工具 ==========
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_excel(df_dict: dict, path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)

def resolve_model_path(rel_path: str) -> str:
    """
    在源码 / PyInstaller 打包环境下，统一解析模型路径
    """
    # PyInstaller 打包后
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # 指向 _internal
        return os.path.join(base, rel_path)

    # 开发环境
    return os.path.abspath(rel_path)
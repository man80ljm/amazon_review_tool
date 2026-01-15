# core/io_utils.py
import os
import pandas as pd

# ========== 通用文件读取（支持 CSV / XLSX） ==========
def load_file(path: str, required_columns: tuple) -> pd.DataFrame:
    path_lower = path.lower()

    if path_lower.endswith(".xlsx"):
        df = pd.read_excel(path, engine="openpyxl")

    elif path_lower.endswith(".csv"):
        last_err = None
        # 多编码兜底
        for enc in ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin1"):
            try:
                df = pd.read_csv(path, encoding=enc)
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
                    df = pd.read_csv(path, encoding=enc, sep=sep)
                    if df.shape[1] > 1:
                        break
                except Exception:
                    pass
    else:
        raise ValueError("仅支持 .csv 或 .xlsx 文件")

    # 列校验
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{missing}\n需要列：{required_columns}")

    # 基础清洗
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)

    return df


# ========== 输出工具 ==========
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_excel(df_dict: dict, path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)

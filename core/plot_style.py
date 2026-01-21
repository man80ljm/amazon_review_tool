# core/plot_style.py
from __future__ import annotations

from typing import Iterable, Optional


def apply_matplotlib_style(
    lang: Optional[str] = None,
    preferred_fonts: Optional[Iterable[str]] = None
) -> None:
    """
    Configure matplotlib to render CJK text when available.
    This is safe to call multiple times.
    """
    import matplotlib
    from matplotlib import font_manager as fm

    # Default candidate fonts for Windows/macOS/Linux
    default_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
    ]

    candidates = list(preferred_fonts) if preferred_fonts else default_fonts

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)

    if chosen:
        matplotlib.rcParams["font.family"] = chosen

    # Ensure minus sign renders correctly even with CJK fonts.
    matplotlib.rcParams["axes.unicode_minus"] = False

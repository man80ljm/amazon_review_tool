# ui/app.py
import sys
import os
from huggingface_hub import snapshot_download
import queue
import traceback

from pathlib import Path
from config import load_user_settings, save_user_settings

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import numpy as np

from config import AppConfig
from core.io_utils import load_file, save_csv, save_excel, ensure_dir
from core.sentiment import SentimentAnalyzer
from core.embedding import Embedder
from core.clustering import scan_k, fit_kmeans
from core.keywords import top_keywords_by_cluster
from core.representatives import top_representatives
from core.robustness import clustering_stability

from core.plot_k import recommend_k, plot_k_curves
from core.plot_style import apply_matplotlib_style
from core.translate import Translator
from core.insights import asin_cluster_percent, plot_heatmap, cluster_priority, plot_priority,cluster_priority_safe

from core.report_word import build_offline_report


class App(ttk.Frame):
    def __init__(self, master, cfg: AppConfig):
        super().__init__(master)
        self.master = master
        self.cfg = cfg

        # ✅【加在这里】设置窗口默认大小
        self.master.geometry("960x550")   # 你可以改这个数
        self.master.minsize(960, 550)      # 可选：防止缩太小
        
        # 加载本地 settings.json
        settings = load_user_settings()
        self.cfg.apply_user_settings(settings)

        #打包后加载模型路径修正
        from config import resolve_path
        self._resolve_path = resolve_path

        self.cfg.sentiment_model = resolve_path(getattr(self.cfg, "sentiment_model", "models/sentiment"))
        self.cfg.embedding_model = resolve_path(getattr(self.cfg, "embedding_model", "models/embedding"))
        self.cfg.translate_model_zh_en = resolve_path(getattr(self.cfg, "translate_model_zh_en", "models/translate/zh_en"))
        self.cfg.translate_model_en_zh = resolve_path(getattr(self.cfg, "translate_model_en_zh", "models/translate/en_zh"))
        self.sentiment_model_label_to_key = {}
        self.sentiment_model_key_to_label = {}

        
        # 设置清华镜像
        os.environ['HF_ENDPOINT'] = 'https://mirrors.tuna.tsinghua.edu.cn/huggingface-hub'
        os.environ['HF_HUB_OFFLINE'] = '0'

        self.df = None
        self.df_work = None
        self.emb = None
        self.labels = None
        self.centers = None
        self.k_scan = None
        self.cluster_keywords = None
        self.cluster_reps = None
        self._translators = {}
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        ensure_dir(self.output_dir)

        self._build_ui()
        self.log_queue = queue.Queue()
        self._start_log_pump()
        self._log("App started. Ready.")
        self._job_lock = threading.Lock()
        self._running = False

        # ====== graceful shutdown support ======
        self._threads = []         # 用来保存后台线程（daemon=False）
        self._closing = False      # 退出标记
        self._log_pump_id = None   # after 句柄（用于 cancel）
        
        # 🔥 关键：绑定窗口关闭事件
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.master.title("Review Analyzer")
        self.pack(fill="both", expand=True)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        # 两行工具栏
        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(6, 0))

        # ===== Row1：主流程 =====
        self.btn_import = ttk.Button(row1, text="导入文件（CSV/XLSX）", command=self.on_load_csv)
        self.btn_import.pack(side="left")

        self.btn_run_all = ttk.Button(row1, text="运行 Step1-5（全流程）", command=self.on_run_all)
        self.btn_run_all.pack(side="left", padx=8)

        self.btn_export = ttk.Button(row1, text="导出结果", command=self.on_export)
        self.btn_export.pack(side="left")

        self.btn_kplot = ttk.Button(row1, text="导出/显示K选择图", command=self.on_plot_k)
        self.btn_kplot.pack(side="left", padx=8)

        self.btn_compare = ttk.Button(row1, text="跨ASIN对比", command=self.on_asin_compare)
        self.btn_compare.pack(side="left", padx=8)

        self.btn_priority = ttk.Button(row1, text="优先级排序", command=self.on_priority)
        self.btn_priority.pack(side="left", padx=8)

        # 右侧状态
        self.status = tk.StringVar(value="Ready")
        ttk.Label(row1, textvariable=self.status).pack(side="right")

        # ===== Row2：语言 + 离线报告（已去掉 LLM 相关） =====
        ttk.Label(row2, text="文本语言:").pack(side="left")

        self.lang_var = tk.StringVar(value=getattr(self.cfg, "text_language", "en"))
        self.lang_box = ttk.Combobox(
            row2,
            textvariable=self.lang_var,
            values=["en", "zh_cn"],
            width=8,
            state="readonly"
        )
        self.lang_box.pack(side="left", padx=(6, 12))
        self.lang_box.bind("<<ComboboxSelected>>", self.on_language_changed)

        # 输出语言（翻译输出）
        ttk.Label(row2, text="输出语言:").pack(side="left")
        self.output_lang_label_to_key = {"原文": "none", "中文": "zh", "英文": "en"}
        self.output_lang_key_to_label = {v: k for k, v in self.output_lang_label_to_key.items()}
        cur_out = (getattr(self.cfg, "output_language", "none") or "none").strip().lower()
        self.output_lang_var = tk.StringVar(value=self.output_lang_key_to_label.get(cur_out, "原文"))
        self.output_lang_box = ttk.Combobox(
            row2,
            textvariable=self.output_lang_var,
            values=list(self.output_lang_label_to_key.keys()),
            width=8,
            state="readonly"
        )
        self.output_lang_box.pack(side="left", padx=(6, 12))
        self.output_lang_box.bind("<<ComboboxSelected>>", self.on_output_language_changed)

        # 情感模型选择
        ttk.Label(row2, text="情感模型:").pack(side="left")
        self.sentiment_model_var = tk.StringVar(value="")
        self.sentiment_model_box = ttk.Combobox(
            row2,
            textvariable=self.sentiment_model_var,
            values=[],
            width=20,
            state="readonly"
        )
        self.sentiment_model_box.pack(side="left", padx=(6, 12))
        self.sentiment_model_box.bind("<<ComboboxSelected>>", self.on_sentiment_model_changed)

        current_key = self._derive_sentiment_key_from_cfg()
        self._refresh_sentiment_model_options(self.cfg.text_language, select_key=current_key, save=False)

        # 离线报告按钮
        self.btn_report_offline = ttk.Button(row2, text="生成Word报告（离线）", command=self.on_report_offline)
        self.btn_report_offline.pack(side="left", padx=(0, 8))

        # 占位：让布局更美观
        ttk.Label(row2, text=" ").pack(side="left", expand=True)

        # 进度条
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=6)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_data = ttk.Frame(nb)
        self.tab_k = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        self.tab_log = ttk.Frame(nb)

        nb.add(self.tab_data, text="数据预览")
        nb.add(self.tab_k, text="选K结果")
        nb.add(self.tab_results, text="聚类结果")
        nb.add(self.tab_log, text="运行日志")

        # Data preview
        self.data_text = tk.Text(self.tab_data, height=18, wrap="none")
        self.data_text.pack(fill="both", expand=True)

        # K scan
        self.k_text = tk.Text(self.tab_k, height=18, wrap="none")
        self.k_text.pack(fill="both", expand=True)

        # Results
        self.res_text = tk.Text(self.tab_results, height=18, wrap="none")
        self.res_text.pack(fill="both", expand=True)

        # Log
        self.log_text = tk.Text(self.tab_log, height=18, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        # Bottom controls
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=6)
        row1 = ttk.Frame(bottom)
        row1.pack(fill="x")
        row2 = ttk.Frame(bottom)
        row2.pack(fill="x", pady=(4, 0))

        # ====== 阈值变量（从 cfg 读，确保 UI 打开就显示当前值）======
        self.star_th_var = tk.DoubleVar(value=float(getattr(self.cfg, "star_negative_threshold", 4.0)))
        self.conf_th_var = tk.DoubleVar(value=float(getattr(self.cfg, "sentiment_conf_threshold", 0.6)))

        # ====== FUSION 权重/阈值 ======
        self.fusion_w_star_var = tk.DoubleVar(value=float(getattr(self.cfg, "fusion_w_star", 1.0)))
        self.fusion_w_sent_var = tk.DoubleVar(value=float(getattr(self.cfg, "fusion_w_sent", 1.0)))
        self.fusion_keep_var = tk.DoubleVar(value=float(getattr(self.cfg, "fusion_keep_threshold", 1.0)))

        # ====== K 推荐参数 ======
        self.k_weight_var = tk.DoubleVar(value=float(getattr(self.cfg, "k_score_weight", 0.7)))
        self.k_penalty_th_var = tk.IntVar(value=int(getattr(self.cfg, "k_penalty_threshold", 12)))
        self.k_penalty_strength_var = tk.DoubleVar(value=float(getattr(self.cfg, "k_penalty_strength", 0.02)))

        self.auto_apply_k = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="扫描后自动应用推荐K", variable=self.auto_apply_k).pack(side="left", padx=10)

        ttk.Label(row1, text="聚类K:").pack(side="left")
        self.k_var = tk.IntVar(value=5)
        ttk.Spinbox(
            row1,
            from_=self.cfg.k_min,
            to=self.cfg.k_max,
            textvariable=self.k_var,
            width=5
        ).pack(side="left", padx=6)

        # NOTE: comment text was garbled; replaced with English placeholder.
        ttk.Label(row1, text="wK").pack(side="left", padx=(8, 2))
        self.k_weight_spin = ttk.Spinbox(
            row1, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.k_weight_var, width=4
        )
        self.k_weight_spin.pack(side="left")

        ttk.Label(row1, text="K>=").pack(side="left", padx=(5, 2))
        self.k_penalty_th_spin = ttk.Spinbox(
            row1, from_=2, to=50, increment=1,
            textvariable=self.k_penalty_th_var, width=4
        )
        self.k_penalty_th_spin.pack(side="left")

        ttk.Label(row1, text="penalty").pack(side="left", padx=(5, 2))
        self.k_penalty_strength_spin = ttk.Spinbox(
            row1, from_=0.0, to=0.5, increment=0.01,
            textvariable=self.k_penalty_strength_var, width=5
        )
        self.k_penalty_strength_spin.pack(side="left")

        # 仅分析负面
        self.only_negative = tk.BooleanVar(value=True)
        self.cb_only_negative = ttk.Checkbutton(
            row2,
            text="仅分析负面",
            variable=self.only_negative
        )
        self.cb_only_negative.pack(side="left", padx=10)

        # ===== 负面筛选策略（STAR / FUSION / SENTIMENT）=====
        ttk.Label(row2, text="负面判定:").pack(side="left", padx=(10, 4))

        self.negative_mode_var = tk.StringVar(
            value=getattr(self.cfg, "negative_mode", "STAR_ONLY")
        )

        self.negative_mode_box = ttk.Combobox(
            row2,
            textvariable=self.negative_mode_var,
            values=["STAR_ONLY", "FUSION", "SENTIMENT_ONLY"],
            width=14,
            state="readonly"
        )
        self.negative_mode_box.pack(side="left")
        self.negative_mode_box.bind("<<ComboboxSelected>>", self.on_negative_mode_changed)

        # =========================================================
        # 🔥 新增：参数微调控件 (Star / Conf / Fusion Weights)
        # =========================================================
        
        # 1. Star 阈值
        ttk.Label(row2, text="Star<=").pack(side="left", padx=(10, 2))
        self.star_th_spin = ttk.Spinbox(
            row2, from_=1.0, to=5.0, increment=0.5,
            textvariable=self.star_th_var, width=4
        )
        self.star_th_spin.pack(side="left")

        # 2. Conf 阈值
        ttk.Label(row2, text="Conf>=").pack(side="left", padx=(8, 2))
        self.conf_th_spin = ttk.Spinbox(
            row2, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.conf_th_var, width=4
        )
        self.conf_th_spin.pack(side="left")

        # 3. Fusion 参数 (W_Star, W_Sent, Keep)
        ttk.Label(row2, text="wStar").pack(side="left", padx=(8, 2))
        self.fusion_w_star_spin = ttk.Spinbox(
            row2, from_=0.0, to=5.0, increment=0.1,
            textvariable=self.fusion_w_star_var, width=4
        )
        self.fusion_w_star_spin.pack(side="left")

        ttk.Label(row2, text="wSent").pack(side="left", padx=(5, 2))
        self.fusion_w_sent_spin = ttk.Spinbox(
            row2, from_=0.0, to=5.0, increment=0.1,
            textvariable=self.fusion_w_sent_var, width=4
        )
        self.fusion_w_sent_spin.pack(side="left")

        ttk.Label(row2, text="Keep>=").pack(side="left", padx=(5, 2))
        self.fusion_keep_spin = ttk.Spinbox(
            row2, from_=0.0, to=5.0, increment=0.1,
            textvariable=self.fusion_keep_var, width=4
        )
        self.fusion_keep_spin.pack(side="left")

        # =========================================================

        self.btn_run_cluster = ttk.Button(
            row2,
            text="仅重跑 Step4-5",
            command=self.on_run_cluster_only
        )
        self.btn_run_cluster.pack(side="right", padx=(10, 0))

        # 🔥 绑定自动保存事件 (回车 OR 失去焦点)
        def _bind_save(widget, func):
            widget.bind("<Return>", func)
            widget.bind("<FocusOut>", func)

        _bind_save(self.star_th_spin, self.on_thresholds_changed)
        _bind_save(self.conf_th_spin, self.on_thresholds_changed)
        _bind_save(self.fusion_w_star_spin, self.on_thresholds_changed)
        _bind_save(self.fusion_w_sent_spin, self.on_thresholds_changed)
        _bind_save(self.fusion_keep_spin, self.on_thresholds_changed)
        _bind_save(self.k_weight_spin, self.on_thresholds_changed)
        _bind_save(self.k_penalty_th_spin, self.on_thresholds_changed)
        _bind_save(self.k_penalty_strength_spin, self.on_thresholds_changed)


    def _set_progress(self, cur, total, msg):
        """线程安全进度更新：后台线程调用也不会碰 Tk。"""
        def _apply():
            try:
                self.status.set(msg)
                self.progress["maximum"] = max(int(total), 1)
                self.progress["value"] = int(cur)
                self.master.update_idletasks()
            except Exception:
                pass

        self._ui(_apply)          # UI 更新回主线程
        self._log(f"[{cur}/{total}] {msg}")  # 日志用队列，线程安全

    def _start_log_pump(self):
        """
        将 log_queue 的内容刷到 Text 控件。
        关键：保存 after id，退出时 after_cancel，避免 mainloop 结束后仍然回调导致崩溃。
        """
        import queue

        def pump():
            # 如果正在关闭，就不再调度
            if getattr(self, "_closing", False):
                return

            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    if hasattr(self, "log_text") and self.log_text is not None:
                        try:
                            self.log_text.insert("end", msg + "\n")
                            self.log_text.see("end")
                        except Exception:
                            pass  # 窗口已销毁
            except queue.Empty:
                pass

            # 保存句柄，退出时可取消
            if not getattr(self, "_closing", False):
                try:
                    self._log_pump_id = self.after(120, pump)
                except Exception:
                    pass  # 窗口已销毁

        pump()

    def _log(self, msg: str):
        """线程安全日志：后台线程也可以调用。"""
        try:
            self.log_queue.put(str(msg))
        except Exception:
            pass  # 队列已关闭/销毁

    def _log_exception(self, e: Exception):
        """把异常堆栈写进日志，方便定位。"""
        self._log("❌ ERROR: " + str(e))
        self._log(traceback.format_exc())

    def _ui(self, fn, *args, **kwargs):
        """保证 UI 操作在主线程执行"""
        try:
            self.master.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass

    def _set_running(self, is_running: bool):
        """统一管理运行状态 + 按钮可用性（防连点）"""
        self._running = is_running
        state = "disabled" if is_running else "normal"
        try:
            # 这些按钮名字按你的实际变量名改一下（下面我也给你最通用写法）
            self.btn_run_all.config(state=state)
            self.btn_run_cluster.config(state=state)
            self.btn_export.config(state=state)
            self.btn_kplot.config(state=state)
            self.btn_compare.config(state=state)
            self.btn_priority.config(state=state)
            self.btn_report_offline.config(state=state)
            self.btn_import.config(state=state)
        except Exception:
            # 如果你没保存按钮引用，也没关系：至少防止线程重入
            pass

    def on_load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV / Excel Files", "*.csv *.xlsx")]
        )

        if not path:
            return

        try:
            # ========= 1) 先完整加载文件（不做任何列名假设） =========
            self.df = load_file(path, required_cols=None)

            # 清空中间状态
            self.df_work = None
            self.emb = None
            self.labels = None
            self.centers = None
            self.k_scan = None
            self.cluster_keywords = None
            self.cluster_reps = None

            # ========= 2) 自动识别列名（关键新增） =========
            self._auto_map_fields(self.df)

            fm = self.cfg.field_map  # 识别后的映射结果

            # ========= 3) 统一内部字段（后续流程只用 _xxx） =========
            self.df["_text"] = self.df[fm["text"]]

            # star / score（可选）
            star_col = fm.get("star")
            if star_col and star_col in self.df.columns:
                self.df["_score"] = self.df[star_col]
            else:
                self.df["_score"] = None

            # asin / group（可选）
            asin_col = fm.get("asin")
            if asin_col and asin_col in self.df.columns:
                self.df["_group"] = self.df[asin_col]
            else:
                self.df["_group"] = None

            # id（可选）
            id_col = fm.get("id")
            if id_col and id_col in self.df.columns:
                self.df["_id"] = self.df[id_col]
            else:
                self.df["_id"] = self.df.index.astype(str)

            # ========= 4) 预览 & 日志 =========
            self.data_text.delete("1.0", "end")
            self.data_text.insert("end", f"Loaded: {path}\n")
            self.data_text.insert("end", f"Rows: {len(self.df)}\n\n")

            self._log(f"Loaded file: {path}")
            self._log(f"Total rows: {len(self.df)}")

            empty_text = self.df["_text"].astype(str).str.strip().eq("").sum()
            self._log(f"Empty text rows: {empty_text}")

            self.data_text.insert("end", self.df.head(20).to_string(index=False))

            self.status.set("Data loaded & auto-mapped")

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _run_in_thread(self, fn, start_msg="Task started..."):
        """
        统一线程安全执行器
        🔥 修复：确保线程正确清理
        """
        import threading

        # 正在关闭就别再启动任务了
        if getattr(self, "_closing", False):
            return

        # 防连点（主线程）
        if getattr(self, "_running", False):
            self._log("⚠️ 正在运行中，请等待当前任务完成后再操作。")
            return

        def ui(callable_):
            # 如果正在关闭，不再触发 UI 更新
            if getattr(self, "_closing", False):
                return
            try:
                self.master.after(0, callable_)
            except Exception:
                pass  # 窗口已销毁，忽略

        def runner():
            try:
                ui(lambda: self._set_running(True))
                ui(lambda: self.status.set("Running"))
                ui(lambda: self._log(f"▶ {start_msg}"))

                fn()  # 后台执行：只做计算/IO/API（不要直接碰 Tk）

                ui(lambda: self.status.set("Done"))
                ui(lambda: self._log("✅ Task finished."))

            except Exception as e:
                ui(lambda: self.status.set("Error"))
                ui(lambda: self._log_exception(e))
                ui(lambda: messagebox.showerror("错误", str(e)))
            finally:
                ui(lambda: self._set_running(False))

        t = threading.Thread(target=runner, daemon=False)  # daemon=False 可以 join
        # 🔥 记录线程，退出时 join
        if not hasattr(self, "_threads"):
            self._threads = []
        self._threads.append(t)
        t.start()
        
    def on_run_all(self):
        """
        全流程：Step1–5（仅计算，不导出、不出图、不生成报告）
        关键修复：
        - 在主线程一次性读取 Tk 变量（only_negative/auto_apply_k/k_var）
        - 后台线程 job() / _pipeline_all() 内禁止再 .get() / .set()
        - 自动应用推荐K：用 _ui 回主线程 set
        - 补齐 self.cluster_reps 等状态变量，保证导出/报告可用
        """
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("提示", "请先导入数据文件")
            return

        # ✅ 主线程一次性读取 Tk 变量（非常关键：后台线程禁止再 get/set）
        try:
            only_negative_flag = bool(self.only_negative.get())
        except Exception:
            only_negative_flag = False

        try:
            auto_apply_flag = bool(self.auto_apply_k.get())
        except Exception:
            auto_apply_flag = False

        try:
            k_used_ui = int(self.k_var.get())
        except Exception:
            k_used_ui = None

        self.artifacts_dirty = True

        # UI：忙碌提示放主线程
        try:
            self._busy(True, "全流程分析中...")
        except Exception:
            pass

        def job():
            try:
                self._pipeline_all(
                    only_negative_flag=only_negative_flag,
                    auto_apply_flag=auto_apply_flag,
                    k_used_ui=k_used_ui
                )

                # UI 结束（必须主线程）
                def done():
                    try:
                        self._busy(False, "分析完成")
                    except Exception:
                        pass
                    messagebox.showinfo(
                        "完成",
                        "Step1–5 已完成。\n\n"
                        "请按需点击：\n"
                        "• 导出结果\n"
                        "• 导出 K 选择图\n"
                        "• 跨 ASIN 对比\n"
                        "• 优先级排序\n"
                        "• 生成 Word 报告"
                    )
                    self._log("Step1–5 completed. Waiting for user actions.")

                self._ui(done)

            except Exception as e:
                # 失败也要解除 busy（主线程）
                self._ui(lambda: self._busy(False, "就绪"))
                raise

        self._run_in_thread(job, "Running Step1-5 (full pipeline)...")

    def on_run_cluster_only(self):
        """
        仅重跑 Step4–5（聚类 + 关键词/代表评论）
        修复点：
        1) 全部走 _run_in_thread
        2) 补齐 self.cluster_reps，否则报告按钮会误判
        3) 重跑后标记 artifacts_dirty=False（数据已齐全）
        """
        if self.df_work is None:
            messagebox.showwarning("提示", "请先运行 Step1–5")
            return
        if self.emb is None:
            messagebox.showwarning("提示", "Embedding 不存在，请先运行 Step1–5")
            return

        # UI：忙碌提示放主线程
        try:
            self._busy(True, "重新聚类中 (Step4–5)...")
        except Exception:
            pass

        # 重跑意味着旧产物作废
        self.artifacts_dirty = True

        def job():
            self.k_used = int(self.k_var.get())
            self._log(f"Re-run clustering with K={self.k_used}")

            # Step4
            self.labels, self.centers = fit_kmeans(
                self.emb,
                k=self.k_used,
                random_state=self.cfg.random_state
            )
            self.df_work["cluster_id"] = self.labels

            # Step5: keywords
            self.cluster_keywords = top_keywords_by_cluster(
                self.df_work[self.cfg.field_map["text"]].tolist(),
                self.labels,
                top_n=self.cfg.top_keywords,
                language=self.cfg.text_language
            )

            # Step5: reps
            reps_dict = top_representatives(
                self.emb,
                self.labels,
                self.centers,
                top_n=self.cfg.top_representatives
            )

            # ★关键：报告/导出依赖这个
            self.cluster_reps = reps_dict

            rows = []
            for cid, idx_list in reps_dict.items():
                for rank, idx in enumerate(idx_list, 1):
                    row = self.df_work.iloc[idx].to_dict()
                    row["cluster_id"] = cid
                    row["rank_in_cluster"] = rank
                    rows.append(row)
            self.reps_df = pd.DataFrame(rows)

            # summary
            cluster_sizes = pd.Series(self.labels).value_counts().sort_index()
            self.cluster_summary = pd.DataFrame({
                "cluster_id": cluster_sizes.index,
                "cluster_size": cluster_sizes.values,
                "ratio": cluster_sizes.values / len(self.labels),
                "keywords": [", ".join(self.cluster_keywords.get(c, [])) for c in cluster_sizes.index]
            }).sort_values("ratio", ascending=False)

            # 标记：当前产物已准备好
            self.artifacts_dirty = False

            def done():
                try:
                    self._busy(False, "聚类完成")
                except Exception:
                    pass
                messagebox.showinfo(
                    "完成",
                    f"已使用 K={self.k_used} 重新完成聚类。\n\n"
                    "请按需点击：\n"
                    "• 导出结果\n"
                    "• 跨 ASIN 对比\n"
                    "• 优先级排序\n"
                    "• 生成 Word 报告"
                )

            self._ui(done)

        self._run_in_thread(job, "Re-running Step4-5 (cluster only)...")

    def _pipeline_all(self, only_negative_flag: bool, auto_apply_flag: bool, k_used_ui: int | None):
        """
        后台线程运行的全流程逻辑（Step1-5）
        """

        # 统一拷贝一份，用于最终导出"带情感标签"的全量数据
        df2 = self.df.copy()

        # ---------- Step1: sentiment（默认启用；失败自动降级） ----------
        sent_model = getattr(self.cfg, "sentiment_model", None)

        # 先给 sentiment 列一个默认值，保证后续逻辑稳
        df2["sentiment"] = np.nan

        if sent_model:
            try:
                self._set_progress(0, 1, "Loading sentiment model...")
                
                # 🔥 修复：这里不要用 embedding_model！
                sa = SentimentAnalyzer(
                    model_name=sent_model,  # ← 用 sentiment_model
                    batch_size=self.cfg.sentiment_batch_size,
                    max_chars=self.cfg.sentiment_max_chars
                )

                # 注意：这里用你内部统一列 _text
                sent, conf = SentimentAnalyzer.predict_sentiment_aligned(
                    sa, df2, "_text", progress=self._set_progress, return_conf=True
                )
                df2["sentiment"] = sent
                df2["sentiment_conf"] = conf

                valid_n = int(pd.Series(sent).notna().sum())
                self._log(f"✅ Sentiment done. valid rows={valid_n}, total={len(df2)}")

            except Exception as e:
                # 情感模型加载/推理失败：不终止，全量继续 + 允许仅负面走星级兜底
                self._log("⚠️ Sentiment failed, fallback to star-based negative filter if needed.")
                self._log_exception(e)
        else:
            # 不禁用，仅提示
            self._log("⚠️ sentiment_model is empty. 'only negative' will fallback to star-based filter if possible.")

        # ---------- Step2: filter（统一负面过滤策略） ----------
        # 🔥🔥🔥 关键：这段代码必须在 Step1 的 if 块之外！
        if only_negative_flag:
            # star 列：优先用 field_map，其次用内部 _score
            star_col = None
            try:
                star_col = (self.cfg.field_map.get("star") or "").strip()
            except Exception:
                star_col = None
            if not star_col or star_col not in df2.columns:
                star_col = "_score"

            # sentiment 列：你这里固定叫 sentiment
            sentiment_col = "sentiment" if "sentiment" in df2.columns else None

            # 置信度列：当前你的 df2 里没有 sentiment_conf（先传 None，后续升级模型输出时补）
            sentiment_conf_col = "sentiment_conf" if "sentiment_conf" in df2.columns else None

            mode = getattr(self.cfg, "negative_mode", "STAR_ONLY")
            star_th = float(getattr(self.cfg, "star_negative_threshold", 4.0))
            conf_th = float(getattr(self.cfg, "sentiment_conf_threshold", 0.6))

            # ✅ 新增：fusion 参数
            w_star = float(getattr(self.cfg, "fusion_w_star", 1.0))
            w_sent = float(getattr(self.cfg, "fusion_w_sent", 1.0))
            keep_th = float(getattr(self.cfg, "fusion_keep_threshold", 1.0))

            dfw = self.apply_negative_filter(
                df2,
                star_col=star_col,
                sentiment_col=sentiment_col,
                sentiment_conf_col=sentiment_conf_col,
                mode=mode,
                star_threshold=star_th,
                conf_threshold=conf_th,
                w_star=w_star,
                w_sent=w_sent,
                fusion_keep_threshold=keep_th
            )

            self._log(
                f"only_negative=True | mode={mode} | star_th={star_th} | conf_th={conf_th} | "
                f"w_star={w_star} | w_sent={w_sent} | keep={keep_th} | rows={len(dfw)}"
            )

            # 安全兜底：如果筛完变成 0 行，自动降级为 ALL（避免用户误操作直接崩）
            if len(dfw) == 0:
                self._log("⚠️ 负面过滤后为 0 行，自动降级为 ALL（不做过滤）。")
                dfw = df2.reset_index(drop=True)

        else:
            dfw = df2.reset_index(drop=True)

        self._log(f"only_negative_flag = {only_negative_flag}")
        self._log(f"Rows after filter: {len(dfw)}")

        if len(dfw) < 30:
            raise ValueError(f"过滤后样本太少（{len(dfw)}）。建议取消仅负面或调整阈值/策略。")

        # 🔥🔥🔥 关键：赋值给 self.df_work
        self.df_work = dfw
        
        self._log(f"✅ df_work 已赋值，shape={self.df_work.shape}")

        # ---------- Step3: embedding ----------
        self._set_progress(0, 1, "Loading embedding model...")
        
        # 🔥 诊断：打印所有关键变量
        self._log(f"🔍 DEBUG - df_work 信息:")
        self._log(f"  - shape: {self.df_work.shape if self.df_work is not None else 'None'}")
        self._log(f"  - columns: {list(self.df_work.columns) if self.df_work is not None else 'None'}")
        self._log(f"  - _text 类型: {type(self.df_work['_text']) if self.df_work is not None and '_text' in self.df_work.columns else 'N/A'}")
        
        if self.df_work is not None and '_text' in self.df_work.columns:
            texts_raw = self.df_work["_text"]
            self._log(f"  - _text 前3条: {texts_raw.head(3).tolist()}")
            self._log(f"  - _text.isnull().sum(): {texts_raw.isnull().sum()}")
        
        self._log(f"🔍 DEBUG - cfg 信息:")
        self._log(f"  - embedding_model: {repr(self.cfg.embedding_model)}")
        self._log(f"  - embedding_batch_size: {repr(self.cfg.embedding_batch_size)}")
        self._log(f"  - type(embedding_batch_size): {type(self.cfg.embedding_batch_size)}")
        
        model_name = getattr(self.cfg, "embedding_model", None)
        batch_size_raw = getattr(self.cfg, "embedding_batch_size", None)
        
        self._log(f"🔍 DEBUG - getattr 结果:")
        self._log(f"  - model_name: {repr(model_name)}")
        self._log(f"  - batch_size_raw: {repr(batch_size_raw)}")
        
        if not model_name:
            raise RuntimeError(
                "embedding_model 为空(None)。\n"
                "通常是 settings.json 覆盖导致。\n"
                "解决:删除 settings.json 或在 settings.json 中设置 embedding_model='models/embedding'。"
            )

        # 🔥 关键防御：batch_size
        if batch_size_raw is None or not isinstance(batch_size_raw, (int, float)) or batch_size_raw <= 0:
            self._log(f"⚠️ embedding_batch_size 无效: {repr(batch_size_raw)}，使用默认值 64")
            batch_size_safe = 64
        else:
            batch_size_safe = int(batch_size_raw)
        
        self._log(f"🔍 准备创建 Embedder:")
        self._log(f"  - model_name: {repr(model_name)}")
        self._log(f"  - batch_size: {repr(batch_size_safe)}")
        
        # cache 路径
        tag = "neg" if only_negative_flag else "all"
        emb_cache = os.path.join(self.output_dir, f"embeddings_{tag}_{len(self.df_work)}.npy")
        
        # 🔥 在创建 Embedder 之前打印
        self._log("🔍 即将执行: emb = Embedder(...)")
        
        try:
            emb = Embedder(
                model_name=model_name,
                batch_size=batch_size_safe
            )
            self._log("✅ Embedder 创建成功")
        except Exception as e:
            self._log(f"❌ Embedder 创建失败!")
            self._log(f"  - Exception type: {type(e)}")
            self._log(f"  - Exception message: {str(e)}")
            raise
        
        # 🔥 准备文本数据
        self._log("🔍 准备 encode 的文本数据:")
        
        if self.df_work is None:
            raise RuntimeError("df_work 为 None!")
        
        if "_text" not in self.df_work.columns:
            raise RuntimeError(f"df_work 缺少 _text 列! 当前列: {list(self.df_work.columns)}")
        
        texts_series = self.df_work["_text"]
        self._log(f"  - texts_series 类型: {type(texts_series)}")
        self._log(f"  - texts_series 长度: {len(texts_series) if texts_series is not None else 'None'}")
        
        if texts_series is None:
            raise RuntimeError("df_work['_text'] 为 None!")
        
        texts_list = texts_series.fillna("").astype(str).tolist()
        self._log(f"  - texts_list 类型: {type(texts_list)}")
        self._log(f"  - texts_list 长度: {len(texts_list) if texts_list is not None else 'None'}")
        self._log(f"  - texts_list 前3条: {texts_list[:3] if texts_list else 'None'}")
        
        if texts_list is None:
            raise RuntimeError("texts_list 为 None!")
        
        if len(texts_list) == 0:
            raise RuntimeError("texts_list 为空列表!")
        
        self._log("🔍 即将执行: emb.encode(...)")
        
        try:
            self.emb = emb.encode(
                texts_list,
                cache_path=emb_cache,
                progress=self._set_progress
            )
            self._log("✅ emb.encode() 成功")
        except Exception as e:
            self._log(f"❌ emb.encode() 失败!")
            self._log(f"  - Exception type: {type(e)}")
            self._log(f"  - Exception message: {str(e)}")
            raise

        self._log(f"Embedding model: {self.cfg.embedding_model}")
        try:
            self._log(f"Embedding shape: {self.emb.shape}")
        except Exception:
            pass

        # ---------- Step4: scan k ----------
        self._set_progress(0, 1, "Scanning k...")
        self.k_scan = scan_k(
            self.emb,
            self.cfg.k_min,
            self.cfg.k_max,
            random_state=self.cfg.random_state
        )
        self._log(f"K scan done. range=[{self.cfg.k_min},{self.cfg.k_max}]")

        rec = recommend_k(
                self.k_scan.k_to_inertia,
                self.k_scan.k_to_silhouette,
                weight=getattr(self.cfg, "k_score_weight", 0.7),
                penalty_threshold=getattr(self.cfg, "k_penalty_threshold", 12),
                penalty_strength=getattr(self.cfg, "k_penalty_strength", 0.02)
            )
        self.k_best = int(rec.best_k)
        score = self.k_scan.k_to_silhouette.get(rec.best_k, float("nan"))
        self._log(f"Recommended K by {rec.method} = {self.k_best} (score={rec.score})")

        # 你原来会渲染 K 扫描结果（注意：此函数内部必须线程安全）
        try:
            self._render_k_scan()
        except Exception as e:
            self._log("⚠️ _render_k_scan failed (ignored).")
            self._log_exception(e)

        # ---------- 自动应用推荐K（如果勾选） ----------
        # ✅ 这里不能 self.k_var.set，必须回主线程
        if auto_apply_flag:
            self._ui(lambda: self.k_var.set(self.k_best))

            # 如果自动应用推荐K，我们也同步把 k_used_ui 改成 k_best（用于本次聚类）
            k_for_cluster = self.k_best
        else:
            # 不自动应用：用用户点击按钮那一刻的 K（主线程读出来的）
            k_for_cluster = int(k_used_ui) if k_used_ui is not None else self.k_best

        self.k_used = int(k_for_cluster)
        self._log(f"Clustering K used = {self.k_used}")

        # ---------- Step4/5: fit kmeans + keywords + representatives ----------
        # 这里不再调用你原来的 _pipeline_cluster_only（它大概率内部也在 get/set Tk 变量）
        self._set_progress(0, 1, f"Clustering with K={self.k_used}...")

        self.labels, self.centers = fit_kmeans(
            self.emb,
            k=self.k_used,
            random_state=self.cfg.random_state
        )

        # 写回 cluster_id（后续导出/对比/报告都依赖）
        self.df_work["cluster_id"] = self.labels

        self._set_progress(0, 1, "Extracting keywords...")
        self.cluster_keywords = top_keywords_by_cluster(
            self.df_work["_text"].tolist(),
            self.labels,
            top_n=self.cfg.top_keywords,
            language=self.cfg.text_language
        )

        self._set_progress(0, 1, "Selecting representatives...")
        reps_dict = top_representatives(
            self.emb,
            self.labels,
            self.centers,
            top_n=self.cfg.top_representatives
        )

        # ★关键：报告/导出依赖这个
        self.cluster_reps = reps_dict

        # reps_df
        rows = []
        for cid, idx_list in reps_dict.items():
            for rank, idx in enumerate(idx_list, 1):
                row = self.df_work.iloc[int(idx)].to_dict()
                row["cluster_id"] = int(cid)
                row["rank_in_cluster"] = int(rank)
                rows.append(row)
        self.reps_df = pd.DataFrame(rows)

        # cluster_summary（导出/报告用）
        cluster_sizes = pd.Series(self.labels).value_counts().sort_index()
        self.cluster_summary = pd.DataFrame({
            "cluster_id": cluster_sizes.index,
            "cluster_size": cluster_sizes.values,
            "ratio": cluster_sizes.values / len(self.labels),
            "keywords": [", ".join(self.cluster_keywords.get(int(c), [])) for c in cluster_sizes.index]
        }).sort_values("ratio", ascending=False)

        # ---------- Export（保留你原来的 reviews_with_sentiment.csv） ----------
        out_csv = os.path.join(self.output_dir, "reviews_with_sentiment.csv")
        save_csv(df2, out_csv)
        self._log(f"Exported: {out_csv}")

        # 标记：当前产物已准备好
        self.artifacts_dirty = False

    def _is_negative_by_star(self, df: pd.DataFrame) -> pd.Series:
        """
        情感模型不可用时，用星级兜底“仅负面”：
        - 优先用 _score（你在 on_load_csv 已映射）
        - 次选用 Star
        规则：<=2 视为负面
        """
        s = None
        if "_score" in df.columns:
            s = df["_score"]
        elif "Star" in df.columns:
            s = df["Star"]

        if s is None:
            return pd.Series([False] * len(df), index=df.index)

        s_num = pd.to_numeric(s, errors="coerce")
        return s_num.le(2)

    def _pipeline_cluster_only(self):
        if self.emb is None:
            raise ValueError("embeddings 为空：请先运行到 Step3（Embedding）或直接运行全流程。")
        if self.df_work is None:
            raise ValueError("df_work 为空：请先运行全流程（至少完成过滤 Step2），或取消“仅负面”后重试。")

        k = int(self.k_var.get())
        self._set_progress(0, 1, f"Clustering k={k} ...")

        # Step4: KMeans
        labels, centers = fit_kmeans(
            self.emb,
            k=k,
            random_state=self.cfg.random_state
        )

        self.labels = labels
        self.centers = centers

        # 写回 df_work
        self.df_work["cluster_id"] = labels

        # 每个 cluster 的样本数
        sizes = self.df_work["cluster_id"].value_counts().sort_index().to_dict()
        self._log(f"Cluster sizes: {sizes}")

        # Step5: keywords（关键：接入语言）
        self._set_progress(0, 1, "Extracting keywords...")

        lang = getattr(self.cfg, "text_language", "en")  # 来自 UI + settings.json
        self.cluster_keywords = top_keywords_by_cluster(
            self.df_work["_text"].tolist(),
            self.labels,
            top_n=self.cfg.top_keywords,
            language=lang
        )

        self._log(
            f"Keywords extracted for clusters (language={lang}): "
            + ", ".join(str(cid) for cid in self.cluster_keywords.keys())
        )

        # Step5: representatives
        self._set_progress(0, 1, "Finding representatives...")
        self.cluster_reps = top_representatives(
            self.emb,
            self.labels,
            self.centers,
            top_n=self.cfg.top_representatives
        )

        # Robustness
        self._set_progress(0, 1, "Robustness (bootstrap ARI)...")
        stab = clustering_stability(
            self.emb,
            k=k,
            runs=5,
            random_state=self.cfg.random_state
        )

        # Render
        self._render_results(stab)

    def _render_k_scan(self):
        self.k_text.delete("1.0", "end")
        self.k_text.insert("end", "k\tinertia(SSE)\tsilhouette\n")
        for k in range(self.cfg.k_min, self.cfg.k_max + 1):
            sse = self.k_scan.k_to_inertia.get(k, None)
            sil = self.k_scan.k_to_silhouette.get(k, None)
            self.k_text.insert("end", f"{k}\t{(sse or 0):.2f}\t\t{(sil or 0):.4f}\n")

    def _render_results(self, stability: dict):
        self.res_text.delete("1.0", "end")
        self.res_text.insert("end", f"Filtered rows: {len(self.df_work)}\n")
        self.res_text.insert(
            "end",
            f"Stability ARI (bootstrap): mean={stability['ari_mean']:.3f}, "
            f"min={stability['ari_min']:.3f}, max={stability['ari_max']:.3f}\n\n"
        )

        for c in sorted(self.cluster_keywords.keys()):
            self.res_text.insert("end", f"=== Cluster {c} ===\n")
            kws = ", ".join(self.cluster_keywords[c])
            self.res_text.insert("end", f"Keywords: {kws}\n")

            reps = self.cluster_reps.get(c, [])
            self.res_text.insert("end", "Representatives:\n")
            for idx in reps:
                row = self.df_work.iloc[idx]
                gid = row["_group"] if row["_group"] is not None else "-"
                score = row["_score"] if row["_score"] is not None else "-"
                text = str(row["_text"])[:180]
                self.res_text.insert(
                    "end",
                    f"- ({gid}, Score={score}) {text}...\n"
                )
            self.res_text.insert("end", "\n")

    def on_plot_k(self):
        if self.k_scan is None:
            messagebox.showwarning("Warning", "Please run the full pipeline (at least through K scan).")
            return

        def job():
            self._set_progress(0, 1, "Generating K selection plot...")

            rec = recommend_k(
                self.k_scan.k_to_inertia,
                self.k_scan.k_to_silhouette,
                weight=getattr(self.cfg, "k_score_weight", 0.7),
                penalty_threshold=getattr(self.cfg, "k_penalty_threshold", 12),
                penalty_strength=getattr(self.cfg, "k_penalty_strength", 0.02)
            )
            best_k = int(rec.best_k)

            labels = {
                "x_label": "K",
                "y1_label": "WCSS / Inertia",
                "y2_label": "Silhouette Score",
                "line1_label": "WCSS/Inertia (Elbow) - solid (blue)",
                "line2_label": "Silhouette Score - dashed (orange)",
                "title": "Optimal K Selection (Elbow & Silhouette)",
                "title_with_k": f"Optimal K Selection (Recommended K={best_k})",
                "vline_label": f"Recommended K = {best_k} - vertical (green)",
            }

            out_lang = self._get_output_language()
            if out_lang in {"zh", "en"}:
                keys = list(labels.keys())
                vals = self._translate_texts_to(list(labels.values()), src_lang="en", tgt_lang=out_lang)
                labels = dict(zip(keys, vals))

            png_path = os.path.join(self.output_dir, "k_selection.png")
            plot_k_curves(
                self.k_scan.k_to_inertia,
                self.k_scan.k_to_silhouette,
                recommended_k=best_k,
                save_path=png_path,
                lang=self.cfg.text_language,
                labels=labels
            )

            def done():
                if self.auto_apply_k.get():
                    try:
                        self.k_var.set(best_k)
                    except Exception:
                        pass

                cur_k = None
                try:
                    cur_k = int(self.k_var.get())
                except Exception:
                    cur_k = best_k

                self._set_progress(1, 1, "K selection plot ready.")
                messagebox.showinfo(
                    "Done",
                    f"K plot exported:\n{png_path}\n\n"
                    f"Recommended K={best_k}\nCurrent K={cur_k}\n\n"
                    "If you want to update clustering, re-run Step4-5 with current K."
                )

            self._ui(done)

        self._run_in_thread(job, "Generating K selection plot...")
    def on_asin_compare(self):
        """
        Cross-ASIN outputs:
        - ASINCluster heatmap + csv
        - Attribute taxonomy
        - ASINAttribute share heatmap
        - ASINAttribute pain heatmap
        - Opportunity insights
        - Export: asin_attribute_matrix.xlsx
        """
        import os
        import pandas as pd
        from tkinter import messagebox

        if self.df_work is None or "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("Warning", "Please finish clustering (Step4-5) first.")
            return

        def job():
            self._set_progress(0, 1, "Generating cross-ASIN outputs...")

            try:
                df_clustered = self.df_work.copy()

                asin_col = None
                try:
                    asin_col = (self.cfg.field_map.get("asin") or "").strip()
                except Exception:
                    asin_col = None
                if not asin_col:
                    asin_col = "_group"

                if asin_col not in df_clustered.columns:
                    self._ui(lambda: messagebox.showwarning(
                        "Warning",
                        f"Missing ASIN column (current: {asin_col}).\n"
                        "Please check your data or field_map."
                    ))
                    return

                star_col = None
                try:
                    star_col = (self.cfg.field_map.get("star") or "").strip()
                except Exception:
                    star_col = None
                if not star_col or star_col not in df_clustered.columns:
                    star_col = "_score" if "_score" in df_clustered.columns else ("Star" if "Star" in df_clustered.columns else None)

                if star_col is None or star_col not in df_clustered.columns:
                    self._ui(lambda: messagebox.showwarning("Warning", "Missing Star/_score column; cannot compute pain/priority."))
                    return

                cluster_keywords = getattr(self, "cluster_keywords", None)
                if not cluster_keywords:
                    cs = getattr(self, "cluster_summary", None)
                    if cs is not None and hasattr(cs, "columns") and ("cluster_id" in cs.columns) and ("keywords" in cs.columns):
                        cluster_keywords = dict(zip(cs["cluster_id"].tolist(), cs["keywords"].tolist()))

                if not cluster_keywords:
                    self._ui(lambda: messagebox.showwarning("Warning", "Missing cluster_keywords. Run Step5 first."))
                    return

                out_dir = getattr(self, "output_dir", None) or os.path.join(os.getcwd(), "outputs")
                os.makedirs(out_dir, exist_ok=True)

                pivot_cluster = asin_cluster_percent(df_clustered, asin_col=asin_col, cluster_col="cluster_id")

                asin_label_map = None
                out_lang = self._get_output_language()
                if out_lang in {"zh", "en"} and self._translation_needed():
                    asin_vals = [str(v) for v in pivot_cluster.index.tolist()]
                    asin_trans = self._translate_labels(asin_vals)
                    asin_label_map = dict(zip(asin_vals, asin_trans))
                    pivot_cluster = pivot_cluster.copy()
                    pivot_cluster.index = [asin_label_map.get(str(v), str(v)) for v in pivot_cluster.index]

                self.asin_pivot = pivot_cluster

                old_png = os.path.join(out_dir, "asin_cluster_percent_heatmap.png")
                title_cluster = "ASIN  Cluster Share (%)"
                if out_lang in {"zh", "en"}:
                    title_cluster = self._translate_texts_to([title_cluster], "en", out_lang)[0]

                labels = {"x_label": "Cluster ID", "y_label": "ASIN"}
                if out_lang in {"zh", "en"}:
                    keys = list(labels.keys())
                    vals = self._translate_texts_to(list(labels.values()), src_lang="en", tgt_lang=out_lang)
                    labels = dict(zip(keys, vals))

                fig = plot_heatmap(
                    pivot_cluster,
                    save_path=old_png,
                    title=title_cluster,
                    lang=self.cfg.text_language,
                    labels=labels
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

                old_csv = os.path.join(out_dir, "asin_cluster_percent.csv")
                pivot_cluster.round(2).to_csv(old_csv, encoding="utf-8-sig")
                self.asin_heatmap_png = old_png

                from core.insights import (
                    build_attribute_taxonomy,
                    asin_attribute_share,
                    asin_attribute_pain,
                    opportunity_insights,
                )

                taxonomy_df = build_attribute_taxonomy(cluster_keywords, topn=3)
                if self._translation_needed() and "attribute_name" in taxonomy_df.columns:
                    taxonomy_df = taxonomy_df.copy()
                    taxonomy_df["attribute_name"] = self._translate_series(taxonomy_df["attribute_name"])

                share_pivot = asin_attribute_share(
                    df_clustered,
                    asin_col=asin_col,
                    cluster_col="cluster_id",
                    taxonomy_df=taxonomy_df
                )
                pain_pivot = asin_attribute_pain(
                    df_clustered,
                    asin_col=asin_col,
                    cluster_col="cluster_id",
                    star_col=star_col,
                    taxonomy_df=taxonomy_df
                )

                if asin_label_map:
                    share_pivot = share_pivot.copy()
                    share_pivot.index = [asin_label_map.get(str(v), str(v)) for v in share_pivot.index]
                    pain_pivot = pain_pivot.copy()
                    pain_pivot.index = [asin_label_map.get(str(v), str(v)) for v in pain_pivot.index]

                opp_df = opportunity_insights(pain_pivot, topk=15)
                if opp_df is not None and len(opp_df) > 0 and "attribute" in opp_df.columns and self._translation_needed():
                    opp_df = opp_df.copy()
                    opp_df["attribute"] = self._translate_series(opp_df["attribute"])

                out_xlsx = os.path.join(out_dir, "asin_attribute_matrix.xlsx")
                with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                    taxonomy_df.to_excel(writer, sheet_name="attribute_taxonomy", index=False)
                    share_pivot.to_excel(writer, sheet_name="asin_attribute_share")
                    pain_pivot.to_excel(writer, sheet_name="asin_attribute_pain")
                    (opp_df if opp_df is not None else pd.DataFrame()).to_excel(writer, sheet_name="opportunity_top", index=False)

                import numpy as np
                import matplotlib.pyplot as plt

                def _plot_heatmap(pivot_df: pd.DataFrame, title: str, out_png: str):
                    if pivot_df is None or pivot_df.shape[0] == 0 or pivot_df.shape[1] == 0:
                        return
                    apply_matplotlib_style(self.cfg.text_language)
                    fig = plt.figure(figsize=(12, max(4, 0.35 * pivot_df.shape[0])))
                    ax = fig.add_subplot(111)
                    data = pivot_df.values.astype(float)
                    im = ax.imshow(data, aspect="auto")
                    ax.set_title(title)
                    ax.set_yticks(np.arange(pivot_df.shape[0]))
                    ax.set_yticklabels(pivot_df.index.astype(str).tolist(), fontsize=8)
                    ax.set_xticks(np.arange(pivot_df.shape[1]))
                    ax.set_xticklabels(pivot_df.columns.astype(str).tolist(), rotation=45, ha="right", fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
                    fig.tight_layout()
                    fig.savefig(out_png, dpi=300)
                    plt.close(fig)

                png_share = os.path.join(out_dir, "asin_attribute_share.png")
                png_pain  = os.path.join(out_dir, "asin_attribute_pain.png")
                title_share = "ASIN  Attribute Share (%)"
                title_pain = "ASIN  Attribute Pain (Priority)"
                if out_lang in {"zh", "en"}:
                    title_share, title_pain = self._translate_texts_to(
                        [title_share, title_pain],
                        src_lang="en",
                        tgt_lang=out_lang
                    )

                _plot_heatmap(share_pivot, title_share, png_share)
                _plot_heatmap(pain_pivot, title_pain, png_pain)

                self._set_progress(1, 1, "Cross-ASIN outputs ready.")
                self._ui(lambda: messagebox.showinfo(
                    "Done",
                    "Cross-ASIN outputs saved:\n"
                    f"[Old] Heatmap: {old_png}\n"
                    f"[Old] CSV: {old_csv}\n\n"
                    f"[New] Excel: {out_xlsx}\n"
                    f"[New] Share heatmap: {png_share}\n"
                    f"[New] Pain heatmap: {png_pain}\n"
                ))

            except Exception as e:
                self._ui(lambda: messagebox.showwarning("Warning", f"Cross-ASIN generation failed: {e}"))

        self._run_in_thread(job, "Generating cross-ASIN outputs...")
    def on_priority(self):
        if self.df_work is None or "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("Warning", "Please finish clustering (Step4-5) first.")
            return

        def job():
            self._set_progress(0, 1, "Generating priority outputs...")
            df = self.df_work.copy()

            group_col = None
            for cand in ["ASIN", "asin", "", "place", "group"]:
                if cand in df.columns:
                    group_col = cand
                    break

            if group_col is None:
                df["__GROUP__"] = "ALL"
                group_col = "__GROUP__"

            try:
                star_col = (self.cfg.field_map.get("star") or "").strip()
            except Exception:
                star_col = None

            if not star_col or star_col not in df.columns:
                star_col = "_score"

            if star_col not in df.columns:
                self._ui(lambda: messagebox.showwarning(
                    "Warning",
                    f"Missing rating column (current: {star_col}).\n"
                    "Please check Star/Rating or field_map."
                ))
                return

            pr = cluster_priority_safe(
                df,
                cluster_col="cluster_id",
                star_col=star_col,
                group_col=group_col
            )

            self.priority_df = pr

            png_path = os.path.join(self.output_dir, "cluster_priority.png")
            labels = {
                "x_label": "Cluster ID",
                "y_label": "Priority Score",
                "title": "Cluster Priority Ranking",
            }
            out_lang = self._get_output_language()
            if out_lang in {"zh", "en"}:
                keys = list(labels.keys())
                vals = self._translate_texts_to(list(labels.values()), src_lang="en", tgt_lang=out_lang)
                labels = dict(zip(keys, vals))

            fig = plot_priority(pr, save_path=png_path, lang=self.cfg.text_language, labels=labels)
            import matplotlib.pyplot as plt
            plt.close(fig)

            csv_path = os.path.join(self.output_dir, "cluster_priority.csv")
            pr.to_csv(csv_path, index=False, encoding="utf-8-sig")

            self._set_progress(1, 1, "Priority outputs ready.")
            self._ui(lambda: messagebox.showinfo(
                "Done",
                f"Priority outputs saved:\n{png_path}\n{csv_path}"
            ))

        self._run_in_thread(job, "Generating priority outputs...")

    def on_export(self):
        if self.df_work is None or self.labels is None:
            messagebox.showwarning("Warning", "Please run the pipeline to get clustering results first.")
            return
        if "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("Warning", "cluster_id missing in df_work; re-run Step4-5.")
            return

        def job():
            self._set_progress(0, 1, "Exporting results...")

            detail_path = os.path.join(self.output_dir, "clustered_reviews.csv")
            save_csv(self.df_work, detail_path)

            asin_col = (getattr(self.cfg, "field_map", {}) or {}).get("asin") or None
            star_col = (getattr(self.cfg, "field_map", {}) or {}).get("star") or None
            text_col = (getattr(self.cfg, "field_map", {}) or {}).get("text") or None
            id_col   = (getattr(self.cfg, "field_map", {}) or {}).get("id") or None

            if not asin_col or asin_col not in self.df_work.columns:
                asin_col = "_group" if "_group" in self.df_work.columns else None
            if not star_col or star_col not in self.df_work.columns:
                star_col = "_score" if "_score" in self.df_work.columns else None
            if not text_col or text_col not in self.df_work.columns:
                text_col = "_text" if "_text" in self.df_work.columns else None
            if not id_col or id_col not in self.df_work.columns:
                id_col = "_id" if "_id" in self.df_work.columns else None

            rows = []
            total = len(self.df_work)
            for c in sorted(self.cluster_keywords.keys()):
                idx = (self.df_work["cluster_id"] == c)
                ratio = float(idx.mean()) if total > 0 else 0.0
                rows.append({
                    "cluster_id": int(c),
                    "cluster_size": int(idx.sum()),
                    "ratio": ratio,
                    "keywords": ", ".join(self.cluster_keywords.get(c, [])),
                })
            summary = pd.DataFrame(rows).sort_values("ratio", ascending=False)

            rep_rows = []
            for c, idx_list in (self.cluster_reps or {}).items():
                for rank, i in enumerate(idx_list, start=1):
                    r = self.df_work.iloc[int(i)]
                    rep_rows.append({
                        "cluster_id": int(c),
                        "rank": int(rank),
                        "ASIN": r.get(asin_col, "-") if asin_col else r.get("_group", "-"),
                        "Star": r.get(star_col, "-") if star_col else r.get("_score", "-"),
                        "review_id": r.get(id_col, "-") if id_col else r.get("_id", "-"),
                        "review_text": r.get(text_col, "") if text_col else r.get("_text", ""),
                    })

            reps_df = pd.DataFrame(rep_rows)

            out_lang = self._get_output_language()
            if self._translation_needed():
                if "keywords" in summary.columns:
                    summary = summary.copy()
                    summary["keywords"] = self._translate_series(summary["keywords"])
                if "review_text" in reps_df.columns:
                    reps_df = reps_df.copy()
                    reps_df["review_text"] = self._translate_series(reps_df["review_text"])

            if out_lang in {"zh", "en"}:
                summary.columns = self._translate_texts_to(list(summary.columns), src_lang="en", tgt_lang=out_lang)
                reps_df.columns = self._translate_texts_to(list(reps_df.columns), src_lang="en", tgt_lang=out_lang)

            xlsx_path = os.path.join(self.output_dir, "results.xlsx")
            sheets = {"cluster_summary": summary, "representatives": reps_df}

            if hasattr(self, "asin_pivot") and self.asin_pivot is not None:
                sheets["asin_cluster_percent"] = self.asin_pivot.reset_index()

            if hasattr(self, "priority_df") and self.priority_df is not None:
                sheets["cluster_priority"] = self.priority_df

            save_excel(sheets, xlsx_path)

            self._set_progress(1, 1, "Export complete.")
            self._ui(lambda: messagebox.showinfo(
                "Export complete",
                f"Exported files:\n- {detail_path}\n- {xlsx_path}"
            ))

        self._run_in_thread(job, "Exporting results...")

    def on_report_offline(self):
        if self.df_work is None or self.cluster_keywords is None or self.cluster_reps is None or self.k_scan is None:
            messagebox.showwarning("提示", "请先运行 Step1-5 并得到聚类结果后再生成报告")
            return
        if "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("提示", "当前 df_work 缺少 cluster_id，请先重跑 Step4-5")
            return

        def job():
            # ====== 列名兼容：优先使用映射列，否则用内部统一列 ======
            asin_col = (getattr(self.cfg, "field_map", {}) or {}).get("asin") or None
            star_col = (getattr(self.cfg, "field_map", {}) or {}).get("star") or None
            text_col = (getattr(self.cfg, "field_map", {}) or {}).get("text") or None
            id_col   = (getattr(self.cfg, "field_map", {}) or {}).get("id") or None

            if not asin_col or asin_col not in self.df_work.columns:
                asin_col = "_group" if "_group" in self.df_work.columns else None
            if not star_col or star_col not in self.df_work.columns:
                star_col = "_score" if "_score" in self.df_work.columns else None
            if not text_col or text_col not in self.df_work.columns:
                text_col = "_text" if "_text" in self.df_work.columns else None
            if not id_col or id_col not in self.df_work.columns:
                id_col = "_id" if "_id" in self.df_work.columns else None

            # summary
            rows = []
            total = len(self.df_work)
            for c in sorted(self.cluster_keywords.keys()):
                idx = (self.df_work["cluster_id"] == c)
                ratio = float(idx.mean()) if total > 0 else 0.0
                rows.append({
                    "cluster_id": int(c),
                    "cluster_size": int(idx.sum()),
                    "ratio": ratio,
                    "keywords": ", ".join(self.cluster_keywords.get(c, [])),
                })
            summary = pd.DataFrame(rows).sort_values("ratio", ascending=False)

            # representatives
            rep_rows = []
            for c, idx_list in (self.cluster_reps or {}).items():
                for rank, i in enumerate(idx_list, start=1):
                    r = self.df_work.iloc[int(i)]
                    rep_rows.append({
                        "cluster_id": int(c),
                        "rank": int(rank),
                        "ASIN": r.get(asin_col, "-") if asin_col else r.get("_group", "-"),
                        "Star": r.get(star_col, "-") if star_col else r.get("_score", "-"),
                        "review_id": r.get(id_col, "-") if id_col else r.get("_id", "-"),
                        "review_text": r.get(text_col, "") if text_col else r.get("_text", ""),
                    })
            reps_df = pd.DataFrame(rep_rows)

            if self._translation_needed():
                if "keywords" in summary.columns:
                    summary = summary.copy()
                    summary["keywords"] = self._translate_series(summary["keywords"])
                if "review_text" in reps_df.columns:
                    reps_df = reps_df.copy()
                    reps_df["review_text"] = self._translate_series(reps_df["review_text"])

            out_lang = self._get_output_language()
            def report_translate(texts):
                if out_lang == "zh":
                    return self._translate_texts_to(texts, src_lang="en", tgt_lang="zh")
                if out_lang == "en":
                    return self._translate_texts_to(texts, src_lang="zh", tgt_lang="en")
                return texts

            # ====== 图/表路径（存在就插入） ======
            k_png = os.path.join(self.output_dir, "k_selection.png")

            # ✅ 旧跨ASIN热力图：你现在生成的是这个名字
            asin_png = os.path.join(self.output_dir, "asin_cluster_percent_heatmap.png")

            # ✅ Priority 图（优先级排序按钮生成）
            pr_png = os.path.join(self.output_dir, "cluster_priority.png")

            # ✅ 新：ASIN×Attribute 核心升级产物
            attr_xlsx = os.path.join(self.output_dir, "asin_attribute_matrix.xlsx")
            attr_share_png = os.path.join(self.output_dir, "asin_attribute_share.png")
            attr_pain_png  = os.path.join(self.output_dir, "asin_attribute_pain.png")

            rec = recommend_k(
                self.k_scan.k_to_inertia,
                self.k_scan.k_to_silhouette,
                weight=getattr(self.cfg, "k_score_weight", 0.7),
                penalty_threshold=getattr(self.cfg, "k_penalty_threshold", 12),
                penalty_strength=getattr(self.cfg, "k_penalty_strength", 0.02)
            )
            w = getattr(self.cfg, "k_score_weight", 0.7)
            th = getattr(self.cfg, "k_penalty_threshold", 12)
            ps = getattr(self.cfg, "k_penalty_strength", 0.02)
            k_note_en = (
                "K selection uses a composite score: "
                "score = w*silhouette_norm + (1-w)*elbow_norm - penalty. "
                f"penalty = max(0, k - {th})*{ps}. "
                f"Params: w={w:.2f}, threshold={th}, strength={ps:.3f}. "
                f"Recommended K={int(rec.best_k)}."
            )
            k_note_zh = (
                "K"
                "score = w*silhouette_norm + (1-w)*elbow_norm - penalty"
                f"penalty = max(0, k - {th})*{ps}"
                f"w={w:.2f}={th}={ps:.3f}"
                f"K={int(rec.best_k)}"
            )
            out_lang = self._get_output_language()
            if out_lang == "zh":
                k_method_note = k_note_en
            elif out_lang == "en":
                k_method_note = k_note_zh
            else:
                k_method_note = k_note_zh if self._lang_bucket(self.cfg.text_language) == "zh" else k_note_en

            out_path = build_offline_report(
                cfg=self.cfg,
                output_dir=self.output_dir,
                df_all=self.df,          # 原始全量
                df_work=self.df_work,    # 用于聚类的那份
                k_to_inertia=self.k_scan.k_to_inertia,
                k_to_silhouette=self.k_scan.k_to_silhouette,
                k_best=int(rec.best_k),
                cluster_summary=summary,
                reps_df=reps_df,

                # 旧：图
                k_plot_png=k_png if os.path.exists(k_png) else None,
                asin_heatmap_png=asin_png if os.path.exists(asin_png) else None,
                priority_png=pr_png if os.path.exists(pr_png) else None,

                # ✅ 新：ASIN×Attribute（核心升级）
                asin_attr_xlsx=attr_xlsx if os.path.exists(attr_xlsx) else None,
                asin_attr_share_png=attr_share_png if os.path.exists(attr_share_png) else None,
                asin_attr_pain_png=attr_pain_png if os.path.exists(attr_pain_png) else None,
                
                key_findings_with_metrics=True,   # ✅ 论文版：带数值
                translate_fn=report_translate
            )

            self._log(f"✅ Offline report generated: {out_path}")
            self._ui(lambda: messagebox.showinfo("完成", f"离线Word报告已生成：\n{out_path}"))

        self._run_in_thread(job, "Generating offline Word report...")

    def _lang_bucket(self, lang: str) -> str:
        if not lang:
            return "en"
        return "zh" if lang.lower().startswith("zh") else "en"

    def _get_output_language(self) -> str:
        out = getattr(self.cfg, "output_language", "none")
        return (out or "none").strip().lower()

    def _translation_needed(self) -> bool:
        out = self._get_output_language()
        if out not in {"zh", "en"}:
            return False
        src = self._lang_bucket(getattr(self.cfg, "text_language", "en"))
        return out != src

    def _get_translator(self, src: str, tgt: str):
        key = f"{src}->{tgt}"
        if key in self._translators:
            return self._translators[key]

        if src == "zh" and tgt == "en":
            model_path = getattr(self.cfg, "translate_model_zh_en", "")
        else:
            model_path = getattr(self.cfg, "translate_model_en_zh", "")

        try:
            translator = Translator(model_path, batch_size=getattr(self.cfg, "translate_batch_size", 16))
        except Exception as e:
            self._log(f"⚠️ Translator init failed: {e}")
            return None

        self._translators[key] = translator
        return translator

    def _translate_texts(self, texts):
        if not texts:
            return texts
        if not self._translation_needed():
            return texts

        src = self._lang_bucket(getattr(self.cfg, "text_language", "en"))
        tgt = self._get_output_language()
        translator = self._get_translator(src, tgt)
        if not translator:
            return texts

        return translator.translate(texts)

    def _translate_texts_to(self, texts, src_lang: str, tgt_lang: str):
        if not texts:
            return texts
        if tgt_lang not in {"zh", "en"} or src_lang == tgt_lang:
            return texts
        translator = self._get_translator(src_lang, tgt_lang)
        if not translator:
            return texts
        return translator.translate(texts)

    def _translate_series(self, s: pd.Series) -> pd.Series:
        texts = ["" if pd.isna(v) else str(v) for v in s.tolist()]
        translated = self._translate_texts(texts)
        return pd.Series(translated, index=s.index)

    def _translate_labels(self, labels):
        texts = ["" if x is None else str(x) for x in labels]
        return self._translate_texts(texts)

    def _sentiment_options_for_lang(self, lang: str):
        if self._lang_bucket(lang) == "zh":
            return [
                ("zh_dianping", "\u4e2d\u6587-\u5927\u4f17\u70b9\u8bc4"),
                ("zh_general", "\u4e2d\u6587-\u901a\u7528"),
                ("zh_chinanews", "\u4e2d\u6587-\u65b0\u95fb"),
                ("zh_jd_binary", "\u4e2d\u6587-\u4eac\u4e1c"),
            ]
        return [("en_sst2", "English-SST2")]

    def _recommended_sentiment_key(self, lang: str) -> str:
        return "zh_dianping" if self._lang_bucket(lang) == "zh" else "en_sst2"

    def _derive_sentiment_key_from_cfg(self) -> str:
        key = getattr(self.cfg, "sentiment_model_key", "") or ""
        if key:
            return key
        model_map = getattr(self.cfg, "sentiment_model_map", {}) or {}
        current = getattr(self.cfg, "sentiment_model", "") or ""
        if not current or not model_map:
            return ""
        for k, p in model_map.items():
            expected = self._resolve_path(p) if hasattr(self, "_resolve_path") else p
            if os.path.normpath(str(current)) == os.path.normpath(str(expected)):
                return k
        return ""

    def _apply_sentiment_model_key(self, key: str, save: bool) -> None:
        model_map = getattr(self.cfg, "sentiment_model_map", {}) or {}
        rel_path = model_map.get(key, "")
        self.cfg.sentiment_model_key = key
        if rel_path:
            self.cfg.sentiment_model = self._resolve_path(rel_path) if hasattr(self, "_resolve_path") else rel_path

        if save:
            save_user_settings({
                "sentiment_model_key": key,
                "sentiment_model": rel_path or getattr(self.cfg, "sentiment_model", None),
            })

    def _refresh_sentiment_model_options(self, lang: str, select_key: str | None = None, save: bool = False) -> None:
        options = self._sentiment_options_for_lang(lang)
        self.sentiment_model_label_to_key = {label: key for key, label in options}
        self.sentiment_model_key_to_label = {key: label for key, label in options}
        self.sentiment_model_box["values"] = [label for _, label in options]

        key = select_key or ""
        if key not in self.sentiment_model_key_to_label:
            key = self._recommended_sentiment_key(lang)
            if key not in self.sentiment_model_key_to_label and options:
                key = options[0][0]

        label = self.sentiment_model_key_to_label.get(key, "")
        if label:
            self.sentiment_model_var.set(label)
        self._apply_sentiment_model_key(key, save=save)

    def on_language_changed(self, event=None):
        self.cfg.text_language = self.lang_var.get().strip()
        self._refresh_sentiment_model_options(
            self.cfg.text_language,
            select_key=self._recommended_sentiment_key(self.cfg.text_language),
            save=False
        )

        model_map = getattr(self.cfg, "sentiment_model_map", {}) or {}
        key = getattr(self.cfg, "sentiment_model_key", "")
        rel_path = model_map.get(key, getattr(self.cfg, "sentiment_model", ""))

        # Save to settings.json
        save_user_settings({
            "text_language": self.cfg.text_language,
            "sentiment_model_key": key,
            "sentiment_model": rel_path,
            "aihubmix_base_url": getattr(self.cfg, "aihubmix_base_url", ""),
            "aihubmix_api_key": getattr(self.cfg, "aihubmix_api_key", None),
            "aihubmix_default_model": getattr(self.cfg, "aihubmix_default_model", ""),
        })

        self._log(f"OK text_language set to: {self.cfg.text_language}")
        messagebox.showinfo("Saved", f"Text language set to: {self.cfg.text_language}\nWill be remembered next time.")

    def on_output_language_changed(self, event=None):
        label = (self.output_lang_var.get() or "").strip()
        key = self.output_lang_label_to_key.get(label, "none")
        self.cfg.output_language = key
        save_user_settings({"output_language": key})
        self._log(f"OK output_language set to: {key}")
        messagebox.showinfo("Saved", f"Output language set to: {label}\nWill be remembered next time.")

    def on_sentiment_model_changed(self, event=None):
        label = self.sentiment_model_var.get().strip()
        key = self.sentiment_model_label_to_key.get(label)
        if not key:
            return
        self._apply_sentiment_model_key(key, save=True)
        self._log(f"OK sentiment_model_key set to: {key}")
        messagebox.showinfo("Saved", f"Sentiment model set to: {label}\nWill be remembered next time.")

    def on_negative_mode_changed(self, event=None):
        """
        UI：负面筛选策略切换
        - STAR_ONLY
        - FUSION
        - SENTIMENT_ONLY
        """
        mode = self.negative_mode_var.get().strip()

        # 写回配置
        self.cfg.negative_mode = mode

        # 保存到 settings.json（和语言切换一致）
        save_user_settings({
            "negative_mode": self.cfg.negative_mode
        })

        self._log(f"✅ negative_mode set to: {self.cfg.negative_mode}")

        messagebox.showinfo(
            "已保存",
            f"负面筛选策略已切换为：\n{self.cfg.negative_mode}\n\n"
            "该设置将用于下一次 Step1–5 运行。"
        )

    def on_thresholds_changed(self, event=None):
        """
        当用户修改任意一个阈值/权重参数并触发保存时调用。
        静默保存到 cfg 和 settings.json，并记录日志。
        """
        try:
            # 1. 从 UI 变量读取数值
            star_th = float(self.star_th_var.get())
            conf_th = float(self.conf_th_var.get())
            w_star = float(self.fusion_w_star_var.get())
            w_sent = float(self.fusion_w_sent_var.get())
            keep_th = float(self.fusion_keep_var.get())

            # 2. 更新内存中的 Config 对象 (下次运行生效)
            self.cfg.star_negative_threshold = star_th
            self.cfg.sentiment_conf_threshold = conf_th
            self.cfg.fusion_w_star = w_star
            self.cfg.fusion_w_sent = w_sent
            self.cfg.fusion_keep_threshold = keep_th

            # 3. 持久化到 settings.json
            save_user_settings({
                "star_negative_threshold": star_th,
                "sentiment_conf_threshold": conf_th,
                "fusion_w_star": w_star,
                "fusion_w_sent": w_sent,
                "fusion_keep_threshold": keep_th,
            })

            # 4. 打印日志 (不弹窗打扰)
            self._log(
                f"✅ Params saved: Star<={star_th} | Conf>={conf_th} | "
                f"Fusion(wStar={w_star}, wSent={w_sent}, Keep>={keep_th})"
            )

        except ValueError:
            # 防止用户输入非法字符导致转换 float 失败
            self._log("⚠️ 参数输入格式错误，未保存。请输入有效数字。")
        except Exception as e:
            self._log(f"❌ 保存参数失败: {e}")

    def _tk_ex_handler(exc, val, tb):
        traceback.print_exception(exc, val, tb)

    def pre_download_models(self):
        """
        -huggingface
        exe 
        ./models/
            embedding/  ()
            sentiment/<key>/  ()

        config 
        cfg.embedding_model = "./models/embedding"
        cfg.sentiment_model = "./models/sentiment/<key>"   # 
        """
        # NOTE: comment text was garbled; replaced with English placeholder.
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        models_root = os.path.join(base_dir, "models")

        emb_path = getattr(self.cfg, "embedding_model", None)
        sent_path = getattr(self.cfg, "sentiment_model", None)
        trans_zh_en = getattr(self.cfg, "translate_model_zh_en", None)
        trans_en_zh = getattr(self.cfg, "translate_model_en_zh", None)

        missing = []

        # NOTE: comment text was garbled; replaced with English placeholder.
        if not emb_path:
            missing.append("embedding_model")
        else:
            # NOTE: comment text was garbled; replaced with English placeholder.
            emb_abs = emb_path
            if not os.path.isabs(emb_abs):
                emb_abs = os.path.join(base_dir, emb_path)
            if not os.path.isdir(emb_abs):
                missing.append(f"Embedding {emb_abs}")
            else:
                self._log(f" Embedding {emb_abs}")
            # NOTE: comment text was garbled; replaced with English placeholder.
            self.cfg.embedding_model = emb_abs

        # NOTE: comment text was garbled; replaced with English placeholder.
        if sent_path:
            sent_abs = sent_path
            if not os.path.isabs(sent_abs):
                sent_abs = os.path.join(base_dir, sent_path)
            if not os.path.isdir(sent_abs):
                missing.append(f"Sentiment {sent_abs}")
            else:
                self._log(f" Sentiment {sent_abs}")
            self.cfg.sentiment_model = sent_abs

        # 2.5) translation models (optional, only if output_language enabled)
        out_lang = self._get_output_language()
        if out_lang in {"zh", "en"}:
            for tag, path in [("Translate zh_en", trans_zh_en), ("Translate en_zh", trans_en_zh)]:
                if not path:
                    missing.append(f"{tag} path missing")
                    continue
                abs_path = path
                if not os.path.isabs(abs_path):
                    abs_path = os.path.join(base_dir, path)
                if not os.path.isdir(abs_path):
                    missing.append(f"{tag} missing: {abs_path}")
                else:
                    self._log(f" {tag} OK: {abs_path}")
                if "zh_en" in tag:
                    self.cfg.translate_model_zh_en = abs_path
                else:
                    self.cfg.translate_model_en_zh = abs_path

        # NOTE: comment text was garbled; replaced with English placeholder.
        # NOTE: comment text was garbled; replaced with English placeholder.
        if emb_path in (None, "", "auto"):
            default_emb = os.path.join(models_root, "embedding")
            if os.path.isdir(default_emb):
                self.cfg.embedding_model = default_emb
                self._log(f" Embedding {default_emb}")
            else:
                missing.append(f"Embedding {default_emb}")

        if sent_path in (None, "", "auto"):
            key = getattr(self.cfg, "sentiment_model_key", "") or ""
            default_sent = None
            if key:
                candidate = os.path.join(models_root, "sentiment", key)
                if os.path.isdir(candidate):
                    default_sent = candidate
            if not default_sent:
                legacy = os.path.join(models_root, "sentiment")
                if os.path.isdir(legacy):
                    default_sent = legacy
            if default_sent:
                self.cfg.sentiment_model = default_sent
                self._log(f" Sentiment {default_sent}")
            else:
                missing.append(f"Sentiment {os.path.join(models_root, 'sentiment')}")

        if out_lang in {"zh", "en"}:
            default_zh_en = os.path.join(models_root, "translate", "zh_en")
            default_en_zh = os.path.join(models_root, "translate", "en_zh")
            if os.path.isdir(default_zh_en):
                self.cfg.translate_model_zh_en = default_zh_en
                self._log(f" Translate zh_en default: {default_zh_en}")
            else:
                missing.append(f"Translate zh_en missing: {default_zh_en}")
            if os.path.isdir(default_en_zh):
                self.cfg.translate_model_en_zh = default_en_zh
                self._log(f" Translate en_zh default: {default_en_zh}")
            else:
                missing.append(f"Translate en_zh missing: {default_en_zh}")

        if missing:
            msg = (
                "Offline models missing; cannot run (release package should include models dir):\n\n"
                + "\n".join(f"- {x}" for x in missing)
                + "\n\nPlease check directory layout:\n"
                + f"{models_root}\\embedding\\...\n"
                + f"{models_root}\\sentiment\\<key>\\...\n"
            )
            self._log(msg)
            messagebox.showerror("Models missing", msg)
            raise RuntimeError(msg)

    def _on_close(self):
        """
        优雅退出：
        1) 标记 closing，阻止新的 after 调度
        2) cancel log pump 的 after
        3) 等后台线程结束（daemon=False）
        4) destroy Tk
        """
        import time

        # 防止重复触发
        if getattr(self, "_closing", False):
            return

        self._closing = True
        print("🔴 正在退出...")

        # 1) 停止 log pump（非常关键）
        try:
            if hasattr(self, "_log_pump_id") and self._log_pump_id is not None:
                self.after_cancel(self._log_pump_id)
                self._log_pump_id = None
        except Exception:
            pass

        # 2) 尝试把状态换回一下（避免析构阶段 Tk 变量还在变）
        try:
            if hasattr(self, "status"):
                self.status.set("Closing...")
        except Exception:
            pass

        # 3) 等待后台线程结束（最多等 5 秒，避免卡死）
        threads = getattr(self, "_threads", [])
        deadline = time.time() + 5.0  # 最多等 5 秒
        for t in threads:
            if t is None:
                continue
            # 分段 join，保证 UI 还活着
            while t.is_alive() and time.time() < deadline:
                try:
                    t.join(timeout=0.1)
                except Exception:
                    break

        # 4) 销毁窗口
        try:
            self.master.destroy()
        except Exception:
            pass

        print("✅ 退出完成")

    def _busy(self, is_busy: bool, message: str = ""):
        """
        设置忙碌状态：禁用主要按钮 + 更新状态栏 + 控制进度条
        只保留一套逻辑，避免重复/错误的 getattr/hasattr。
        """
        state = "disabled" if is_busy else "normal"

        # 这些是你在 _build_ui 里创建的按钮属性名
        button_names = [
            "btn_import",
            "btn_run_all",
            "btn_export",
            "btn_kplot",
            "btn_compare",
            "btn_priority",
            "btn_run_cluster",
            # 如果你还有其它按钮，比如报告按钮，也可以加：
            # "btn_report_offline",
        ]

        for name in button_names:
            if hasattr(self, name):
                btn = getattr(self, name)
                # ttk.Button / tk.Button 都支持 config(state=...)
                try:
                    btn.config(state=state)
                except Exception:
                    pass

        # 状态栏
        try:
            if hasattr(self, "status"):
                if is_busy:
                    self.status.set(message or "运行中...")
                else:
                    self.status.set("就绪")
        except Exception:
            pass

        # 进度条：你现在 progress 默认是 determinate，
        # 但你这里用 start/stop 是 indeterminate 的方式。
        # 保险做法：切换 mode 并 start/stop。
        try:
            if hasattr(self, "progress"):
                if is_busy:
                    self.progress.config(mode="indeterminate")
                    self.progress.start(10)
                else:
                    self.progress.stop()
                    self.progress.config(mode="determinate")
                    self.progress["value"] = 0
        except Exception:
            pass

        try:
            self.update_idletasks()
        except Exception:
            pass

    def _normalize_col(self, s: str) -> str:
        """用于匹配列名：去空格/下划线/短横线/大小写统一"""
        if s is None:
            return ""
        s = str(s).strip().lower()
        for ch in [" ", "\t", "\n", "\r", "_", "-", "—", "－", "·", ".", "，", ",", "：", ":"]:
            s = s.replace(ch, "")
        return s

    def _auto_map_fields(self, df):
        """
        自动识别列名，回写到 self.cfg.field_map
        目标字段：
        - text（必需）
        - asin（可选）
        - star（可选）
        - time（可选）
        """
        cols = list(df.columns)
        norm2orig = {self._normalize_col(c): c for c in cols}

        # 候选词库（你后面遇到新列名，往这里加就行）
        candidates = {
            "text": [
                "reviewtext", "review", "reviewcontent", "reviewbody", "content", "text", "body", "comment",
                "评论内容", "评论正文", "评价内容", "评价正文", "评论", "评价", "内容", "正文"
            ],
            "asin": [
                "asin", "productasin", "parentasin", "sku", "spu", "商品asin", "产品asin", "商品id", "产品id"
            ],
            "star": [
                "star", "stars", "rating", "score", "rate", "评分", "星级", "星", "打分"
            ],
            "time": [
                "reviewtime", "time", "date", "datetime", "timestamp", "评论时间", "评价时间", "时间", "日期", "发表时间"
            ]
        }

        def find_col(key: str):
            # 1) 先尊重用户/配置写死的列名（存在就用）
            cfg_name = (self.cfg.field_map.get(key) or "").strip()
            if cfg_name and cfg_name in cols:
                return cfg_name

            # 2) 精确匹配（normalize后相等）
            for cand in candidates[key]:
                n = self._normalize_col(cand)
                if n in norm2orig:
                    return norm2orig[n]

            # 3) 模糊包含（normalize后包含）
            for cand in candidates[key]:
                n = self._normalize_col(cand)
                for cn, orig in norm2orig.items():
                    if n and (n in cn or cn in n):
                        return orig

            return None

        mapped = {}
        for k in ["text", "asin", "star", "time"]:
            col = find_col(k)
            if col:
                mapped[k] = col

        # 至少要有 text
        if "text" not in mapped:
            raise ValueError(
                "缺少必要列：评论正文列（text）。\n"
                f"当前列：{cols}\n"
                "请把评论正文列命名为：ReviewText/评论内容/评论正文/评价内容 等，或在 settings.json 里配置 field_map。"
            )

        # 回写配置（让后续全流程都用识别结果）
        self.cfg.field_map.update(mapped)

        # 日志提示（中英）
        self._log("✅ Auto field mapping / 自动列名识别：")
        self._log(f"   text -> {self.cfg.field_map.get('text')}")
        self._log(f"   asin -> {self.cfg.field_map.get('asin')}")
        self._log(f"   star -> {self.cfg.field_map.get('star')}")
        self._log(f"   time -> {self.cfg.field_map.get('time')}")

        # 自动降级提示
        if not self.cfg.field_map.get("asin"):
            self._log("ℹ️ 未识别到 ASIN 列：跨ASIN对比/热力图功能将不可用或自动跳过。")
        if not self.cfg.field_map.get("star"):
            self._log("ℹ️ 未识别到 Star/评分 列：情感模型失效时的“星级兜底负面过滤”将不可用。")

    @staticmethod
    def apply_negative_filter(
        df: pd.DataFrame,
        star_col: str,
        sentiment_col: str | None,
        sentiment_conf_col: str | None,
        mode: str,
        star_threshold: float,
        conf_threshold: float,
        # ===== 新增：fusion 参数 =====
        w_star: float = 1.0,
        w_sent: float = 1.0,
        fusion_keep_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        统一负面过滤逻辑（科研 + 产品安全）
        mode:
        - STAR_ONLY
        - SENTIMENT_ONLY
        - FUSION   (✅ 升级为 WEIGHTED_FUSION：真正拉开差异)
        """

        work = df.copy()

        # ---------- 星级条件 ----------
        if star_col in work.columns:
            work[star_col] = pd.to_numeric(work[star_col], errors="coerce")
            star_neg = work[star_col] <= float(star_threshold)
        else:
            star_neg = pd.Series(False, index=work.index)

        # ---------- 情感条件（带置信度） ----------
        if sentiment_col and sentiment_col in work.columns:
            sent_neg = (work[sentiment_col] == "negative")

            # 默认 conf=0，避免缺列导致崩
            if sentiment_conf_col and sentiment_conf_col in work.columns:
                conf = pd.to_numeric(work[sentiment_conf_col], errors="coerce").fillna(0.0)
            else:
                conf = pd.Series(0.0, index=work.index)

            # conf_ok：必须 >= 阈值才算“有效负面”
            conf_ok = conf >= float(conf_threshold)
            sent_neg = sent_neg & conf_ok

            # sentiment 强度：把 conf 映射到 [0,1]（阈值以上才有强度）
            # 例：conf_th=0.6，conf=0.6 -> 0；conf=1.0 -> 1
            denom = max(1e-6, (1.0 - float(conf_threshold)))
            sent_strength = ((conf - float(conf_threshold)) / denom).clip(lower=0.0, upper=1.0)
            sent_strength = sent_strength * sent_neg.astype(float)
        else:
            sent_neg = pd.Series(False, index=work.index)
            sent_strength = pd.Series(0.0, index=work.index)

        # ---------- 策略选择 ----------
        mode = (mode or "STAR_ONLY").strip()

        if mode == "STAR_ONLY":
            mask = star_neg

        elif mode == "SENTIMENT_ONLY":
            mask = sent_neg

        elif mode == "FUSION":
            # ✅ WEIGHTED_FUSION：真正能拉开差异
            # 星级信号：负面=1，非负面=0
            star_signal = star_neg.astype(float)

            # 情感信号：sent_strength in [0,1]，越接近1越“强负面”
            # 融合得分
            neg_score = float(w_star) * star_signal + float(w_sent) * sent_strength

            # 保留阈值：越高越严格
            mask = neg_score >= float(fusion_keep_threshold)

            # 可选：把 score 留下来（后续你做分析/论文消融很好用）
            work["_neg_score"] = neg_score

        else:
            # 防御：未知模式，退化为 STAR_ONLY
            mask = star_neg

        return work[mask].reset_index(drop=True)

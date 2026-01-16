# ui/app.py
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from config import AppConfig
from core.io_utils import load_file, save_csv, save_excel, ensure_dir
from core.sentiment import SentimentAnalyzer
from core.embedding import Embedder
from core.clustering import scan_k, fit_kmeans
from core.keywords import top_keywords_by_cluster
from core.representatives import top_representatives
from core.robustness import clustering_stability

from core.plot_k import recommend_k, plot_k_curves
from core.insights import asin_cluster_percent, plot_heatmap, cluster_priority, plot_priority


class App(ttk.Frame):
    def __init__(self, master, cfg: AppConfig):
        super().__init__(master)
        self.master = master
        self.cfg = cfg

        self.df = None
        self.df_work = None  # 过滤后的（如negative）
        self.emb = None
        self.labels = None
        self.centers = None
        self.k_scan = None
        self.cluster_keywords = None
        self.cluster_reps = None
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        ensure_dir(self.output_dir)

        self._build_ui()

    def _build_ui(self):
        self.master.title("Review Analyzer")
        self.pack(fill="both", expand=True)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Button(top, text="导入CSV", command=self.on_load_csv).pack(side="left")
        ttk.Button(top, text="运行 Step1-5（全流程）", command=self.on_run_all).pack(side="left", padx=8)
        ttk.Button(top, text="导出结果", command=self.on_export).pack(side="left")

        ttk.Button(top, text="导出/显示K选择图", command=self.on_plot_k).pack(side="left", padx=8)
        ttk.Button(top, text="跨ASIN对比", command=self.on_asin_compare).pack(side="left", padx=8)
        ttk.Button(top, text="优先级排序", command=self.on_priority).pack(side="left", padx=8)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status).pack(side="right")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=6)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_data = ttk.Frame(nb)
        self.tab_k = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        nb.add(self.tab_data, text="数据预览")
        nb.add(self.tab_k, text="选K结果")
        nb.add(self.tab_results, text="聚类结果")

        # Data preview
        self.data_text = tk.Text(self.tab_data, height=18, wrap="none")
        self.data_text.pack(fill="both", expand=True)

        # K scan
        self.k_text = tk.Text(self.tab_k, height=18, wrap="none")
        self.k_text.pack(fill="both", expand=True)

        # Results
        self.res_text = tk.Text(self.tab_results, height=18, wrap="none")
        self.res_text.pack(fill="both", expand=True)

        # Bottom controls
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=6)
        self.auto_apply_k = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="扫描后自动应用推荐K", variable=self.auto_apply_k).pack(side="left", padx=10)

        ttk.Label(bottom, text="聚类K:").pack(side="left")
        self.k_var = tk.IntVar(value=5)
        ttk.Spinbox(bottom, from_=self.cfg.k_min, to=self.cfg.k_max, textvariable=self.k_var, width=5).pack(side="left", padx=6)

        self.only_negative = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="仅分析负面评论（推荐）", variable=self.only_negative).pack(side="left", padx=10)
        
        self.auto_apply_k = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="扫描后自动应用推荐K", variable=self.auto_apply_k).pack(side="left", padx=10)

        ttk.Button(bottom, text="仅重跑 Step4-5（用当前K）", command=self.on_run_cluster_only).pack(side="right")

    def _set_progress(self, cur, total, msg):
        self.status.set(msg)
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = cur
        self.master.update_idletasks()

    def on_load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV / Excel Files", "*.csv *.xlsx")]
            )

        if not path:
            return
        try:
            self.df = load_file(path, self.cfg.required_columns)
            self.df_work = None
            self.emb = None
            self.labels = None
            self.centers = None
            self.k_scan = None
            self.cluster_keywords = None
            self.cluster_reps = None

            self.data_text.delete("1.0", "end")
            self.data_text.insert("end", f"Loaded: {path}\nRows: {len(self.df)}\n\n")
            self.data_text.insert("end", self.df.head(20).to_string(index=False))
            self.status.set("CSV loaded")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _run_in_thread(self, fn):
        def runner():
            try:
                fn()
                self.status.set("Done")
            except Exception as e:
                self.status.set("Error")
                messagebox.showerror("错误", str(e))
        threading.Thread(target=runner, daemon=True).start()

    def on_run_all(self):
        if self.df is None:
            messagebox.showwarning("提示", "请先导入CSV")
            return
        self._run_in_thread(self._pipeline_all)

    def on_run_cluster_only(self):
        if self.df is None:
            messagebox.showwarning("提示", "请先导入CSV")
            return
        if self.emb is None:
            messagebox.showwarning("提示", "请先跑完整流程或至少跑到Embedding")
            return
        self._run_in_thread(self._pipeline_cluster_only)

    def _pipeline_all(self):
        # Step1 sentiment
        self._set_progress(0, 1, "Loading sentiment model...")
        sa = SentimentAnalyzer(
            model_name=self.cfg.sentiment_model,
            batch_size=self.cfg.sentiment_batch_size,
            max_chars=self.cfg.sentiment_max_chars
        )
        texts = self.df["review_text"].tolist()
        labels = sa.predict(texts, progress=self._set_progress)
        df2 = self.df.copy()
        df2["sentiment"] = labels

        # Step2 filter
        if self.only_negative.get():
            dfw = df2[df2["sentiment"] == "negative"].reset_index(drop=True)
        else:
            dfw = df2.reset_index(drop=True)

        if len(dfw) < 30:
            raise ValueError(f"过滤后样本太少（{len(dfw)}）。建议取消仅负面或补充数据。")

        self.df_work = dfw

        # Step3 embedding (cache)
        self._set_progress(0, 1, "Loading embedding model...")
        emb_cache = os.path.join(self.output_dir, "embeddings.npy")
        emb = Embedder(self.cfg.embedding_model, batch_size=self.cfg.embedding_batch_size)
        self.emb = emb.encode(self.df_work["review_text"].tolist(), cache_path=emb_cache, progress=self._set_progress)

        # Step4 scan k
        self._set_progress(0, 1, "Scanning k...")
        self.k_scan = scan_k(self.emb, self.cfg.k_min, self.cfg.k_max, random_state=self.cfg.random_state)
        self._render_k_scan()

        rec = recommend_k(self.k_scan.k_to_silhouette)
        if self.auto_apply_k.get():
            self.k_var.set(rec.best_k)


        # Step4 fit with chosen k
        self._pipeline_cluster_only()

        # 保存带情感标签的数据
        out_csv = os.path.join(self.output_dir, "reviews_with_sentiment.csv")
        save_csv(df2, out_csv)

    def _pipeline_cluster_only(self):
        k = int(self.k_var.get())
        self._set_progress(0, 1, f"Clustering k={k} ...")
        labels, centers = fit_kmeans(self.emb, k=k, random_state=self.cfg.random_state)
        self.labels = labels
        self.centers = centers
        self.df_work["cluster_id"] = labels

        # Step5 keywords + representatives
        self._set_progress(0, 1, "Extracting keywords...")
        self.cluster_keywords = top_keywords_by_cluster(
            self.df_work["review_text"].tolist(),
            self.labels,
            top_n=self.cfg.top_keywords
        )
        self._set_progress(0, 1, "Finding representatives...")
        self.cluster_reps = top_representatives(self.emb, self.labels, self.centers, top_n=self.cfg.top_representatives)

        # robustness
        self._set_progress(0, 1, "Robustness (bootstrap ARI)...")
        stab = clustering_stability(self.emb, k=k, runs=5, random_state=self.cfg.random_state)

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
        self.res_text.insert("end", f"Stability ARI (bootstrap): mean={stability['ari_mean']:.3f}, min={stability['ari_min']:.3f}, max={stability['ari_max']:.3f}\n\n")

        # 每簇输出
        for c in sorted(self.cluster_keywords.keys()):
            self.res_text.insert("end", f"=== Cluster {c} ===\n")
            kws = ", ".join(self.cluster_keywords[c])
            self.res_text.insert("end", f"Keywords: {kws}\n")

            reps = self.cluster_reps.get(c, [])
            self.res_text.insert("end", "Representatives:\n")
            for idx in reps:
                row = self.df_work.iloc[idx]
                self.res_text.insert("end", f"- ({row['ASIN']}, Star={row['Star']}) {str(row['review_text'])[:180]}...\n")
            self.res_text.insert("end", "\n")

    def on_plot_k(self):
        if self.k_scan is None:
            messagebox.showwarning("提示", "请先运行全流程（至少跑到选K扫描）")
            return
        rec = recommend_k(self.k_scan.k_to_silhouette)
        png_path = os.path.join(self.output_dir, "k_selection.png")
        plot_k_curves(self.k_scan.k_to_inertia, self.k_scan.k_to_silhouette, rec.best_k, png_path)
        messagebox.showinfo("完成", f"K选择图已导出：\n{png_path}\n推荐K={rec.best_k}（可手动修改K再重跑Step4-5）")

    def on_asin_compare(self):
        if self.df_work is None or "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("提示", "请先完成聚类（Step4-5）")
            return
        pivot = asin_cluster_percent(self.df_work, asin_col="ASIN", cluster_col="cluster_id")
        self.asin_pivot = pivot

        png_path = os.path.join(self.output_dir, "asin_cluster_heatmap.png")
        fig = plot_heatmap(pivot, save_path=png_path)
        import matplotlib.pyplot as plt
        plt.close(fig)

        csv_path = os.path.join(self.output_dir, "asin_cluster_percent.csv")
        pivot.round(2).to_csv(csv_path, encoding="utf-8-sig")
        messagebox.showinfo("完成", f"跨ASIN对比已导出：\n{png_path}\n{csv_path}")

    def on_priority(self):
        if self.df_work is None or "cluster_id" not in self.df_work.columns:
            messagebox.showwarning("提示", "请先完成聚类（Step4-5）")
            return
        pr = cluster_priority(self.df_work, cluster_col="cluster_id", star_col="Star")
        self.priority_df = pr

        png_path = os.path.join(self.output_dir, "cluster_priority.png")
        fig = plot_priority(pr, save_path=png_path)
        import matplotlib.pyplot as plt
        plt.close(fig)

        csv_path = os.path.join(self.output_dir, "cluster_priority.csv")
        pr.to_csv(csv_path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("完成", f"优先级排序已导出：\n{png_path}\n{csv_path}")

    def on_export(self):
        if self.df_work is None or self.labels is None:
            messagebox.showwarning("提示", "请先运行流程得到聚类结果")
            return

        # 1) 导出带cluster的明细
        detail_path = os.path.join(self.output_dir, "clustered_reviews.csv")
        save_csv(self.df_work, detail_path)

        # 2) 导出簇摘要表（论文Table 1）
        rows = []
        total = len(self.df_work)
        for c in sorted(self.cluster_keywords.keys()):
            idx = (self.df_work["cluster_id"] == c)
            ratio = idx.mean()
            rows.append({
                "cluster_id": c,
                "cluster_size": int(idx.sum()),
                "ratio": float(ratio),
                "keywords": ", ".join(self.cluster_keywords[c]),
            })
        summary = pd.DataFrame(rows).sort_values("ratio", ascending=False)

        # 3) 代表评论表
        rep_rows = []
        for c, idx_list in self.cluster_reps.items():
            for rank, i in enumerate(idx_list, start=1):
                r = self.df_work.iloc[i]
                rep_rows.append({
                    "cluster_id": c,
                    "rank": rank,
                    "ASIN": r["ASIN"],
                    "Star": r["Star"],
                    "review_id": r["review_id"],
                    "review_text": r["review_text"],
                })
        reps_df = pd.DataFrame(rep_rows)

        xlsx_path = os.path.join(self.output_dir, "results.xlsx")

        sheets = {"cluster_summary": summary, "representatives": reps_df}

        if hasattr(self, "asin_pivot") and self.asin_pivot is not None:
            sheets["asin_cluster_percent"] = self.asin_pivot.reset_index()

        if hasattr(self, "priority_df") and self.priority_df is not None:
            sheets["cluster_priority"] = self.priority_df

        save_excel(sheets, xlsx_path)


        messagebox.showinfo("导出完成", f"已导出：\n- {detail_path}\n- {xlsx_path}")

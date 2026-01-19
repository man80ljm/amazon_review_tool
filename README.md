
---

# Amazon Review Analyzer

**A Data-Driven Framework for User Feedback Mining and Design Attribute Analysis**

---

## 📌 Project Overview | 项目简介

**Amazon Review Analyzer** 是一款基于 **BERT 情感分析 + 语义向量聚类（Sentence Embedding + KMeans）** 的用户评论分析工具，用于从大规模用户评论中自动挖掘：

* 核心用户痛点（Pain Points）
* 关键设计属性（Design Attributes）
* 跨产品 / 跨 ASIN 的差异与机会点（Opportunities）

该工具既可用于：

* 📄 **学术研究**（论文实验、方法验证）
* 🎨 **设计决策支持**（产品改进、设计优化）
* 📊 **用户反馈分析**（多产品对比、竞品分析）

---

## 🧠 Methodology | 方法框架

整体分析流程如下：

1. **数据导入**（CSV / XLSX）
2. **负面评论筛选**

   * 基于星级（Star）
   * 基于情感模型置信度（Sentiment Confidence）
   * 支持 **STAR_ONLY / SENTIMENT_ONLY / WEIGHTED_FUSION**
3. **文本向量化**

   * Sentence Embedding（本地模型）
4. **聚类分析**

   * KMeans
   * 自动 K 值推荐（Silhouette）
5. **关键词与代表性样本提取**
6. **设计属性建模**

   * Attribute Taxonomy
   * ASIN × Attribute 热力分析
7. **优先级与机会点分析**
8. **自动生成 Word 离线报告**

---

## 🗂 Project Structure | 项目结构说明

```text
amazon_review_tool/
│
├─ core/                 # 核心算法模块
│   ├─ sentiment.py      # 情感分析（本地 BERT）
│   ├─ embedding.py      # 文本向量化
│   ├─ clustering.py     # KMeans + K 扫描
│   ├─ keywords.py       # 关键词提取
│   ├─ insights.py       # ASIN / Attribute 分析
│   ├─ report_word.py    # Word 报告生成
│   └─ ...
│
├─ ui/
│   └─ app.py            # Tkinter 图形界面
│
├─ models/               # 本地模型（不入 git）
│   ├─ sentiment/
│   └─ embedding/
│
├─ outputs/              # 分析结果输出目录
│
├─ main.py               # 程序入口
├─ config.py             # 配置与路径管理
├─ settings.json         # 用户参数配置
├─ download_models.py    # 模型下载脚本
└─ README.md
```

---

## ⚙️ Environment Setup | 环境准备（开发模式）

### 1️⃣ Python 版本

* Python **3.9 – 3.11**（推荐 3.10+）

### 2️⃣ 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

## 📥 Model Preparation | 模型准备（必须）

本工具 **默认只使用本地模型，不联网运行**。

### 下载模型：

```bash
python download_models.py
```

完成后应得到：

```text
models/
├─ sentiment/
└─ embedding/
```

---

## ▶️ Running the Application | 启动程序

### 开发模式：

```bash
python main.py
```

### 打包版本（exe）：

```text
dist/
└─ ReviewAnalyzer/
   └─ ReviewAnalyzer.exe
```

---

## 🖥 User Interface Guide | 界面与功能说明

### 🔹 主功能按钮

| 按钮          | 功能                     |
| ----------- | ---------------------- |
| 导入文件        | 加载 CSV / XLSX 评论数据     |
| 运行 Step1-5  | 全流程自动分析                |
| 仅重跑 Step4-5 | 调整 K 值后重新聚类            |
| 跨 ASIN 对比   | 生成 ASIN × Attribute 分析 |
| 优先级排序       | 痛点优先级与机会点              |
| 生成 Word 报告  | 自动生成离线分析报告             |

---

### 🔹 负面评论判定模式

* **STAR_ONLY**：仅基于星级阈值
* **SENTIMENT_ONLY**：仅基于情感模型
* **WEIGHTED_FUSION**（推荐）：

[
Score = w_{star} \cdot f(star) + w_{sent} \cdot f(sentiment)
]

参数可在界面中调节：

* Star Threshold
* Sentiment Confidence
* 权重系数

---

## 📊 Outputs | 输出结果说明

### 1️⃣ 表格输出（Excel）

* `cluster_summary`
* `asin_attribute_share`
* `asin_attribute_pain`
* `opportunity_top`

### 2️⃣ 图像输出（PNG）

* K 值选择曲线
* ASIN × Cluster 热力图
* ASIN × Attribute 热力图
* Cluster Priority 图

### 3️⃣ Word 报告（自动生成）

包含：

* 方法说明
* 参数设置
* 聚类结果
* Attribute Taxonomy
* 跨 ASIN 对比
* **Key Findings / 关键发现**
* Opportunity Insights

---

## 🔑 Key Findings | 关键发现（示例）

* Global pain Top3
* Global share Top3
* Top opportunity gaps
* Per-ASIN primary pain points

支持：

* 是否显示数值（mean / delta）
* 百分比 or 数值格式

---

## 🧪 Reproducibility | 可复现性说明

* 所有模型本地加载（`local_files_only=True`）
* 参数保存至 `settings.json`
* 输出结果可重复生成
* 适合论文复现实验与附录代码提交

---

## 📦 Packaging | 打包说明（推荐 onedir）

```bash
pyinstaller --onedir --noconsole --clean --name ReviewAnalyzer main.py ^
  --add-data "models;models" ^
  --add-data "settings.json;." ^
  --add-data "outputs;outputs"
```

发布时请分发整个 `ReviewAnalyzer/` 文件夹。

---

## 📜 License | 许可

本项目仅用于 **学术研究与教学演示**。
如需商业用途，请自行确认模型与第三方库的许可条款。

---

## ✉️ Contact

如有学术或方法问题，欢迎交流。

---

## 🧠 Notes

> This tool is designed as a **research-oriented analysis framework**,
> not just a visualization script.
> It emphasizes **interpretability, reproducibility, and design relevance**.

---

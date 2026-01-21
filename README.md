# Amazon Review Analyzer

**A Data-Driven Framework for User Feedback Mining and Design Attribute Analysis**  
**用户反馈挖掘与设计属性分析工具（离线）**

---

## 📌 Project Overview | 项目简介

**Amazon Review Analyzer** 是一款基于 **情感分析 + 语义向量聚类（Embedding + KMeans）** 的离线评论分析工具，支持从大规模用户评论中自动挖掘：

- 核心用户痛点（Pain Points）
- 关键设计属性（Design Attributes）
- 跨 ASIN 的差异与机会点（Opportunities）

适用于：
- 📄 学术研究（论文实验、方法验证）
- 🎨 设计决策支持（产品改进、设计优化）
- 📊 用户反馈分析（多产品对比、竞品分析）

---

## 🧠 Methodology | 方法框架

1) 数据导入（CSV / XLSX）  
2) 负面筛选（Star / Sentiment / Fusion）  
3) 文本向量化（本地模型）  
4) K 扫描 + 聚类  
5) 关键词与代表评论抽取  
6) 属性聚合与跨 ASIN 分析  
7) 优先级 / 机会点排序  
8) 离线 Word 报告生成  

---

## 🗂 Project Structure | 项目结构

```text
amazon_review_tool/
├─ core/                 # 核心算法模块
├─ ui/                   # Tkinter 界面
├─ models/               # 本地模型（不入 git）
├─ outputs/              # 结果输出
├─ main.py               # 程序入口
├─ config.py             # 配置
├─ settings.json         # 参数配置
├─ download_models.py    # 模型下载
└─ README.md
```

---

## ⚙️ Environment Setup | 环境准备（开发模式）

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📥 Model Preparation | 模型准备（必需）

全部模型**本地离线加载**，不联网运行：

```bash
python download_models.py
```

下载完成后目录：

```text
models/
├─ embedding/
├─ sentiment/
└─ translate/
```

---

## ▶️ Run | 启动

```bash
python main.py
```

---

## 🧭 UI Parameters | 界面参数说明

### 1) 负面判定模式

- STAR_ONLY：仅用星级阈值  
- SENTIMENT_ONLY：仅用情感模型  
- WEIGHTED_FUSION：星级 + 情感融合（推荐）  

### 2) Star / Sentiment / Fusion 参数

- Star <= X：星级阈值  
- Conf >= X：情感置信阈值  
- wStar / wSent：融合权重  
- Keep >= X：融合分数保留阈值  

建议：  
- Star <= X 越大，保留评论越多  
- Conf >= X 越大，过滤越严格  
- Keep >= X 越大，负面更“强”  
- wStar / wSent 控制谁更主导  

### 3) K 值推荐控制

综合评分：

```
score = wk * norm(silhouette) + (1 - wk) * norm(elbow)
score -= penalty * max(0, K - k_threshold)
```

- wk：轮廓系数权重（0..1）  
- K >=（k_threshold）：惩罚起点  
- penalty：惩罚强度  

默认：wk=0.7, K>=12, penalty=0.02  

建议：  
- wk 大 → 更偏轮廓系数（常更小 K）  
- wk 小 → 更偏肘部法（常更大 K）  
- penalty 大 → 更强抑制大 K  

---

## 🌐 Language | 语言设置

- 文本语言：输入评论的语言（zh/en）
- 输出语言：none / zh / en

说明：  
- 输出语言为 zh/en 时，会翻译标题、表头、关键词、属性名、代表评论等  
- 英文评论翻译成中文可能出现噪声或乱码，尤其在报告中（属正常现象）  

---

## 📤 Outputs | 输出结果

- Excel 表：聚类汇总、ASIN×属性占比、ASIN×痛点、机会点  
- PNG 图：K 选择图、热力图、优先级图  
- Word 报告：方法、参数、聚类结果、属性分析、跨 ASIN 对比  

---

## 📦 Packaging | 打包（稳定方案）

此方案通过 **Launcher + 复制完整 venv**，避免 PyInstaller + torch DLL 报错。

CMD 单行命令：

```bat
del /f /q ReviewAnalyzer.spec 2>nul & rmdir /s /q build dist 2>nul & venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name ReviewAnalyzer launcher.py & xcopy /e /i /y /q venv dist\ReviewAnalyzer\venv & xcopy /e /i /y /q core dist\ReviewAnalyzer\app\core & xcopy /e /i /y /q ui dist\ReviewAnalyzer\app\ui & xcopy /e /i /y /q models dist\ReviewAnalyzer\app\models & xcopy /e /i /y /q outputs dist\ReviewAnalyzer\app\outputs & copy /y main.py dist\ReviewAnalyzer\app\main.py & copy /y config.py dist\ReviewAnalyzer\app\config.py & copy /y settings.json dist\ReviewAnalyzer\app\settings.json
```

分发方式：  
- 打包后将 `dist/ReviewAnalyzer` 整个文件夹打包成 zip  
- 用户解压后直接运行 `ReviewAnalyzer.exe`  

启动说明：  
- 首次启动会慢（模型初始化）  
- 后续启动明显更快  

---

## 📝 License | 许可

仅用于学术研究与教学演示，商业用途请自查模型与依赖许可。


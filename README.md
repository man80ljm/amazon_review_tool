# Amazon Review Analyzer（中文说明）

Amazon Review Analyzer 是一款离线评论分析工具，集成情感过滤、本地向量化与聚类分析，用于挖掘用户痛点、设计属性以及跨 ASIN 的机会点。适用于学术研究、产品改进和用户反馈分析。

---

## 一、功能概览

核心能力:
- 负面评论筛选（星级 / 情感 / 融合）
- 本地嵌入向量化（离线模型）
- 聚类分析（KMeans / DBSCAN / 层次聚类）
- 聚类关键词 + 代表评论抽取
- 跨 ASIN 属性占比与痛点分析
- 优先级排序与机会点输出
- 离线 Word 报告生成

适用场景:
- 学术研究与方法对比
- 竞品对比与用户洞察
- 产品改进与设计决策支持

---

## 二、整体流程

1) 导入数据（CSV / XLSX）
2) 负面筛选（星级 / 情感 / 融合）
3) 文本向量化（本地 embedding）
4) 聚类分析（KMeans / DBSCAN / 层次聚类）
5) 关键词与代表评论抽取
6) 属性聚合与跨 ASIN 分析
7) 优先级排序与机会点
8) 生成离线 Word 报告

---

## 三、环境准备

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 四、模型准备（离线）

```bash
python download_models.py
```

目录结构:

```text
models/
├─ embedding/
├─ sentiment/
└─ translate/
```

---

## 五、启动程序

```bash
python main.py
```

---

## 六、详细使用说明

### 1) 数据导入

点击「导入文件（CSV/XLSX）」。

建议字段:
- 评论文本（必需）
- ASIN（可选，用于跨 ASIN 对比）
- Star（可选，用于负面筛选和痛点评分）
- ReviewTime、ReviewId（可选）

程序会自动尝试列名映射。映射失败时可通过 settings.json 修正。

### 2) 负面筛选

负面判定模式:
- STAR_ONLY: 仅用星级阈值
- SENTIMENT_ONLY: 仅用情感模型
- FUSION: 星级 + 情感融合（推荐）

参数解释:
- Star <= X: 星级阈值
- Conf >= X: 情感置信阈值
- wStar / wSent: 融合权重
- Keep >= X: 融合分数保留阈值

建议:
- Star 阈值越大，保留评论越多
- Conf 越高，筛选越严格
- Keep 越高，保留的“负面强度”越高

### 3) 聚类算法

下拉框选择:
- KMeans
- DBSCAN
- 层次聚类（Agglomerative）

#### KMeans
- 有 K 扫描（轮廓系数 + 惯性 SSE）
- 推荐 K 由加权评分给出

#### 层次聚类
- 有 K 扫描（轮廓系数 + CH）
- 推荐 K 主要由轮廓系数决定
- K 图图例: 实线 = 轮廓系数, 虚线 = CH

#### DBSCAN
- 无 K 扫描
- 参数: eps / min_samples
- 默认噪声不参与下游分析
- 可勾选 "Include noise in downstream" 将噪声计入

### 4) 聚类指标

指标（在非噪声子集上计算）:
- 轮廓系数（Silhouette）
- Calinski-Harabasz
- Davies-Bouldin（可选）

若有效簇数 < 2，则指标为空并记录提示。

采样:
- Metrics sample 用于加速指标计算
- 默认 2000

### 5) 输出按钮说明

主按钮:
- 导出结果: 导出聚类明细与汇总表
- 导出/显示 K 选择图: 生成 K 扫描图（仅 KMeans/层次聚类）
- 跨 ASIN 对比: 生成 ASIN 分布与属性表
- 优先级排序: 生成痛点优先级图表
- 生成 Word 报告（离线）: 输出完整报告

常见输出文件:
- outputs/clustered_reviews.csv
- outputs/results.xlsx
- outputs/k_selection.png
- outputs/asin_cluster_percent.csv
- outputs/asin_attribute_matrix.xlsx
- outputs/cluster_priority.png
- outputs/review_analysis_report.docx

---

## 七、结果解读指南

### 1) 聚类结果（Cluster Summary）

字段含义:
- cluster_id: 聚类编号
- cluster_size: 聚类样本数
- ratio: 聚类占比
- keywords: 该簇关键词（用于语义解释）

解读:
- ratio 高: 该簇出现频次高，是主要问题或高频话题
- keywords 与代表评论一起用于命名聚类

### 2) 代表评论（Representatives）

用于解释聚类的真实语句，建议挑选 1-3 条作为报告示例。

### 3) K 选择图

KMeans:
- 实线: 惯性 SSE（肘部法）
- 虚线: 轮廓系数
- 竖线: 推荐 K

层次聚类:
- 实线: 轮廓系数
- 虚线: CH
- 竖线: 推荐 K

### 4) 跨 ASIN 分布

ASIN × Cluster 占比热力图:
- 颜色越深，表示该 ASIN 在该簇占比越高
- 可用于发现某产品特有问题

### 5) 设计属性分析

asin_attribute_matrix.xlsx:
- attribute_taxonomy: 聚类映射成设计属性
- asin_attribute_share: 各 ASIN 的属性占比
- asin_attribute_pain: 各 ASIN 的痛点强度
- opportunity_top: 机会点（高于基准痛点）

解读:
- Share 高: 该属性是产品评论中高频话题
- Pain 高: 该属性存在明显用户痛点
- Opportunity: 相对行业均值的高痛点区域

### 6) 优先级排序（Priority）

Priority = 频次 × 严重度
- 频次高且评分低的簇优先级更高

### 7) 方法对比表

列:
- 聚类方法
- 聚类数
- 轮廓系数
- Calinski-Hara
- 聚类结果可解释性

解释性规则:
- 根据轮廓系数给出 高 / 中 / 低
- 噪声比例过高会提示“噪声较多”
- 若有效簇不足，说明“簇数不足”

---

## 八、常见问题

1) 运行很慢
- 首次启动需要加载模型
- 大数据量时建议提高筛选阈值
- 使用 Metrics sample 加速指标计算

2) DBSCAN 全是噪声
- eps 过小或 min_samples 过大
- 适当调大 eps 或降低 min_samples

3) 指标为空
- 聚类结果不足 2 个簇
- DBSCAN 噪声占比过高

---

## 九、打包（稳定方案）

```bat
del /f /q ReviewAnalyzer.spec 2>nul & rmdir /s /q build dist 2>nul & venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name ReviewAnalyzer launcher.py & xcopy /e /i /y /q venv dist\ReviewAnalyzer\venv & xcopy /e /i /y /q core dist\ReviewAnalyzer\app\core & xcopy /e /i /y /q ui dist\ReviewAnalyzer\app\ui & xcopy /e /i /y /q models dist\ReviewAnalyzer\app\models & xcopy /e /i /y /q outputs dist\ReviewAnalyzer\app\outputs & copy /y main.py dist\ReviewAnalyzer\app\main.py & copy /y config.py dist\ReviewAnalyzer\app\config.py & copy /y settings.json dist\ReviewAnalyzer\app\settings.json
```

分发方式:
- 打包 dist/ReviewAnalyzer 文件夹为 zip
- 用户解压后运行 ReviewAnalyzer.exe

---

## 十、许可

仅用于学术研究与教学演示，商业用途请自查模型与依赖许可。


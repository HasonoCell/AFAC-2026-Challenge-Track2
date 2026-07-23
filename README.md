# AFAC 2026 挑战组赛题二

本项目用于参加 AFAC 2026 挑战组赛题二，并作为“机器学习综合实践”课程项目。目标是把复杂金融文档图片解析为高保真 Markdown，最终生成比赛要求的 `file_name,ground_truth` 两列提交文件。

## 项目结构

```text
.
├── data/
│   ├── raw/                 # 官网原始 ZIP 数据，Git 忽略
│   └── extracted/           # 可选的解压结果，Git 忽略
├── docs/
│   └── Plan.md              # 当前状态、里程碑和验收目标
├── outputs/                 # 缓存、错误记录和提交文件，Git 忽略
├── src/
│   └── afac_pipeline/
│       ├── api.py           # FinixDoc-VL 客户端与响应解析
│       ├── cli.py           # 命令行入口
│       ├── datasets.py      # 数据发现、检查和解压
│       ├── evaluation.py    # 训练集代理评分和实验对比
│       ├── images.py        # 图片缩放和切片
│       ├── pipeline.py      # 预测、重试、缓存和提交生成
│       └── tables.py        # Markdown 表格解析和保守重建
├── tests/
├── main.py                  # 兼容旧运行方式的入口
└── pyproject.toml
```

## 环境准备

项目使用 `uv` 管理 Python 环境：

```bash
uv sync
```

推荐使用新的命令行入口：

```bash
uv run afac --help
```

原来的 `uv run python main.py ...` 仍然可用。

## 数据准备

将官网下载的三个压缩包放在 `data/raw/`：

```text
data/raw/
  AFAC 训练数据集.zip
  AFAC A榜评测数据集(2).zip
  finix_ab_A_submit_mock.csv.zip
```

检查数据是否完整：

```bash
uv run afac inspect-data
```

当前本地数据检查结果：

- 训练集：200 张图片、200 份 Markdown 标注、200 条映射记录。
- A 榜：100 张图片。
- 模拟提交：100 行，文件名与当前 A 榜图片完全匹配。

一般无需解压即可预测。如需查看原图和标注：

```bash
uv run afac extract --dataset all
```

## 运行预测

先使用 dry-run 验证完整流程，不调用 API：

```bash
uv run afac predict \
  --dataset a \
  --dry-run \
  --limit 3 \
  --output-csv outputs/smoke.csv
```

调用 FinixDoc-VL 测试一张图片：

```bash
uv run afac predict \
  --dataset a \
  --limit 1 \
  --user-id finixB2002 \
  --max-width 2200 \
  --slice-width 1200 \
  --slice-height 1800 \
  --timeout 300 \
  --output-csv outputs/test_one.csv
```

API 参数也可通过环境变量提供：

```bash
export FINIXDOC_USER_ID=finixB2002
export FINIXDOC_API_KEY=your_api_key
uv run afac predict --dataset a --limit 1
```

可用的 `userId`：

```text
finixA1001
finixB2002
finixC3003
finixD4004
finixE5005
```

接口可能存在 RPM 限制。`baseline-submit` 和 `experiment-train` 默认在每次 API 调用后等待 12 秒；可通过 `--sleep`、`--retries` 和 `--retry-sleep` 调整调用节奏。HTTP 429 会在重试时使用指数退避；通过 `--offset` 和 `--limit` 分批运行。

预测结果默认缓存到 `outputs/cache/`。缓存会按 `max_width`、`slice_width`、`slice_height`、重叠量和 JPEG 质量等影响输出的参数自动分目录隔离；相同参数再次运行时会复用有效缓存，使用 `--no-resume` 可以强制重新请求。

## 测试

项目目前使用 Python 标准库 `unittest`：

```bash
uv run python -m unittest discover -s tests -v
```

当前有 171 个测试，覆盖 API 响应与 HTML 可选闭合修复、截断拒绝、数据集隔离、图片头部尺寸读取与内容区域切片、并发 repair、长文档部分缓存恢复、Markdown/HTML 表格解析、HTML row-group 隔离、空表头保真、整列重叠判断、边界重复/逆序行与孤立行片段、数值表头裁片缝合并、横向 `th/td` 投票、分片 `rowspan/colspan` 拓扑恢复、带噪边界的坐标拼表与保守去重、稳定原始 tile 缓存、family 隔离实验、结构/读序代理评分、大字段 CSV、B 榜发现及提交校验。

如果需要使用 `pytest`：

```bash
uv run --with pytest python -m pytest
```

## 本地评测

训练集实验可以用 `evaluate-train` 做本地代理评分。它不是官方 TEDS 实现，而是用文本相似度、表格结构统计和阅读顺序相似度形成可复现的 proxy，适合比较不同参数方案：

```bash
uv run afac evaluate-train \
  --prediction-csv outputs/train_probe.csv \
  --raw-dir data/raw \
  --allow-subset \
  --output-csv outputs/train_probe_eval.csv
```

`--allow-subset` 适合只预测训练集一小批样本时使用；全量训练集实验可以去掉该参数，用缺失行检查保证覆盖完整。

如果要直接用当前 baseline 参数在训练集子集上做真实 API 对照，也可以先用 `baseline-submit --dataset train` 生成预测，再接同一套评测命令：

```bash
uv run afac baseline-submit \
  --dataset train \
  --limit 5 \
  --user-id finixB2002 \
  --output-csv outputs/train_probe.csv

uv run afac evaluate-train \
  --prediction-csv outputs/train_probe.csv \
  --raw-dir data/raw \
  --allow-subset
```

## 提交前校验

天池提交要求为 CSV 文件，且文件大小不能超过 100MB。当前提交次数有限，正式上传前先运行：

```bash
uv run afac validate-submission --submission-csv outputs/submission.csv
```

校验会检查：

- CSV 表头必须精确为 `file_name,ground_truth`。
- 文件大小必须小于 100MB。
- 文件名必须覆盖当前 A 榜 100 张图片，不能缺失、重复或混入未知文件。
- `ground_truth` 不能为空，不能包含 dry-run 占位内容或 `ERROR:` 标记。
- 明显截断的 Markdown fence 和未闭合 HTML table 会被拦截。

如果决定先提交一份部分空结果用于试探平台格式，可以显式加 `--allow-empty`，但这会消耗提交次数，不建议作为默认流程。

## 当前上榜 Baseline

当前推荐使用 `baseline-submit` 生成 A 榜提交。它会对长文档使用全页纵向分片；对表格页先尝试低调用量 anchor crop，如果输出过短或 crop 全部失败，则自动检测内容区域并进行网格切片修复，并跳过明显空白或低信息的修复切片。

先小批量试跑：

```bash
uv run afac baseline-submit \
  --dataset a \
  --limit 3 \
  --user-id finixB2002 \
  --output-csv outputs/baseline_probe.csv
```

确认稳定后跑全量：

```bash
uv run afac baseline-submit \
  --dataset a \
  --user-id finixB2002 \
  --on-error placeholder \
  --long-slice-height 12000 \
  --long-slice-overlap 400 \
  --table-repair-min-chars 600 \
  --table-repair-grid 4x4 \
  --table-repair-min-content-pixels 1000 \
  --table-repair-min-content-ratio 0.001 \
  --output-csv outputs/submission.csv

uv run afac validate-submission --submission-csv outputs/submission.csv
```

表头/左列上下文复用目前作为实验参数保留，不默认启用。小批量真实 API probe 显示 `240px` 或 `420px` 上下文可以补到局部表头，但不稳定增加主体文本量；需要实验时再显式加入：

```bash
--table-repair-header-context-height 240 \
--table-repair-left-context-width 240
```

覆盖切片会保守跳过只含表格线、没有可识别文字的 tile，以减少三角费率表等场景中的无效 API 调用。`--table-repair-min-text-pixels 0` 可关闭此优化。

`--on-error placeholder` 只在某张图片所有真实 API 小裁片都失败时写入一个空表兜底，确保 CSV 可以被平台接收；这类样本基本不得分，但可以避免整份提交因为个别失败样本缺行。

## B 榜冻结预设

`b-generalization-v1` 是远程 FinixDoc 路线的稳定回退点，固化了经过 family 隔离 dev/validation 验证的自适应表格修复、失败比例、重复输出熔断、HTML 结构检查和长文档参数。预设会轮转五个官方 userId，并以 5 worker 并发读取独立 repair tile；不要再把这些质量参数手工复制成一条很长的命令。

`b-generalization-v2` 是稳定的本地 OCR 回退点。它完整继承 v1，只对不少于 `100,000,000` 像素、且通过强语义表头与坐标晶格护栏的规则数值矩阵启用本地 PP-OCRv4/RapidOCR；不适用或重建失败时仍回到 v1 远程 repair。冻结实测覆盖单表、10/12 表页面和 `428×109` 超宽三角表，代理分为 `90.68–99.35`，三张 family 隔离 validation 页为 `98.43 / 96.64 / 99.19`。

`b-generalization-v3` 是当前唯一推荐版本。它只在 RapidOCR 父 tile 正好触及 1000 个检测候选上限时，以 2×2 子块替换该不完整父块；正常 tile 与 v2 逐字一致。dev 极密表有 10/48 个饱和块，细分后代理由 `99.3484` 提升到 `99.5950`、文本由 `98.5080` 提升到 `99.2479`；四个非饱和多结构页（含三张冻结 validation）输出 SHA-256 逐字不变。这些是离线代理证据，不是对未知 B 榜分数的保证。

B 榜数据到位后的全量 v3 运行已保留在 `outputs/submission_b_v3.csv`。在此基础上，长文档低覆盖页会按内容高度自动用更小切片回退，窄高表格按内容框纵横比自动改为单列 repair，并对表格候选执行失败比例、HTML 嵌套和重复行护栏。`submission_b_v5_candidate.csv` 与 `submission_b_v7_local_matrix_candidate.csv` 都收到过 `result not found!`；因此 100 行、HTML 闭合和 200 KB 单字段门都不足以证明平台可测评。`outputs/submission_b_v9_all_tables_compact_local_matrix.csv` 从 v7 出发，将全部 59 条尚含完整 HTML 表的输出逐表无损转为 Markdown 表，保留解析后的单元格文字、行列顺序及周边 OCR 文本，同时保留 v7 已恢复的超宽矩阵。它已在 B 榜正常完成评测，但分数为 `45.6108`，证明全量去除 HTML 能解决可处理性，却会损失表格结构分。

`outputs/submission_b_v13_hybrid_125k_candidate.csv` 和 `submission_b_v16_local_matrix_validated.csv` 都保留 HTML 表，但 v16 已再次得到 B 平台 `result not found!`；所以 A 榜 HTML 复杂度对照不能作为 B 可测评性的充分条件。`outputs/submission_b_v17_all_tables_compact_validated.csv` 以 v16 内容为基础，将所有完整 HTML 表无损转换为 pipe Markdown，彻底消除 `<table>/<tr>/<td>`。v17 与唯一 B 榜成功评测的 v9 在 100 行中有 99 行逐字一致；唯一不同的超宽表已用本地坐标证据将旧的伪“第106保单年度”纠正为第105年度。文件为 100 行、3,480,661 字节、最大字段 184,673 字节，严格 B-list 和 200 KB 字段门通过，SHA-256 为 `9df8726bed65764e938163e6e14e22d898dd74026860fac5774d21767ea572c7`。v17 已在 B 榜成功完成评测，线上分数为 `45.8110`，较 v9 的 `45.6108` 提升 `0.2002`。B 预设现将“纯 pipe Markdown”作为正式输出契约：每张完整 HTML 表须先验证展开后的表头、行列和单元格不变，才允许进入最终 CSV；不会再依赖提交前的临时补救。

本地后端固定为 `rapidocr-onnxruntime==1.4.4`，CPU/ONNX 完全离线运行。一次性 ONNX 图审计得到检测、识别、方向分类合计 `3,996,025` 个常量参数，低于赛题 `10M` 上限；macOS Vision 仅保留为显式诊断后端，不进入冻结预设。

同一预设也可直接用于训练集实验，例如 `uv run afac experiment-train --preset b-generalization-v3 --split dev --kind table --offset 0 --limit 1`；实验与提交因此共享完全相同的冻结参数和 userId 轮转行为。

```bash
uv run afac inspect-data

uv run afac baseline-submit \
  --dataset b \
  --preset b-generalization-v3 \
  --output-csv outputs/submission_b_v3.csv \
  --cache-dir outputs/cache/b-v3

uv run afac validate-submission \
  --dataset b \
  --submission-csv outputs/submission_b_v3.csv
```

只有 `inspect-data` 已发现 B 榜图片与独立 B 模板后才能运行和校验上述提交；A/B 文件名集合不会互相代用。预设拥有其路由与安全参数，显式写出的同名细粒度开关会被预设值覆盖。

## 当前限制

A 榜样本包含超宽、超密集费率表：

- 整页缩小后文字过小，API 可能返回空内容。
- 保留较高分辨率时输出表格过长，模型响应可能截断。
- 单纯纵向切片仍保留过多列，并会破坏表格上下文。

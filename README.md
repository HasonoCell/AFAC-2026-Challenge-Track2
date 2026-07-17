# AFAC 2026 挑战组赛题二

本项目用于参加 AFAC 2026 挑战组赛题二，并作为“机器学习综合实践”课程项目。目标是把复杂金融文档图片解析为高保真 Markdown，最终生成比赛要求的 `file_name,ground_truth` 两列提交文件。

当前工程已经完成一个可上榜 baseline：A 榜 CSV 可以稳定生成并通过本地校验，线上成绩为 `50.209198`。实现包含数据读取、FinixDoc-VL API 调用、参数隔离缓存、重试、CSV 生成链路、长文档纵向分片、表格短输出的内容区域网格修复，以及 Markdown/HTML 表格的保守重建。

下一阶段的核心目标：在保持 100 行完整提交和可复现运行的前提下，提高表格结构还原质量，重点优化超宽、超密集现金价值表的行列顺序、表头复用和重复区域去重。

详细完成情况和后续计划见 [docs/Plan.md](docs/Plan.md)。

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

接口可能存在 RPM 限制。可通过 `--sleep`、`--retries` 和 `--retry-sleep` 控制调用节奏；通过 `--offset` 和 `--limit` 分批运行。

预测结果默认缓存到 `outputs/cache/`。缓存会按 `max_width`、`slice_width`、`slice_height`、重叠量和 JPEG 质量等影响输出的参数自动分目录隔离；相同参数再次运行时会复用有效缓存，使用 `--no-resume` 可以强制重新请求。

## 测试

项目目前使用 Python 标准库 `unittest`：

```bash
uv run python -m unittest discover -s tests -v
```

当前有 36 个测试，覆盖 API 嵌套响应解析、普通 Markdown、截断响应拒绝、宽松 fence 修复、内容区域检测、二维切片坐标、横向切片命名、Markdown/HTML 表格解析、HTML `rowspan`/`colspan` 展开、多级表头扁平化、横向拼列、纵向拼行、重叠去重、参数隔离缓存，以及预测流水线和 baseline 表格修复路径。

如果需要使用 `pytest`：

```bash
uv run --with pytest python -m pytest
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

当前推荐使用 `baseline-submit` 生成 A 榜提交。它会对长文档使用全页纵向分片；对表格页先尝试低调用量 anchor crop，如果输出过短或 crop 全部失败，则自动检测内容区域并进行网格切片修复。

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
  --table-repair-min-chars 100 \
  --table-repair-grid 4x4 \
  --output-csv outputs/submission.csv

uv run afac validate-submission --submission-csv outputs/submission.csv
```

`--on-error placeholder` 只在某张图片所有真实 API 小裁片都失败时写入一个空表兜底，确保 CSV 可以被平台接收；这类样本基本不得分，但可以避免整份提交因为个别失败样本缺行。

## 当前限制

A 榜样本包含超宽、超密集费率表：

- 整页缩小后文字过小，API 可能返回空内容。
- 保留较高分辨率时输出表格过长，模型响应可能截断。
- 单纯纵向切片仍保留过多列，并会破坏表格上下文。

当前 baseline 已能稳定生成完整提交，但表格结构还原仍是主要失分点。下一阶段将继续增强表头与关键列复用、低置信度分块诊断，以及复杂切片场景下的行列级重建。

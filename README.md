# AFAC 2026 挑战组赛题二

本项目用于参加 AFAC 2026 挑战组赛题二，并作为“机器学习综合实践”课程项目。目标是把复杂金融文档图片解析为高保真 Markdown，最终生成比赛要求的 `file_name,ground_truth` 两列提交文件。

当前版本为 `v0.1.0`：数据读取、FinixDoc-VL API 调用、缓存、重试和 CSV 生成链路已经完成；针对超宽、超密集表格的二维切片与结构化重建仍在开发中，暂不建议直接消耗额度运行完整 A 榜数据。

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
│       └── pipeline.py      # 预测、重试、缓存和提交生成
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

- 训练集：100 张图片、100 份 Markdown 标注、100 条映射记录。
- A 榜：50 张图片。
- 模拟提交：100 行，其中 50 个文件名与当前 A 榜图片匹配。

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

预测结果默认缓存到 `outputs/cache/`。再次运行时会复用有效缓存，使用 `--no-resume` 可以强制重新请求。

## 测试

项目目前使用 Python 标准库 `unittest`：

```bash
uv run python -m unittest discover -s tests -v
```

当前覆盖 API 嵌套响应解析、普通 Markdown、截断响应拒绝，以及缩放后图片上传字节的回归测试。

## 当前限制

A 榜样本包含超宽、超密集费率表：

- 整页缩小后文字过小，API 可能返回空内容。
- 保留较高分辨率时输出表格过长，模型响应可能截断。
- 单纯纵向切片仍保留过多列，并会破坏表格上下文。

因此当前工程链路可以运行，但还不能稳定生成比赛级完整提交。下一阶段将实现二维表格切片、表头与关键列复用，以及分块结果的行列级重建。

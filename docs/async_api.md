# Async API Pipeline

使用异步方式调用大模型 API 进行批量推理，并将结果导出为 CSV 文件。本流程包含两个步骤：

1. **[1_async_api.py](../examples/multi_classification/1_async_api.py)** — 从 CSV 读取数据，构建提示词，并发调用 LLM API，结果存入 SQLite 缓存数据库
2. **[2_export_csv.py](../examples/multi_classification/2_export_csv.py)** — 从缓存数据库中读取 LLM 返回结果，解析 JSON 响应，导出为结构化 CSV

> **设计思路参考**: 当前文档对应的是异步 API 调用方案。若需在 Linux 上使用 SGLang 进行批量推理，请参考 [SGLang 批处理方案](../cc_prompt/sglang.md) 中的说明。

---

## 环境配置

### 1. 模型配置（`.env`）

```bash
base_url=http://xxx
api_key=sk-xxx
model=qwen/qwen3.5-9b
```

- `base_url`: OpenAI 兼容 API 的地址（支持 vLLM、SGLang、Ollama 等框架）
- `api_key`: API 密钥
- `model`: 模型名称

### 2. 任务配置（`settings.py`）

参考模板 [examples/multi_classification/settings.py](../examples/multi_classification/settings.py):

```python
PROMPT_FORMAT = """
    把下述给定的文本分类到以下类别中：['World', 'Sports', 'Business', 'Sci/Tech']。
    待分类文本: ${text}
    按照下述格式返回，要输出完整的json格式的数据：
    {
        "reason": "简要说明为什么将文本分类到该类别中",
        "llm_answer": "类别名",
    }
    """.strip()

DB_PATH = "api_cache.db"
CSV_FILE = "train.csv"
TASK_TYPE = "multiclass"

MULTICLASS_CONFIG = {
    "classes": ["World", "Sports", "Business", "Sci/Tech"],
}
```

关键字段说明:

| 字段 | 说明 |
| --- | --- |
| `PROMPT_FORMAT` | 提示词模板，`${列名}` 会被替换为 CSV 数据行中对应列的值 |
| `CSV_FILE` | 输入数据文件路径 |
| `DB_PATH` | 缓存数据库文件路径 |
| `TASK_TYPE` | 任务类型，`"binary"` 为二分类，`"multiclass"` 为多类别分类 |
| `MULTICLASS_CONFIG` | 多类别分类的类别列表 |

---

## 工作流概览

```text
                      ┌──────────────────┐
                      │   train.csv      │
                      │  (原始数据)      │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  PromptBuilder   │
                      │  构建提示词       │
                      │  ${列名} 替换    │
                      └────────┬─────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │  每个样本生成一个 prompt            │
             ▼                 ▼                 ▼
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ prompt 1   │  │ prompt 2   │  │ prompt N   │
      └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
             │               │               │
             ▼               ▼               ▼
      ┌───────────────────────────────────────────┐
      │           CachedAPIClient                 │
      │  ┌─────────┐    ┌─────────┐              │
      │  │  缓存检查 │───→│  API调用 │             │
      │  │ (SQLite) │    │ (并发10)│             │
      │  └─────────┘    └─────────┘              │
      └──────────────────┬────────────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  api_cache.db    │
                │  (缓存数据库)     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  LLMParser      │
                │  JSON 解析       │
                │  标签映射        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  llm_parsed_    │
                │  results.csv    │
                │  (导出结果)      │
                └──────────────────┘
```

---

## 步骤一：批量推理（`1_async_api.py`）

使用 `CachedAPIClient` 批量调用 LLM API，包含 SQLite 缓存机制。

### 核心代码

```python
# 1. 构建提示词
builder = PromptBuilder(PROMPT_FORMAT)
df = pd.read_csv(CSV_FILE, low_memory=False, nrows=10)
prompts_with_metadata = builder.build_prompts_with_metadata(df)

# 2. 初始化缓存客户端
client = CachedAPIClient(
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    model=os.getenv("model"),
    db_path=DB_PATH,
    max_concurrent=10,   # 最大并发数
    max_retries=3,       # 失败重试次数
)

# 3. 并发执行
async def process_prompt(item):
    prompt = item["prompt"]
    attr = item["row_data"]
    result = await client.chat(prompt, attr=attr)
    return result

tasks = [process_prompt(item) for item in prompts_with_metadata]
results = await asyncio.gather(*tasks)
```

### 缓存机制

`CachedAPIClient`（详见 [api_with_cache.py](../src/llm2bert/llm_api/api_with_cache.py)）的工作流程：

1. **查缓存**: 对 prompt 计算 SHA256 哈希，在 SQLite 中查找是否已有结果
2. **缓存命中**: 直接返回缓存结果，不调用 API
3. **缓存未命中**: 调用 API，将结果保存到 SQLite 缓存（便于后续复用）
4. **失败重试**: 最多重试 `max_retries` 次，间隔递增（2s、4s、6s...）

这一设计在调试和重复实验时极为有用——修改少量数据后重新运行，已有结果的 prompt 会直接从缓存返回，无需重复调用 API。

### 运行方式

```bash
cd examples/multi_classification
python 1_async_api.py
```

输出示例:

```text
Processing: 100%|████████████████████| 10/10 [00:15<00:00,  1.53s/it]

============================================================
Finished! Show the first result to you:
  1. {'response': '{"reason": "...", "llm_answer": "Sci/Tech"}', 'success': True, 'error': None}
```

---

## 步骤二：导出 CSV（`2_export_csv.py`）

从缓存数据库中读取 LLM 返回的原始响应，使用 `LLMParser` 解析 JSON 并映射标签，最终导出为 CSV。

### 核心流程

```python
output_path = parse_and_export_from_db(
    db_path=DB_PATH,
    output_filename="llm_parsed_results.csv",
    task_type=TASK_TYPE,
    multiclass_config=MULTICLASS_CONFIG,
    include_reason=True,     # 包含 reason 列
    include_errors=True,     # 包含解析失败的记录
)
```

### LLM 响应解析

`LLMParser`（详见 [parser.py](../src/llm2bert/llm_api/parser.py)）做以下事情：

1. **JSON 提取**: 从 LLM 返回的文本中提取 JSON 对象（支持纯 JSON 和混杂文本两种场景）
2. **字段提取**: 读取 `llm_answer` 和 `reason` 字段
3. **标签映射**: 将文本答案转换为数字标签（例如 `"Sci/Tech"` → 3）
4. **模糊匹配**: 如果文本答案不完全匹配，尝试忽略大小写和空格进行模糊匹配

### 导出的 CSV 列

| 列名 | 说明 |
| --- | --- |
| `prompt` | 发送给 LLM 的完整提示词 |
| `llm_resp` | LLM 返回的原始响应 |
| `llm_answer` | LLM 回答的类别文本 |
| `reason` | LLM 给出的分类理由 |
| `llm_pred_label` | 映射后的数字标签 |
| `success` | 解析是否成功 |
| `error` | 解析错误信息（如果有） |
| `(原始列)` | 输入 CSV 中的所有原始列（通过 `attr` 字段保存） |

### 运行导出

```bash
cd examples/multi_classification
python 2_export_csv.py
```

输出示例:

```text
数据库中共有 10 条记录

开始解析...

成功! 结果已导出到: llm_parsed_results.csv

导出数据统计:
  总记录数: 10
  列名: ['prompt', 'llm_resp', 'llm_answer', 'reason', 'llm_pred_label', ...]
  标签分布:
    标签 World: 3 条
    标签 Sports: 2 条
    标签 Business: 2 条
    标签 Sci/Tech: 3 条
```

---

## 关键模块说明

### [PromptBuilder](../src/llm2bert/llm_api/prompt_builder.py)

- 从 `PROMPT_FORMAT` 中自动提取 `${列名}` 占位符
- 支持单行构建 (`build_prompt`) 和批量构建 (`build_prompts_with_metadata`)
- `build_prompts_with_metadata` 会同时返回原始行数据，便于后续关联

### [CachedAPIClient](../src/llm2bert/llm_api/api_with_cache.py)

- 基于 OpenAI 兼容 API 的异步客户端
- 内置 SQLite 缓存，减少重复 API 调用
- 使用 `asyncio.Semaphore` 控制并发数，防止过载
- 自动重试机制（网络错误、限流等可恢复错误）

### [LLMParser](../src/llm2bert/llm_api/parser.py)

- 支持二分类和多类别分类的标签映射
- 智能 JSON 提取（正则匹配嵌套 `{}`）
- 模糊标签匹配（忽略大小写、空格）
- 一站式函数 `parse_and_export_from_db` 直接从数据库导出 CSV

---

## 常见问题

**Q: 如何测试少量数据？**

修改 `1_async_api.py` 中的 `nrows` 参数:

```python
df = pd.read_csv(CSV_FILE, low_memory=False, nrows=5)  # 只读 5 条
```

**Q: 如何清空缓存重跑？**

删除数据库文件即可:

```bash
rm api_cache.db
```

**Q: 如何调整并发数？**

修改 `CachedAPIClient` 的 `max_concurrent` 参数:

```python
client = CachedAPIClient(
    ...
    max_concurrent=20,  # 根据 API 限流调整
)
```

**Q: 导出 CSV 时出现解析错误？**

- 检查 `settings.py` 中的 `PROMPT_FORMAT` 是否指示 LLM 返回正确的 JSON 格式
- 检查 `MULTICLASS_CONFIG.classes` 是否与实际类别一致
- 开启 `include_errors=True` 查看具体错误信息

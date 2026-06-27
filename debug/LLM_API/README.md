# Async LLM API DB

带 SQLite 缓存的异步 LLM API 调用工具，支持并发请求、自动重试和可视化管理界面。

## 功能特性

- ✅ **智能缓存**: 自动缓存 API 响应，相同请求直接返回缓存结果
- 🚀 **异步并发**: 基于 asyncio 实现高效并发请求
- 🔄 **失败重试**: 自动重试失败的 API 调用，避免程序崩溃
- 📊 **Web 可视化**: 提供美观的 Web 界面查看和管理缓存
- 💾 **持久化存储**: 使用 SQLite 数据库持久化缓存
- 🎯 **Token 统计**: 记录每次调用的 token 使用量
- 🔍 **搜索功能**: 在 Web 界面中搜索缓存内容
- 📝 **提示词模板**: 灵活的提示词模板系统，支持 CSV 数据批量填充

## 项目结构

```
Async_LLM_API_DB/
├── api_with_cache.py      # 带缓存的 API 客户端（核心模块）
├── cache_web_server.py    # Web 可视化服务器
├── check_db.py            # 数据库检查工具
├── prompt_builder.py      # 提示词模板构建器（新！）
├── main.py                # 批量处理主程序（新！）
├── settings.py            # 配置文件（新！）
├── person.csv             # 示例数据 CSV
├── .env                   # 环境配置（API Key 等）
├── pyproject.toml         # 项目依赖配置
└── README.md              # 本文件
```

## 快速开始

### 1. 安装依赖

本项目使用 `uv` 作为包管理器。如果尚未安装 uv，请先安装：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或者使用 pip
pip install uv
```

安装项目依赖：

```bash
uv sync
```

### 2. 配置环境变量

复制或修改 `.env` 文件，填入你的 API 配置：

```env
api_key=your_api_key_here
base_url=https://api.example.com/v1
model=your-model-name
```

### 3. 批量处理 CSV 数据（推荐）

```bash
# 1. 修改 settings.py 中的 PROMPT_FORMAT
# 2. 准备你的 CSV 文件（如 person.csv）
# 3. 运行批量处理
uv run python main.py
```

这会：
- 读取 CSV 文件
- 根据模板生成提示词
- 并发调用 API
- 保存结果到 results.csv

### 4. 运行 API 调用示例

```bash
uv run python api_with_cache.py
```

这会运行一个测试程序，发送 50+5 个请求（其中 5 个重复用于测试缓存），你会看到：
- 进度条显示处理进度
- 缓存命中的请求会标记 `[CACHE]`
- 新请求会标记 `[API]`
- 最后显示统计汇总

### 5. 启动 Web 可视化界面

```bash
uv run python cache_web_server.py
```

然后在浏览器访问：http://localhost:5001

Web 界面功能：
- 📊 查看缓存统计（总数量、按模型分类）
- 🔍 搜索 prompt 或 response 内容
- 📄 查看完整的缓存详情
- 🗑️ 删除单个或全部缓存
- 🔄 自动刷新数据（每 10 秒）

## 核心模块说明

### api_with_cache.py

主要包含两个类：

#### `CacheDB` - SQLite 缓存管理

```python
from api_with_cache import CacheDB

# 初始化缓存数据库
cache = CacheDB("api_cache.db")

# 查询缓存
result = cache.get("你的 prompt", "model-name")
if result:
    print(f"缓存命中: {result['response']}")

# 保存缓存
cache.put(
    prompt="你的 prompt",
    model="model-name",
    response="API 响应内容",
    usage_prompt_tokens=100,
    usage_completion_tokens=200,
    usage_total_tokens=300
)

# 获取统计
stats = cache.get_stats()
print(f"缓存总数: {stats['total_cached']}")
```

#### `CachedAPIClient` - 带缓存的 API 客户端

```python
from api_with_cache import CachedAPIClient
import asyncio

async def main():
    client = CachedAPIClient(
        api_key="your-api-key",
        base_url="https://api.example.com/v1",
        model="model-name",
        db_path="api_cache.db",
        max_concurrent=10,  # 最大并发数
        max_retries=3        # 最大重试次数
    )

    # 单次调用
    result = await client.chat("你好，请介绍自己")
    if result["success"]:
        print(result["response"])
        print(f"来自缓存: {result['from_cache']}")

    # 并发调用多个请求
    prompts = ["问题1", "问题2", "问题3"]
    tasks = [client.chat(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    # 查看统计
    print(f"缓存命中: {client.stats['cache_hits']}")
    print(f"API 调用: {client.stats['api_calls']}")

asyncio.run(main())
```

### cache_web_server.py

启动 Web 服务器来管理缓存：

```bash
# 默认启动（端口 5001）
uv run python cache_web_server.py

# 指定数据库和端口
uv run python cache_web_server.py --db my_cache.db --port 8080

# 调试模式
uv run python cache_web_server.py --debug
```

命令行参数：
- `--db`: 数据库文件路径（默认: `api_cache.db`）
- `--host`: 监听地址（默认: `0.0.0.0`）
- `--port`: 监听端口（默认: `5001`）
- `--debug`: 启用调试模式

### check_db.py

快速检查数据库内容：

```bash
uv run python check_db.py
```

## 数据库结构

缓存表 `cache` 的结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键，自增 |
| prompt_hash | TEXT | Prompt 的 SHA256 哈希（唯一） |
| prompt | TEXT | 原始 Prompt 内容 |
| model | TEXT | 使用的模型名称 |
| response | TEXT | API 响应内容 |
| created_at | TIMESTAMP | 创建时间 |
| usage_prompt_tokens | INTEGER | Prompt Token 数 |
| usage_completion_tokens | INTEGER | Completion Token 数 |
| usage_total_tokens | INTEGER | 总 Token 数 |

## 提示词模板使用说明

### 快速开始

1. **准备 CSV 数据**（如 `person.csv`）：
```csv
name,age
Alice,18
Bob,25
```

2. **修改模板**（在 `settings.py` 中）：
```python
PROMPT_FORMAT = "把名字变成大写，根据年龄计算出生年份：名字:{name}，年龄:{age}"
```

3. **运行处理**：
```bash
uv run python main.py
```

### 模板语法

使用 `{列名}` 作为占位符，系统会自动从 CSV 的对应列读取数据填充。

#### 示例模板

```python
# 简单模板
PROMPT_FORMAT = "把名字变成大写：名字:{name}"

# 多字段模板
PROMPT_FORMAT = "请介绍 {name}，TA 今年 {age} 岁，住在 {city}"

# 复杂格式
PROMPT_FORMAT = """
用户信息：
- 姓名: {name}
- 年龄: {age}
- 城市: {city}

请根据以上信息生成一段自我介绍。
"""
```

### 代码使用示例

#### 基本用法

```python
from prompt_builder import PromptBuilder

# 创建构建器
builder = PromptBuilder("请介绍 {name}，TA 今年 {age} 岁")

# 查看需要的字段
print(builder.get_required_columns())  # ['age', 'name']

# 从单行数据生成提示词
row = {"name": "Alice", "age": 18}
prompt = builder.build_prompt(row)
print(prompt)  # "请介绍 Alice，TA 今年 18 岁"
```

#### 从 CSV 批量生成

```python
from prompt_builder import PromptBuilder
import pandas as pd

builder = PromptBuilder("请介绍 {name}，TA 今年 {age} 岁")

# 方法 1: 直接从 CSV 文件
prompts = builder.build_prompts_from_csv("person.csv")

# 方法 2: 从 DataFrame
df = pd.read_csv("person.csv")
prompts = builder.build_prompts_from_dataframe(df)

# 方法 3: 带元数据（保留原始数据）
results = builder.build_prompts_with_metadata(df)
for item in results:
    print(f"原始数据: {item['row_data']}")
    print(f"提示词: {item['prompt']}")
```

#### 验证模板

```python
builder = PromptBuilder("请介绍 {name}，TA 今年 {age} 岁")

# 验证 CSV 是否有需要的列
validation = builder.validate_template(["name", "age", "city"])
print(validation)
# {
#     "valid": True,
#     "placeholders": ["age", "name"],
#     "available_columns": ["age", "city", "name"],
#     "missing_columns": [],
#     "extra_columns": ["city"]
# }
```

#### 便捷函数

```python
from prompt_builder import build_prompts_from_csv, build_prompts_from_dataframe

# 一行代码从 CSV 生成
prompts = build_prompts_from_csv(
    "请介绍 {name}，TA 今年 {age} 岁",
    "person.csv"
)
```

## 使用示例

### 基础用法：批量处理任务

```python
import asyncio
from api_with_cache import CachedAPIClient
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

async def process_batch(prompts):
    client = CachedAPIClient(
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url"),
        model=os.getenv("model"),
        db_path="api_cache.db",
        max_concurrent=10
    )

    # 带进度条的并发处理
    pbar = tqdm(total=len(prompts), desc="处理中")

    async def process_one(prompt):
        result = await client.chat(prompt)
        pbar.update(1)
        return result

    tasks = [process_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # 统计结果
    success = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r["from_cache"])

    print(f"总数: {len(results)}, 成功: {success}, 缓存命中: {cached}")
    return results

# 使用
prompts = [f"请解释: {topic}" for topic in ["Python", "JavaScript", "Rust"]]
results = asyncio.run(process_batch(prompts))
```

### 结合 Web 界面使用

1. 先运行 API 调用生成缓存
2. 同时启动 Web 服务器实时查看缓存情况

```bash
# 终端 1: 运行 API 调用
uv run python api_with_cache.py

# 终端 2: 启动 Web 服务器
uv run python cache_web_server.py
```

## 依赖说明

项目主要依赖：

| 包名 | 版本 | 用途 |
|------|------|------|
| openai | >=2.43.0 | OpenAI 兼容的异步 API 客户端 |
| flask | >=3.1.3 | Web 服务器框架 |
| flask-cors | >=6.0.5 | CORS 支持 |
| pandas | >=3.0.3 | 数据处理（可选） |
| tenacity | >=9.1.4 | 重试工具 |
| python-dotenv | >=1.2.1 | 环境变量管理 |
| tqdm | - | 进度条 |

完整依赖见 `pyproject.toml` 和 `uv.lock`。

## 常见问题

### 1. 缓存是如何工作的？

缓存基于 Prompt 和 Model 的 SHA256 哈希值。只要 Prompt 和 Model 完全相同，就会命中缓存。注意：空格、标点符号的差异都会导致不同的哈希值。

### 2. 如何清空缓存？

有两种方式：
- 使用 Web 界面的「清空缓存」按钮
- 直接删除 `api_cache.db` 文件

### 3. 如何调整并发数？

在创建 `CachedAPIClient` 时设置 `max_concurrent` 参数：

```python
client = CachedAPIClient(
    ...
    max_concurrent=20,  # 增加到 20 并发
    ...
)
```

### 4. 支持哪些 API 提供商？

所有兼容 OpenAI API 格式的提供商都支持，包括：
- OpenAI 官方 API
- Azure OpenAI
- 火山引擎（Doubao）
- 通义千问
- DeepSeek
- vLLM 本地部署
- 等等...

## 开发说明

### 安装开发依赖

```bash
uv sync --dev
```

### 运行 Jupyter Notebook（可选）

```bash
uv run jupyter notebook
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v0.2.0 (2026-06-26)

- ✨ 新增提示词模板系统（prompt_builder.py）
- ✅ 支持 {列名} 占位符自动提取和填充
- ✅ 新增配置文件（settings.py）
- ✅ 新增批量处理主程序（main.py）
- ✅ 自动从 CSV 读取数据并生成提示词
- ✅ 结果保存到 CSV

### v0.1.0 (2026-06-26)

- ✨ 初始版本
- ✅ 带缓存的异步 API 客户端
- ✅ Web 可视化管理界面
- ✅ 并发请求和自动重试
- ✅ SQLite 持久化存储

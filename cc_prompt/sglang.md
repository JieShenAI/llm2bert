我现在想在linux系统上使用 sglang 的generate处理批量数据。请你参考 1_async_api.py 中的代码代码进行实现，帮我编写使用 sglang 进行批量数据推理的代码。

需要从 settings.py 中导入以下配置：

```python
from settings import PROMPT_FORMAT, CSV_FILE, TASK_TYPE, MULTICLASS_CONFIG
```

- **PROMPT_FORMAT**: 提示词模板，占位符格式为 `${列名}`，用于 PromptBuilder 构建提示词
- **CSV_FILE**: 输入数据文件路径，用 pandas 读取后逐行构建 prompt
- **TASK_TYPE**: 任务类型（`"binary"` 或 `"multiclass"`），用于后续的标签映射
- **MULTICLASS_CONFIG**: 多类别分类配置，包含 `classes` 列表，用于将 llm_answer 映射为数字 label

要求：

1. 使用 PromptBuilder 从 PROMPT_FORMAT 和 CSV 数据构建提示词
2. 调用 sglang 的 generate 进行批量推理
3. 解析 LLM 返回的 JSON 结果，提取 `llm_answer` 和 `reason`
4. 根据 TASK_TYPE 和 MULTICLASS_CONFIG 将 llm_answer 映射为数字 label
5. **不需要数据库缓存中间结果**，直接推理并导出
6. 最终导出的 csv 包含原始表格的所有字段 + prompt、llm_answer、reason、label 四列

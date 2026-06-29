# Sglang Infer

## 介绍

硬件条件：Linux系统 + Nvidia显卡

利用 Sglang 框架使用本地Qwen大模型对大批量数据进行推理。优点是推理速度很快。通过下面的终端信息可看出处理2800条数据，耗时123秒，从而算出推理速度是 22.76 样本/秒。这是非常快的速度，远远快于调用大模型API的速度。

补充：即便在Linux系统上，利用 Sglang 部署大模型，再通过异步调用API的速度也不到 3 样本/秒，这个速度远小于 Sglang 的推理速度。


## 终端输出信息分析

`python 1_sglang_infer.py --model Qwen/Qwen3-4B-Instruct-2507`

```
(llm2bert) jie@Jie:~/github/PUBLIC/llm2bert/examples/multi_classification$ python 1_sglang_infer.py --model Qwen/Qwen3-4B-Instruct-2507
正在读取数据: train.csv
读取到 2800 条数据
正在构建提示词...
把下述给定的文本分类到以下类别中：['World', 'Sports', 'Business', 'Sci/Tech']。输出结果按照json格式返回。
【示例】：
text: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.
output: {""reason"": ""该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为'Sports'。"", ""llm_answer"": ""Sports""}
【待分类文本】:
text: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.
output:
正在加载模型: Qwen/Qwen3-4B-Instruct-2507
...
Loading safetensors checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  33% Completed | 1/3 [00:01<00:02,  1.10s/it]
Loading safetensors checkpoint shards: 100% Completed | 3/3 [00:03<00:00,  1.08s/it]
Loading safetensors checkpoint shards: 100% Completed | 3/3 [00:03<00:00,  1.08s/it]

Capturing batches (bs=1 avail_mem=19.00 GB): 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.21it/s]
模型加载完成！
开始批量生成...
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:03<00:00, 123.38s/it]

处理完成！
总记录数: 2800
成功解析: 2800
解析失败: 0
结果已保存到: llm_parsed_results.csv
/usr/lib/python3.12/multiprocessing/resource_tracker.py:123: UserWarning: resource_tracker: process died unexpectedly, relaunching.  Some resources might leak.
  warnings.warn('resource_tracker: process died unexpectedly, '
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/resource_tracker.py", line 239, in main
    cache[rtype].remove(name)
KeyError: '/loky-109805-we4l2lc5'

```

在运行后，可以在终端中看到第一条数据的提示词，用于验证提示词封装是否正确。同时可看到成功解析2800。

遇到了 KeyError: '/loky-109805-we4l2lc5' 的报错，我选择忽略，程序运行正常，这个报错对结果没有影响。

## 配置文件（settings.py）

项目使用 `settings.py` 统一管理任务配置，无需在命令行参数中重复指定。配置项如下：

```python
# 提示词模板
PROMPT_FORMAT = """
    把下述给定的文本分类到以下类别中：['World', 'Sports', 'Business', 'Sci/Tech']。
    待分类文本: ${text}
    按照下述格式返回，要输出完整的json格式的数据：
    {
        "reason": "简要说明为什么将文本分类到该类别中",
        "llm_answer": "类别名",
    }
    """.strip()

# CSV 数据文件
CSV_FILE = "train.csv"

# 任务类型：二分类 "binary" 或 多类别分类 "multiclass"
TASK_TYPE = "multiclass"

# 多类别分类的类别列表
MULTICLASS_CONFIG = {
    "classes": ["World", "Sports", "Business", "Sci/Tech"],
}

# 二分类配置（TASK_TYPE = "binary" 时使用）
# BINARY_CLASS_CONFIG = {
#     "positive_label": "是",
#     "negative_label": "否",
# }
```

### 各字段说明

| 字段 | 说明 |
| --- | --- |
| `PROMPT_FORMAT` | 提示词模板，`${列名}` 会被替换为 CSV 中对应列的值 |
| `CSV_FILE` | 输入数据文件路径 |
| `TASK_TYPE` | 任务类型，`"binary"` 为二分类，`"multiclass"` 为多类别分类 |
| `MULTICLASS_CONFIG` | 多类别分类的类别列表，`classes` 顺序决定了数字 label 的映射顺序 |
| `BINARY_CLASS_CONFIG` | 二分类配置，指定正负类别的文本值（仅在 `TASK_TYPE = "binary"` 时需要） |

### 修改提示词模板

`PROMPT_FORMAT` 中的 `${text}` 是占位符，运行时会被 CSV 文件中 `text` 列的实际值替换。如果 CSV 列名不同（如 `content`、`sentence`），相应修改占位符即可：

```python
PROMPT_FORMAT = "请将以下文本分类：${content} ..."
```

### 适配不同分类任务

修改 `MULTICLASS_CONFIG.classes` 即可适配任意的多类别分类任务，类别顺序决定了最终 CSV 中 `llm_pred_label` 的数字编号：

```python
MULTICLASS_CONFIG = {
    "classes": ["positive", "negative", "neutral"],  # 三分类情感分析
}
# "positive" → label 0, "negative" → label 1, "neutral" → label 2
```

## 参数介绍

```sh
python 1_sglang_infer.py \
--model Qwen/Qwen3-4B-Instruct-2507 \
--nrows 10240 \
--block_size 4096
```

- `--model`： 开源模型名，默认与 huggingface 的模型名保持一致，当然也可以使用大模型的绝对地址；选择 Qwen/Qwen3-4B-Instruct-2507 是考虑到它是指令模型，遵循指令结构化输出能力强，且没有思考过程能更快完成推理。
- `--nrows`: 只加载 nrows 行表格数据，避免表格数据过多；
- `--block_size`: 推理 block_size 条数据后，做一下数据保存，避免程序中途崩溃导致所有数据丢失；

以上参数与 `settings.py` 中的配置共同发挥作用——`settings.py` 管理任务本身的配置（提示词、分类标签），命令行参数管理执行层面的配置（模型、批大小）。脚本通过导入 `settings.py` 获取 PROMPT_FORMAT、TASK_TYPE、MULTICLASS_CONFIG 等配置：

```python
from settings import PROMPT_FORMAT, CSV_FILE, TASK_TYPE, MULTICLASS_CONFIG

process_data(
    model_path=args.model,
    csv_file=args.csv,
    prompt_format=PROMPT_FORMAT,
    task_type=TASK_TYPE,
    multiclass_config=MULTICLASS_CONFIG,
    ...
)
```



## 代码介绍

在代码中，通过`LLMPredictResult`定义大模型输出的结构，Sglang会通过提示词的方式让大模型返回LLMPredictResult定义的结构，这被称为大模型结构化输出。Sglang 结构化输出相关的文档：[https://docs.sglang.io/docs/advanced_features/structured_outputs#offline-engine-api](https://docs.sglang.io/docs/advanced_features/structured_outputs#offline-engine-api)

若想让大模型返回更多的字段，通过调整 LLMPredictResult 的结构即可，同时可能也需要对提示词模板中的示例进行相应的修改。可通过修改description字段的描述，给大模型介绍该字段的含义。

```python
class LLMPredictResult(BaseModel):
    reason: str = Field(..., description="一步一步思考的过程")
    llm_answer: str = Field(..., description="最终预测的结果")
...

llm = sgl.Engine(
    model_path=model_path,
    attention_backend="triton",
    tp_size=1,  # tensor parallel size
    mem_fraction_static=0.9,  # Use 90% of GPU memory
    chunked_prefill_size=8192,  # Larger chunks for better throughput
    max_total_tokens=8192,  # Limit total KV cache size
)

sampling_params = {
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
    "json_schema": json.dumps(LLMPredictResult.model_json_schema()),
}
...

batch_outputs = llm.generate(
    batch_prompts,
    sampling_params=sampling_params,
)
```



## 输出结果

该脚本在大模型推理完成之后，会把所有的推理结果以csv表格的形式导出。下述是第一条数据的各个字段：

```json
{'prompt': '把下述给定的文本分类到以下类别中：[\'World\', \'Sports\', \'Business\', \'Sci/Tech\']。输出结果按照json格式返回。\n【示例】：\ntext: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.\noutput: {""reason"": ""该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为\'Sports\'。"", ""llm_answer"": ""Sports""}\n【待分类文本】:\ntext: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.\noutput:',
 'llm_resp': '{"reason": "该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为\'Sports\'。","llm_answer": "Sports"}',
 'llm_answer': 'Sports',
 'reason': "该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为'Sports'。",
 'llm_pred_label': 1,
 'success': True,
 'error': nan,
 'text': 'Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.',
 'true_label': 1,
 'label_name': 'Sports'}
```

逐个字段介绍如下：

- prompt: 封装完成的提示词；
- llm_resp：大模型输出的完整结果；
- reason：llm_resp 中的reason字段；
- llm_answer： llm_resp 中的 llm_answer 字段；推理的结果；
- llm_pred_label：paser 根据 llm_answer 字段解析的 label 数字编号；该编号用于作为训练BERT模型的label。
- success：paser 解析成功；
- error：paser 解析失败的报错；
- 其他字段是原始表格的字段；

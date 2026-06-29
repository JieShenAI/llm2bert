"""
使用 sglang 进行批量数据推理

功能：
1. 使用 sglang 的 generate 接口进行批量推理
2. 支持二分类和多类别分类（通过 settings.py 配置）
3. 直接导出结果到 CSV（不使用数据库缓存）
4. 输出包含原始表格字段 + prompt + llm_answer + reason + label

"""

import os
import sys
import json
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pydantic import BaseModel, Field

# 添加项目路径
sys.path.append("../../src")

# 导入项目模块
from settings import (
    # PROMPT_FORMAT,
    CSV_FILE,
    TASK_TYPE,
    # BINARY_CLASS_CONFIG,
    MULTICLASS_CONFIG,
    LLM_PREDICT_CSV_FILE,
)
from llm2bert.llm_api.prompt_builder import PromptBuilder
from llm2bert.llm_api.parser import LLMParser

# 【注意】：没有使用settings.py 里面的PROMPT_FORMAT，这里重写了 PROMPT_FORMAT
PROMPT_FORMAT = """
把下述给定的文本分类到以下类别中：['World', 'Sports', 'Business', 'Sci/Tech']。输出结果按照json格式返回。
【示例】：
text: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.
output: {""reason"": ""该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为'Sports'。"", ""llm_answer"": ""Sports""}
【待分类文本】:
text: ${text}
output:
""".strip()
# {
#     "reason": "简要说明为什么将文本分类到该类别中",
#     "llm_answer": "类别名",
# }


class LLMPredictResult(BaseModel):
    reason: str = Field(..., description="一步一步思考的过程")
    llm_answer: str = Field(..., description="最终预测的结果")


def process_data(
    model_path: str,
    csv_file: str,
    prompt_format: str,
    task_type: str,
    binary_config: Optional[Dict[str, str]] = None,
    multiclass_config: Optional[Dict[str, List[str]]] = None,
    output_file: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    batch_size: int = 32,
    nrows: Optional[int] = None,
) -> str:
    """
    完整的数据处理流程

    Args:
        model_path: 模型路径
        csv_file: 输入 CSV 文件
        prompt_format: 提示词模板
        task_type: 任务类型 ("binary" 或 "multiclass")
        binary_config: 二分类配置
        multiclass_config: 多类别分类配置
        output_file: 输出文件路径（如果为 None 则自动生成）
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数
        batch_size: 批处理大小
        nrows: 只读取前 n 行（用于测试）

    Returns:
        输出文件路径
    """
    import sglang as sgl

    # 1. 读取数据
    print(f"正在读取数据: {csv_file}")
    df = pd.read_csv(csv_file, nrows=nrows, low_memory=False)
    print(f"读取到 {len(df)} 条数据")

    # 2. 构建提示词
    print("正在构建提示词...")
    builder = PromptBuilder(prompt_format)
    prompts_with_metadata = builder.build_prompts_with_metadata(df)
    prompts = [item["prompt"] for item in prompts_with_metadata]
    print(prompts[0])
    row_data_list = [item["row_data"] for item in prompts_with_metadata]

    # 3. 初始化 sglang 模型
    print(f"正在加载模型: {model_path}")
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
    print("模型加载完成！")

    # 4. 初始化输出文件路径
    if output_file is None:
        csv_path = Path(csv_file)
        output_file = str(csv_path.parent / f"{csv_path.stem}_sglang_results.csv")

    temp_output_file = output_file.replace(".csv", "_temp.csv")

    # 5. 检查断点：如果已有临时文件，加载已处理的结果并跳过
    processed_count = 0
    parsed_results = []
    if os.path.exists(temp_output_file):
        print(f"检测到临时文件: {temp_output_file}")
        existing_df = pd.read_csv(temp_output_file, low_memory=False)
        processed_count = len(existing_df)
        print(f"已处理 {processed_count} 条数据，将从第 {processed_count} 条继续")
        # 用已有结果填充 parsed_results 用于最终统计
        for _, row in existing_df.iterrows():
            parsed_results.append(row.to_dict())
    elif os.path.exists(output_file):
        print(f"检测到已有结果文件: {output_file}")
        existing_df = pd.read_csv(output_file, low_memory=False)
        processed_count = len(existing_df)
        print(f"已处理 {processed_count} 条数据，将从第 {processed_count} 条继续")
        for _, row in existing_df.iterrows():
            parsed_results.append(row.to_dict())

    # 6. 批量生成 + 即时保存
    print("开始批量生成...")

    parser = LLMParser(
        task_type=task_type,
        # binary_config=binary_config,
        multiclass_config=multiclass_config,
    )

    # 分批处理
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        # 跳过已处理的批次
        if i < processed_count:
            continue

        batch_prompts = prompts[i : i + batch_size]
        batch_row_data = row_data_list[i : i + batch_size]

        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
        )

        # 解析并保存当前批次的结果
        batch_results = []
        for j, output in enumerate(batch_outputs):
            response = output["text"].strip()
            parsed = parser.parse_response(response)

            result = {
                "prompt": batch_prompts[j],
                "llm_resp": response,
                "llm_answer": parsed["llm_answer"],
                "reason": parsed["reason"],
                "llm_pred_label": parsed["label"],
                "success": parsed["success"],
                "error": parsed["error"],
            }
            # 添加原始数据
            result.update(batch_row_data[j])
            batch_results.append(result)
            parsed_results.append(result)

        # 即时保存当前批次到临时文件
        batch_df = pd.DataFrame(batch_results)
        header = not os.path.exists(temp_output_file)
        batch_df.to_csv(
            temp_output_file,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8-sig",
        )

        # 同时覆盖写入正式输出文件作为备份 【重复写入没有必要】
        # pd.DataFrame(parsed_results).to_csv(
        #     output_file, index=False, encoding="utf-8-sig"
        # )

    try:
        llm.shutdown()
    except Exception as e:
        print("shutdown skipped:", e)

    del llm

    # 7. 确保最终输出文件存在（即使所有批次被跳过，也重新写入一遍）
    if parsed_results:
        pd.DataFrame(parsed_results).to_csv(
            output_file, index=False, encoding="utf-8-sig"
        )

    # 8. 清理临时文件
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    # 统计
    success_count = sum(1 for r in parsed_results if r["success"])
    print(f"\n处理完成！")
    print(f"总记录数: {len(parsed_results)}")
    print(f"成功解析: {success_count}")
    print(f"解析失败: {len(parsed_results) - success_count}")
    print(f"结果已保存到: {output_file}")
    return output_file


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="使用 sglang 进行批量推理")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_FILE,
        help=f"输入 CSV 文件 (默认: {CSV_FILE})",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=LLM_PREDICT_CSV_FILE,
        help="输出 CSV 文件路径 (默认: 自动生成)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4096,
        help="批处理大小 (默认: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度 (默认: 0.0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="最大生成 token 数 (默认: 2048)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="只读取前 n 行 (用于测试)",
    )

    args = parser.parse_args()

    # 处理数据
    process_data(
        model_path=args.model,
        csv_file=args.csv,
        prompt_format=PROMPT_FORMAT,
        task_type=TASK_TYPE,
        # binary_config=BINARY_CLASS_CONFIG,
        multiclass_config=MULTICLASS_CONFIG,
        output_file=args.output_file,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.block_size,
        nrows=args.nrows,
    )


if __name__ == "__main__":
    main()

# python 1_sglang_infer.py --model Qwen/Qwen3-4B-Instruct-2507 --nrows 20480 --block_size 4096
# python 1_sglang_infer.py --model Qwen/Qwen3-4B-Instruct-2507 --nrows 4096 --block_size 2048

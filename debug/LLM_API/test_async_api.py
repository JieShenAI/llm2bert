import os

import sys

sys.path.append("../../src")
import asyncio
from tqdm import tqdm
import pandas as pd

from llm2bert.llm_api.api_with_cache import CachedAPIClient
from llm2bert.llm_api.prompt_builder import PromptBuilder

DF_PATH = "../../data/北京_2020.csv"


async def main():
    # 获取提示词
    with open("../../data/gemini-prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    builder = PromptBuilder(prompt_template)
    df = pd.read_csv(DF_PATH, nrows=10)  # 只读取前 10 行数据
    prompts_with_metadata = builder.build_prompts_with_metadata(df)

    client = CachedAPIClient(
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url"),
        model=os.getenv("model"),
        db_path="api_cache.db",
        max_concurrent=10,
        max_retries=3,
    )


    # 创建进度条
    pbar = tqdm(total=len(prompts_with_metadata), desc="Processing")

    # 并发处理
    async def process_prompt(item):
        prompt = item["prompt"]
        attr = item["row_data"]
        result = await client.chat(prompt, attr=attr)
        pbar.update(1)
        return result

    tasks = [process_prompt(item) for item in prompts_with_metadata]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    result = asyncio.run(main())
    print("\n" + "=" * 60)
    print("Final Results:")
    for i, res in enumerate(result):
        print(f"  {i + 1}. {{'response': {res['response']}, 'success': {res['success']}, 'error': {res['error']}}}")

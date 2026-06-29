from llm2bert.llm_api.prompt_builder import PromptBuilder
import pandas as pd
from settings import PROMPT_FORMAT, CSV_FILE

print(f"使用模板: {PROMPT_FORMAT}")
print(f"使用 CSV: {CSV_FILE}")

builder = PromptBuilder(PROMPT_FORMAT)
try:
    df = pd.read_csv(CSV_FILE)
    print(f"CSV 列: {list(df.columns)}")

    prompts = builder.build_prompts_from_dataframe(df)
    print(f"\n生成的 {len(prompts)} 个提示词:")
    for i, p in enumerate(prompts, 1):
        print(f"{i}. {p}")

except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
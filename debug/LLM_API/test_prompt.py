"""
测试提示词构建器
"""

import sys

# 修复 Windows 编码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from llm2bert.llm_api.prompt_builder import PromptBuilder
import pandas as pd

print("=" * 60)
print("提示词构建器测试")
print("=" * 60)

# 测试不同的模板
test_cases = [
    ("把名字变成大写，根据年龄计算出生年份：名字:{name}，年龄:{age}", ["name", "age"]),
    ("请介绍 {name}，TA 今年 {age} 岁", ["name", "age"]),
    ("把名字变成大写：名字:{name}", ["name"]),
    ("USER: {name} AGE: {age} CITY: {city}", ["name", "age", "city"]),
]

for i, (template, required) in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {template}")
    builder = PromptBuilder(template)
    print(f"  提取的占位符: {builder.get_required_columns()}")

    # 测试数据
    test_data = {
        "name": "Alice",
        "age": 18,
        "city": "Beijing"
    }

    try:
        prompt = builder.build_prompt(test_data)
        print(f"  生成结果: {prompt}")
    except Exception as e:
        print(f"  错误: {e}")

# 测试从 CSV 读取
print("\n" + "=" * 60)
print("测试从 CSV 读取")
print("=" * 60)



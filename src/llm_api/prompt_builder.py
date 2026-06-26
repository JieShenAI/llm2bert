"""
提示词模板构建器

功能：
1. 从 PROMPT_FORMAT 中自动提取占位符（如 {name}, {age}）
2. 将 CSV 数据填充到模板中生成完整提示词
3. 支持任意格式的模板
"""

import re
from typing import Dict, List, Any, Set
import pandas as pd


class PromptBuilder:
    """提示词模板构建器"""

    def __init__(self, prompt_format: str):
        """
        初始化构建器

        Args:
            prompt_format: 提示词模板，如 "把名字变成大写：名字:{name}，年龄:{age}"
        """
        self.prompt_format = prompt_format
        self.placeholders = self._extract_placeholders(prompt_format)

    @staticmethod
    def _extract_placeholders(prompt_format: str) -> Set[str]:
        """
        从模板中提取所有占位符

        Args:
            prompt_format: 提示词模板

        Returns:
            占位符名称集合，如 {'name', 'age'}
        """
        # 匹配 {name} 格式的占位符
        pattern = r'\{([^}]+)\}'
        placeholders = set(re.findall(pattern, prompt_format))
        return placeholders

    def get_required_columns(self) -> List[str]:
        """
        获取需要的 CSV 列名

        Returns:
            列名列表
        """
        return sorted(list(self.placeholders))

    def build_prompt(self, row_data: Dict[str, Any]) -> str:
        """
        根据一行数据构建提示词

        Args:
            row_data: 包含所需字段的字典，如 {'name': 'Alice', 'age': 18}

        Returns:
            填充后的提示词

        Raises:
            ValueError: 当缺少必要字段时
        """
        # 检查是否有缺失的字段
        missing = self.placeholders - set(row_data.keys())
        if missing:
            raise ValueError(f"缺少必要字段: {sorted(missing)}")

        # 使用字符串格式化填充模板
        try:
            return self.prompt_format.format(**row_data)
        except KeyError as e:
            raise ValueError(f"模板中引用了不存在的字段: {e}")
        except Exception as e:
            raise ValueError(f"构建提示词时出错: {e}")

    def build_prompts_from_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        从 DataFrame 批量构建提示词

        Args:
            df: pandas DataFrame

        Returns:
            提示词列表
        """
        # 检查 DataFrame 是否包含所有需要的列
        missing = self.placeholders - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        prompts = []
        for _, row in df.iterrows():
            # 将行转换为字典
            row_dict = row.to_dict()
            # 构建提示词
            prompt = self.build_prompt(row_dict)
            prompts.append(prompt)

        return prompts

    def build_prompts_from_csv(self, csv_path: str) -> List[str]:
        """
        从 CSV 文件批量构建提示词

        Args:
            csv_path: CSV 文件路径

        Returns:
            提示词列表
        """
        df = pd.read_csv(csv_path)
        return self.build_prompts_from_dataframe(df)

    def build_prompts_with_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        构建带元数据的提示词列表（包含原始数据）

        Args:
            df: pandas DataFrame

        Returns:
            字典列表，每个字典包含 {'prompt': str, 'row_data': dict}
        """
        missing = self.placeholders - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            prompt = self.build_prompt(row_dict)
            results.append({
                "prompt": prompt,
                "row_data": row_dict
            })

        return results

    def validate_template(self, available_columns: List[str]) -> Dict[str, Any]:
        """
        验证模板是否有效

        Args:
            available_columns: 可用的列名列表

        Returns:
            验证结果字典
        """
        available = set(available_columns)
        missing = self.placeholders - available
        extra = available - self.placeholders

        return {
            "valid": len(missing) == 0,
            "placeholders": sorted(list(self.placeholders)),
            "available_columns": sorted(list(available)),
            "missing_columns": sorted(list(missing)),
            "extra_columns": sorted(list(extra))
        }


# ============================================
# 便捷函数
# ============================================

def build_prompts_from_csv(prompt_format: str, csv_path: str) -> List[str]:
    """
    从 CSV 文件生成提示词列表（便捷函数）

    Args:
        prompt_format: 提示词模板
        csv_path: CSV 文件路径

    Returns:
        提示词列表
    """
    builder = PromptBuilder(prompt_format)
    return builder.build_prompts_from_csv(csv_path)


def build_prompts_from_dataframe(prompt_format: str, df: pd.DataFrame) -> List[str]:
    """
    从 DataFrame 生成提示词列表（便捷函数）

    Args:
        prompt_format: 提示词模板
        df: pandas DataFrame

    Returns:
        提示词列表
    """
    builder = PromptBuilder(prompt_format)
    return builder.build_prompts_from_dataframe(df)


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 示例 1: 基本用法
    print("=" * 60)
    print("示例 1: 基本用法")
    print("=" * 60)

    PROMPT_FORMAT = "把名字变成大写，根据年龄计算出生年份：名字:{name}，年龄:{age}"

    builder = PromptBuilder(PROMPT_FORMAT)
    print(f"模板: {PROMPT_FORMAT}")
    print(f"需要的字段: {builder.get_required_columns()}")

    # 测试单行数据
    row = {"name": "Alice", "age": 18}
    prompt = builder.build_prompt(row)
    print(f"\n输入数据: {row}")
    print(f"生成提示词: {prompt}")

    # 示例 2: 从 CSV 读取
    print("\n" + "=" * 60)
    print("示例 2: 从 CSV 读取")
    print("=" * 60)

    try:
        prompts = builder.build_prompts_from_csv("person.csv")
        print(f"从 person.csv 生成了 {len(prompts)} 个提示词:")
        for i, p in enumerate(prompts, 1):
            print(f"{i}. {p}")
    except FileNotFoundError:
        print("person.csv 不存在，跳过此示例")

    # 示例 3: 验证模板
    print("\n" + "=" * 60)
    print("示例 3: 验证模板")
    print("=" * 60)

    validation = builder.validate_template(["name", "age", "city"])
    print(f"验证结果: {'[OK] 有效' if validation['valid'] else '[FAIL] 无效'}")
    print(f"占位符: {validation['placeholders']}")
    print(f"可用列: {validation['available_columns']}")
    print(f"缺失列: {validation['missing_columns']}")
    print(f"额外列: {validation['extra_columns']}")

    # 示例 4: 带元数据的输出
    print("\n" + "=" * 60)
    print("示例 4: 带元数据的输出")
    print("=" * 60)

    try:
        df = pd.read_csv("person.csv")
        results = builder.build_prompts_with_metadata(df)
        for i, r in enumerate(results, 1):
            print(f"{i}. 原始数据: {r['row_data']}")
            print(f"   提示词: {r['prompt']}")
    except FileNotFoundError:
        print("person.csv 不存在，跳过此示例")

    # 示例 5: 不同模板测试
    print("\n" + "=" * 60)
    print("示例 5: 不同模板测试")
    print("=" * 60)

    test_templates = [
        "请介绍 {name}，TA 今年 {age} 岁",
        "用户: {name}, 年龄: {age} - 生成一句问候语",
        "NAME={name}, AGE={age} - 转换成 JSON",
    ]

    test_row = {"name": "Bob", "age": 25}

    for template in test_templates:
        b = PromptBuilder(template)
        p = b.build_prompt(test_row)
        print(f"模板: {template}")
        print(f"结果: {p}\n")

"""
LLM 结果解析器

功能：
1. 解析大模型返回的 JSON 响应
2. 根据 settings.py 配置将 llm_answer 转换为数字 label
3. 支持二分类和多类别分类
4. 导出为 CSV 格式
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

# 尝试导入 settings，如果失败则使用默认配置
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import settings
except ImportError:
    # 默认配置
    class DefaultSettings:
        TASK_TYPE = "binary"
        BINARY_CLASS_CONFIG = {
            "positive_label": "是",
            "negative_label": "否",
        }
        MULTICLASS_CONFIG = {
            "classes": [],
        }
    settings = DefaultSettings()


class LLMParser:
    """LLM 结果解析器"""

    def __init__(
        self,
        task_type: Optional[str] = None,
        binary_config: Optional[Dict[str, str]] = None,
        multiclass_config: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化解析器

        Args:
            task_type: 任务类型，"binary" 或 "multiclass"，如果为 None 则从 settings 读取
            binary_config: 二分类配置，如果为 None 则从 settings 读取
            multiclass_config: 多类别分类配置，如果为 None 则从 settings 读取
        """
        self.task_type = task_type or getattr(settings, "TASK_TYPE", "binary")

        if binary_config is None:
            self.binary_config = getattr(settings, "BINARY_CLASS_CONFIG", {
                "positive_label": "是",
                "negative_label": "否",
            })
        else:
            self.binary_config = binary_config

        if multiclass_config is None:
            self.multiclass_config = getattr(settings, "MULTICLASS_CONFIG", {
                "classes": [],
            })
        else:
            self.multiclass_config = multiclass_config

        # 构建多类别分类的标签映射
        self._build_label_mapping()

    def _build_label_mapping(self):
        """构建标签映射字典"""
        if self.task_type == "binary":
            self.label_to_idx = {
                self.binary_config["positive_label"]: 1,
                self.binary_config["negative_label"]: 0,
            }
            self.idx_to_label = {
                1: self.binary_config["positive_label"],
                0: self.binary_config["negative_label"],
            }
        elif self.task_type == "multiclass":
            classes = self.multiclass_config["classes"]
            self.label_to_idx = {label: idx for idx, label in enumerate(classes)}
            self.idx_to_label = {idx: label for idx, label in enumerate(classes)}
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")

    @staticmethod
    def extract_json(response_str: str) -> Optional[Dict[str, Any]]:
        """
        从字符串中提取 JSON

        Args:
            response_str: 可能包含 JSON 的字符串

        Returns:
            解析后的 JSON 字典，如果失败返回 None
        """
        if not response_str:
            return None

        # 尝试直接解析
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            pass

        # 尝试用正则提取 JSON 部分
        # 匹配从 { 到 } 的内容，支持嵌套
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_str)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def parse_response(self, response_str: str) -> Dict[str, Any]:
        """
        解析单个 LLM 响应

        Args:
            response_str: LLM 返回的字符串

        Returns:
            解析结果，包含:
                - llm_answer: 原始答案
                - label: 转换后的数字标签
                - reason: 理由（如果有）
                - success: 是否解析成功
                - error: 错误信息（如果有）
        """
        result = {
            "llm_answer": None,
            "label": None,
            "reason": None,
            "success": False,
            "error": None,
        }

        # 提取 JSON
        json_data = self.extract_json(response_str)

        if json_data is None:
            result["error"] = "无法解析 JSON"
            return result

        # 提取 llm_answer
        llm_answer = json_data.get("llm_answer")
        # if llm_answer is None:
        #     # 尝试其他可能的字段名
        #     for key in ["answer", "label", "classification", "result"]:
        #         if key in json_data:
        #             llm_answer = json_data[key]
        #             break

        if llm_answer is None:
            result["error"] = "找不到 llm_answer 字段"
            return result

        result["llm_answer"] = llm_answer
        result["reason"] = json_data.get("reason", "")

        # 转换为 label
        try:
            label = self.text_to_label(llm_answer)
            result["label"] = label
            result["success"] = True
        except ValueError as e:
            result["error"] = str(e)

        return result

    def text_to_label(self, text: str) -> int:
        """
        将文本答案转换为数字标签

        Args:
            text: 文本答案

        Returns:
            数字标签

        Raises:
            ValueError: 当无法匹配到标签时
        """
        text = str(text).strip()

        # 精确匹配
        if text in self.label_to_idx:
            return self.label_to_idx[text]

        # 尝试模糊匹配（去掉空格、转小写等）
        text_normalized = text.lower().replace(" ", "").replace("　", "")
        for label, idx in self.label_to_idx.items():
            label_normalized = label.lower().replace(" ", "").replace("　", "")
            if text_normalized == label_normalized:
                return idx
            # 包含关系
            if label_normalized in text_normalized or text_normalized in label_normalized:
                return idx

        raise ValueError(f"无法将 '{text}' 转换为标签，可用标签: {list(self.label_to_idx.keys())}")

    def label_to_text(self, label: int) -> str:
        """
        将数字标签转换回文本

        Args:
            label: 数字标签

        Returns:
            文本标签
        """
        if label not in self.idx_to_label:
            raise ValueError(f"标签 {label} 不在可用标签中: {list(self.idx_to_label.keys())}")
        return self.idx_to_label[label]

    def parse_database_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析数据库中的所有记录

        Args:
            records: 数据库记录列表，每条记录需包含 'prompt' 和 'response' 字段

        Returns:
            解析结果列表
        """
        results = []

        for record in records:
            prompt = record.get("prompt", "")
            response_str = record.get("response", "")

            parsed = self.parse_response(response_str)

            results.append({
                "prompt": prompt,
                "llm_answer": parsed["llm_answer"],
                "label": parsed["label"],
                "reason": parsed["reason"],
                "success": parsed["success"],
                "error": parsed["error"],
            })

        return results

    def export_to_csv(
        self,
        parsed_results: List[Dict[str, Any]],
        output_path: str,
        include_reason: bool = True,
        include_errors: bool = False,
    ) -> str:
        """
        将解析结果导出为 CSV

        Args:
            parsed_results: parse_database_records 返回的结果列表
            output_path: 输出 CSV 文件路径
            include_reason: 是否包含 reason 列
            include_errors: 是否包含解析失败的记录

        Returns:
            输出文件路径
        """
        # 过滤数据
        if not include_errors:
            parsed_results = [r for r in parsed_results if r["success"]]

        # 构建 DataFrame
        data = []
        for r in parsed_results:
            row = {
                "prompt": r["prompt"],
                "llm_answer": r["llm_answer"],
                "label": r["label"],
            }
            if include_reason:
                row["reason"] = r["reason"]
            data.append(row)

        df = pd.DataFrame(data)

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 导出
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        return output_path


# ============================================
# 便捷函数
# ============================================

def parse_and_export_from_db(
    db_path: str,
    output_path: str,
    task_type: Optional[str] = None,
    binary_config: Optional[Dict[str, str]] = None,
    multiclass_config: Optional[Dict[str, List[str]]] = None,
    limit: Optional[int] = None,
    include_reason: bool = True,
    include_errors: bool = False,
) -> str:
    """
    从数据库读取数据并导出为 CSV（一站式函数）

    步骤：
    1. 从数据库读取出全部数据
    2. 根据配置判断当前解析是二分类还是多类别分类，准备对应的 label 值
    3. 导出 CSV，包含 prompt、llm_answer 和 label 属性

    Args:
        db_path: SQLite 数据库路径
        output_path: 输出 CSV 文件路径
        task_type: 任务类型，"binary" 或 "multiclass"
        binary_config: 二分类配置
        multiclass_config: 多类别分类配置
        limit: 限制读取的条数
        include_reason: 是否包含 reason 列
        include_errors: 是否包含解析失败的记录

    Returns:
        输出文件路径
    """
    # 导入 CacheDB（延迟导入避免循环依赖）
    from .api_with_cache import CacheDB

    # 1. 从数据库读取数据
    db = CacheDB(db_path)
    records = db.get_all(limit=limit)

    # 2. 解析
    parser = LLMParser(
        task_type=task_type,
        binary_config=binary_config,
        multiclass_config=multiclass_config,
    )
    parsed_results = parser.parse_database_records(records)

    # 3. 导出 CSV
    output_path = parser.export_to_csv(
        parsed_results=parsed_results,
        output_path=output_path,
        include_reason=include_reason,
        include_errors=include_errors,
    )

    return output_path


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM 结果解析器 - 使用示例")
    print("=" * 60)

    # 示例 1: 基本用法 - 二分类
    print("\n示例 1: 二分类解析")
    print("-" * 60)

    parser_binary = LLMParser(
        task_type="binary",
        binary_config={
            "positive_label": "是",
            "negative_label": "否",
        }
    )

    test_responses = [
        '{"llm_answer": "是", "reason": "企业主营新能源业务"}',
        '{"llm_answer": "否", "reason": "传统商贸企业"}',
        '有些额外文本 {"llm_answer": "是", "reason": "测试"} 有些额外文本',
    ]

    for resp in test_responses:
        result = parser_binary.parse_response(resp)
        print(f"输入: {resp[:50]}...")
        print(f"  llm_answer: {result['llm_answer']}")
        print(f"  label: {result['label']}")
        print(f"  success: {result['success']}")
        print()

    # 示例 2: 多类别分类
    print("\n示例 2: 多类别分类解析")
    print("-" * 60)

    parser_multiclass = LLMParser(
        task_type="multiclass",
        multiclass_config={
            "classes": ["体育", "财经", "科技", "娱乐"],
        }
    )

    multiclass_responses = [
        '{"llm_answer": "科技", "reason": "关于人工智能"}',
        '{"llm_answer": "财经", "reason": "股票分析"}',
    ]

    for resp in multiclass_responses:
        result = parser_multiclass.parse_response(resp)
        print(f"输入: {resp}")
        print(f"  llm_answer: {result['llm_answer']}")
        print(f"  label: {result['label']}")
        print()

    # 示例 3: 实际使用 - 从数据库导出
    print("\n示例 3: 从数据库导出 CSV")
    print("-" * 60)
    print("使用方式:")
    print("""
    from llm2bert.llm_api.parser import parse_and_export_from_db

    output_file = parse_and_export_from_db(
        db_path="api_cache.db",
        output_path="parsed_results.csv",
        task_type="binary",  # 或 "multiclass"
    )

    print(f"结果已导出到: {output_file}")
    """)

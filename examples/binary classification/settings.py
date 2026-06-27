"""
配置文件:
修改 PROMPT_FORMAT 即可自定义提示词模板
占位符格式: ${列名}
"""

with open("gemini-prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

PROMPT_FORMAT = prompt_template

DB_PATH = "api_cache.db"

# CSV 文件路径
CSV_FILE = "北京_2020.csv"

# 二分类配置 (TASK_TYPE = "binary" 时使用)
TASK_TYPE = "binary"
# 二分类标签映射
# label 1 代表"是"，label 0 代表"否"
BINARY_CLASS_CONFIG = {
    "positive_label": "是",  # 对应 label 1
    "negative_label": "否",  # 对应 label 0
}
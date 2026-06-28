"""
配置文件:
修改 PROMPT_FORMAT 即可自定义提示词模板
占位符格式: ${列名}
"""

# 提示词模板配置（在此处修改你的模板）
PROMPT_FORMAT = """
    把下述给定的文本分类到以下类别中：['World', 'Sports', 'Business', 'Sci/Tech']。
    待分类文本: ${text}
    按照下述格式返回，要输出完整的json格式的数据：
    {
        "reason": "简要说明为什么将文本分类到该类别中",
        "llm_answer": "类别名",
    }
    """.strip()

DB_PATH = "api_cache.db"

# CSV 文件路径
CSV_FILE = "train.csv"

# 多类别分类配置 (TASK_TYPE = "multiclass" 时使用)
TASK_TYPE = "multiclass"

MULTICLASS_CONFIG = {
    "classes": ["World", "Sports", "Business", "Sci/Tech"],  # 在此处填入所有类别名称
}

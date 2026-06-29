"""
配置文件:
修改 PROMPT_FORMAT 即可自定义提示词模板
占位符格式: ${列名}
"""

with open("gemini-prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

PROMPT_FORMAT = prompt_template

BERT_FORMAT = """
请对以下处于待审状态的企业进行精准判定：
- 企业名称：${企业名称}
- 经营范围：${经营范围}
- 预分类战新产业：战新产业二级分类名称：${战新产业二级分类名称}；战新产业三级分类名称：${战新产业三级分类名称}
"""


DB_PATH = "api_cache.db"

# CSV 文件路径
CSV_FILE = "北京_2020.csv"

# TASK_TYPE: 任务类型，用于区分NLP任务（文本分类、实体识别等），目前仅实现文本分类
TASK_TYPE = "binary"

MULTICLASS_CONFIG = {
    "classes": ["否", "是"],  # label 0 代表"否"，label 1 代表"是"
}
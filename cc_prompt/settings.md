# prompt_1 统一类别分类设置

@examples/binary classification/settings.py  有 BINARY_CLASS_CONFIG。
@examples/multi_classification/settings.py  有 MULTICLASS_CONFIG。

上述两个settings.py无需单独区分二分类与多类别分类，二分类也完全可以使用MULTICLASS_CONFIG进行表示。
比如：
MULTICLASS_CONFIG = {
    "classes": ["否", "是"],
}
请你调整相应的代码，把二分类相关的参数给删除，统一使用 MULTICLASS_CONFIG 管理所有的文本分类。

保留 TASK_TYPE，该参数我的本意是用于区分NLP的文本分类、实体识别等任务，虽然目前只实现了文本分类功能。
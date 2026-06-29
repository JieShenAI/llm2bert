我正在搭建一个通用化的包，我的雄心是支持NLP的多种任务，目前暂时先实现二分类与多类别分类。

在运行 `test_async_api.py` 后，得到的结果如下所示，llm_answer 是大模型回答的结果。接下来你要编写大模型解析的代码。

```
{'response': {
  "llm_answer": "否",
  "reason": "主营教育咨询销售培训，不涉及知识产权相关服务。"
}, 'success': True, 'error': None}
```

你要编写的代码如下：

`settings.py`: 在其中设置二分类还是多类别分类的解析。如果是二分类 label 1 代表是，label 0代表否。如果是多类别分类，则要在settings.py设置所有的类别。

接下来，你要编写 llm2bert/src/llm2bert/llm_api/parser.py，在其中实现对大模型返回结果的解析。并把解析的结果导出为csv，csv有prompt、llm_answer和label属性（label属性为数字）。

若要把数据库中的数据导出为csv，那脚本运行的步骤如下：

1. 从数据库读取出全部数据；
2. parser.py 根据 settings.py 里面的设置信息，判断当前解析是二分类还是多类别分类，准备对应的label值；
3. 导出csv，包含prompt、llm_answer和label属性（label属性为数字）；


# Prompt 1 字段调整

## 数据库字段调整

在 test_async_api.py 构建提示词，也需要把 PromptBuilder.placeholders 对应的键值对，以json格式保存到数据库中。数据库增加一个 attr 字段存储该json数据。

在 export_csv.py 把数据库的数据导出来时，需要把 attr 转换为字典，其key作为csv的属性，从而把attr的数据也保存到csv中。

## 字段调整

下述是运行 `2_export_csv.py` 后，导出的一条数据的示例。

原始导出结果：
 ```json
{'prompt': '把下述给定的文本分类到以下类别中：[\'World\', \'Sports\', \'Business\', \'Sci/Tech\']。\n    待分类文本: Dorman #146;s been dandy of late Revolution coach Steve Nicol is not taking credit for the emergence of Andy Dorman. But Nicol\'s tactical moves helped place Dorman in a position to score his first two goals as a professional, in the final seconds of a 3-0 win at Dallas Wednesday and on his first touch of Saturday night\'s game at D.C. United for the Revolution\'s second ...\n    按照下述格式返回，要输出完整的json格式的数据：\n    {\n        "reason": "简要说明为什么将文本分类到该类别中",\n        "llm_answer": "类别名",\n    }',
 'llm_answer': 'Sports',
 'label': 1,
 'reason': '该文本围绕足球赛事、球队教练与职业球员的赛场表现展开，属于体育领域相关内容',
 'label_name': 'Sports',
 'text': "Dorman #146;s been dandy of late Revolution coach Steve Nicol is not taking credit for the emergence of Andy Dorman. But Nicol's tactical moves helped place Dorman in a position to score his first two goals as a professional, in the final seconds of a 3-0 win at Dallas Wednesday and on his first touch of Saturday night's game at D.C. United for the Revolution's second ...",
 'true_label': 1}
 ```
我想让你参考 `1_sglang_infer.py` 对导出csv字段的处理，修改`2_export_csv.py`的代码，让其达到下述目标导出结果的效果。
你要做如下的修改：
1. label 改名为 llm_pred_label；
2. 增加 llm_resp 字段，其是大模型回答的完整文本；
3. 增加 success 字段，其代表 parser 解析是否成功；
4. 增加 error 字段，若parser解析失败，在这里显示解析失败详情；

目标导出结果：
 ```json
{'prompt': '把下述给定的文本分类到以下类别中：[\'World\', \'Sports\', \'Business\', \'Sci/Tech\']。输出结果按照json格式返回。\n【示例】：\ntext: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.\noutput: {""reason"": ""该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为\'Sports\'。"", ""llm_answer"": ""Sports""}\n【待分类文本】:\ntext: Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.\noutput:',
 'llm_resp': '{"reason": "该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为\'Sports\'。","llm_answer": "Sports"}',
 'llm_answer': 'Sports',
 'reason': "该文本提到的事件是Vijay Singh赢得PGA锦标赛，PGA锦标赛是高尔夫球赛事，属于体育范畴，因此该文本应归类为'Sports'。",
 'llm_pred_label': 1,
 'success': True,
 'error': nan,
 'text': 'Singh Snares PGA Title Vijay Singh outlasts Justin Leonard and Chris DiMarco in a three-way playoff to win the PGA Championship on Sunday at Whistling Straits in Haven, Wisconsin.',
 'true_label': 1,
 'label_name': 'Sports'}
 ```



# Prompt 2 大模型预测结果 split

从大模型预测的结果 @examples/binary classification/llm_parsed_results.csv  中，切分出 训练集和测试集。 请你编写 @examples/multi_classification/llm_pred_split2test.py  代码，其接收一个参数 per_num_cls，该参数表每个类别的样本数量。

 ``` python llm_pred_split2test.py --per_num_cls 50 ``` 

该脚本运行后，会在 ./data 文件夹里面 创建 train.csv 和 test.csv。若data文件夹不存在，则自动创建。
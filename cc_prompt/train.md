[cc-prompt start]

## 数据集代码编写

@examples/multi_classification/llm_parsed_results.csv  的字段如下：

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

在 @examples/multi_classification/settings.py  的 BERT_FORMAT 定义了使用 llm_parsed_results.csv 里面的字段作为BERT的输入文本封装形式，llm_pred_label 用作训练模型的label。

你要在 @src/llm2bert/bert_finetune/data.py  定义一个数据集类，其接收 一个csv表格名(llm_parsed_results.csv )和BERT_FORMAT，其通过读取该表格实现数据集。

## 文本分类代码编写

你要使用 AutoModelForSequenceClassification 在 @src/llm2bert/bert_finetune/model.py  中定义文本分类的Class，其forward方法的输入参数与返回值如下所示：

```python
def forward(self, text_tokens, labels=None):
    output = self.model(**text_tokens)
    ...
    if labels is not None:
        return {
            "logits": logits,
            "loss": self.ce_loss(logits, labels),
        }
    return {"logits": logits}
```

## datacollator_class 代码编写

在 @src/llm2bert/bert_finetune/data.py  定义 datacollator_class，用于 @src/llm2bert/bert_finetune/trainer.py  的Trainer的data_collator参数。

self.datacollator_class(
    data_args=self.data_args, tokenizer=self.tokenizer
),

self.trainer = Trainer(
    model=self.model,
    args=self.training_args,
    train_dataset=self.train_dataset,
    eval_dataset=self.eval_dataset,
    data_collator=self.datacollator_class(
        data_args=self.data_args, tokenizer=self.tokenizer
    ),
    compute_metrics=compute_metrics,
)

## train 脚本实现

在 @examples/multi_classification/train.py  编写新的 TrainerUtil 类继承 @src/llm2bert/bert_finetune/trainer.py  的TrainerUtil，要实现它的 set_model、set_dataset、set_datacollator的重写。

[cc-prompt end]
# prompt1 训练框架搭建

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

# prompt2 预测代码实现

参考 train.sh 的实现，帮我设计 predict.sh 传递csv表格名(csv_file)、本地训练好的模型作为参数，对该表格的数据作为预测。完成相应代码的编写。
预测的结果保存为 bert_predict_{csv_file}.csv。在原始 csv_file 所有字段的基础上增加 bert_pred_label:int 和 bert_pred_answer:str 表示类别名。

反馈prompt:
你应该使用 @TrainerUtilForMultiClass(TrainerUtil) 里面的代码实现预测，若里面没有预测代码，你需要为 TrainerUtilForMultiClass 编写预测的方法。你要使用 @src/llm2bert/bert_finetune/model.py  的 SequenceClassificationModel 模型，而不是在 @examples/multi_classification/predict.py  中使用  AutoModelForSequenceClassification

反馈prompt:
我记得 transformer的trainer有predict方法，你为什么要使用 for i in range(0, len(texts), batch_size): 。哪种方式做预测效率高

# prompt3 数据集样本的平衡

我想在 @src/llm2bert/bert_finetune/model.py  的SequenceClassificationModel 中, 使用 Focal Loss 降低容易分类的大类样本的损失权重，聚焦难分的少数类，用于应对不平衡数据集的分类。

按照下述几个步骤进行：

1. ModelArguments 增加 use_focal_loss: bool 参数，默认是False。若为True，则启用 FocalLoss 的计算。

2. 在 model.py 中 增加 **自动根据各类样本数量计算初始权重，再配合 Focal Loss 的 γ 参数共同调节，无需手动硬写权重。**权重严格和样本反比，数学上均衡各类原始贡献；

   



# prompt4 模型训练的csv文件导入



TrainerUtilForMultiClass 的数据集的csv文件，改成从 argument参数导入。

@arguments.py 的 DataArguments 增加一个 train_val_csv 参数，用于指定csv。
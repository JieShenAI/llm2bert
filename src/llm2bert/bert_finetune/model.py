## 文本分类模型

from torch import nn
from transformers import AutoModelForSequenceClassification


class SequenceClassificationModel(nn.Module):
    """使用 AutoModelForSequenceClassification 的文本分类模型。

    forward 方法接受 text_tokens（tokenizer 输出）和可选的 labels，
    返回 logits 和（当 labels 不为 None 时）loss。
    """

    def __init__(self, model_args, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_args.hf_name,
            num_labels=num_labels,
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, text_tokens, labels=None):
        output = self.model(**text_tokens)
        logits = output.logits

        if labels is not None:
            return {
                "logits": logits,
                "loss": self.ce_loss(logits, labels),
            }
        return {"logits": logits}

    def save_pretrained(self, save_directory):
        """保存底层 HuggingFace 模型。"""
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_args, num_labels, pretrained_model_name_or_path):
        """从本地目录或 HuggingFace Hub 加载模型。"""
        model = cls(model_args, num_labels)
        model.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
        )
        return model

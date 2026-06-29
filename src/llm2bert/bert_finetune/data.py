import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorWithPadding

from ..llm_api.prompt_builder import PromptBuilder
from .arguments import DataArguments


class ClassificationDataset(Dataset):
    """从本地 CSV 文件读取数据集的封装类。

    使用 BERT_FORMAT 模板对文本进行格式化（支持任意 ${列名} 占位符），
    并返回模型训练所需的 {"text": ..., "label": ...} 格式。
    """

    def __init__(
        self,
        csv_file: str,
        bert_format: str,
        label_column: str = "llm_pred_label",
    ):
        self.df = pd.read_csv(csv_file)
        self.prompt_builder = PromptBuilder(bert_format)
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.prompt_builder.build_prompt(row.to_dict())
        label = int(row[self.label_column])
        return {"text": text, "label": label}


class DiyDataCollator(DataCollatorWithPadding):

    def __init__(self, data_args: DataArguments, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.data_args = data_args

    def __call__(self, features):
        texts = [f["text"] for f in features]
        labels = [f["label"] for f in features]
        labels = torch.tensor(labels)

        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.data_args.text_max_length,
            return_tensors="pt",
        )

        return {
            "text_tokens": text_tokens,
            "labels": labels,
        }

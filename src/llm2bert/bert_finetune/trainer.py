import logging
import os
import sys
from typing import Union, Literal
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

from .arguments import DataArguments, ModelArguments

# from .model import DiyConfig, DiyModel


def compute_metrics(eval_pred):
    """
    计算文本分类任务的评估指标

    Args:
        eval_pred: 包含预测结果和真实标签的EvalPrediction对象
            - predictions: 模型输出的原始logits
            - label_ids: 真实标签
        eval_pred: 包含预测结果和真实标签的EvalPrediction对象
            - predictions: 模型输出的原始logits
            - label_ids: 真实标签

    Returns:
        包含评估指标的字典
    """
    # 从EvalPrediction对象中获取预测结果和真实标签
    # predictions, label = eval_pred

    if isinstance(eval_pred.predictions, tuple):
        if len(eval_pred.predictions) >= 2:
            logits = eval_pred.predictions[0]
    else:
        # 兼容未返回alpha的情况（兜底）
        logits = eval_pred.predictions
        # alpha = None

    # 真实标签
    labels = eval_pred.label_ids

    # 将logits转换为预测类别（取概率最大的类别）
    preds = np.argmax(logits, axis=-1)

    # 计算准确率
    accuracy = accuracy_score(labels, preds)

    # 计算精确率、召回率和F1分数（支持多类别和二分类）
    # average参数：'micro'、'macro'、'weighted'或None
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    # 返回评估指标字典
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class TrainerUtil:
    def __init__(self):
        self.last_checkpoint = None
        # self.datacollator_class = datacollator_class
        self.start_up()

        # 如果 best_model 已存在则跳过训练（除非指定了 overwrite_output_dir）
        self.best_model_exists = os.path.exists(self.best_model_dir)

        if self.training_args.do_train and self.best_model_exists and not self.training_args.overwrite_output_dir:
            self.trainer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.hf_name
            )
            self.set_dataset()
            self.logger.info(self.train_dataset[0])
            self.logger.info(
                f"train on {len(self.train_dataset)} samples, eval on {len(self.eval_dataset)} samples"
            )
            self.set_model()
            self._init_focal_loss()
            self.set_datacollator()
            self.set_trainer()

    
    def set_model(self):
        # set self.model
        raise NotImplementedError

    def set_dataset(self):
        """
        若数据集加载的方式不一样，需要重写该函数
        :return:
        """
        # set self.train_dataset
        # set self.eval_dataset
        raise NotImplementedError

    def set_datacollator(self):
        # set self.datacollator_class
        raise NotImplementedError

    def start_up(self):
        self.logger = logging.getLogger()
        self.general_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.general_formatter)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("root logger")

        parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))

        if len(sys.argv) == 3 and sys.argv[1] == "--config":
            json_path = sys.argv[2]
            self.data_args, self.model_args, self.training_args = (
                parser.parse_json_file(json_file=json_path)
            )
        else:
            self.data_args, self.model_args, self.training_args = (
                parser.parse_args_into_dataclasses()
            )

        self.data_args: DataArguments
        self.model_args: ModelArguments
        self.training_args: TrainingArguments
        self.best_model_dir = os.path.join(self.training_args.output_dir, "best_model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.hf_name
        )

    def set_trainer(self):
        # 自定义 DataCollator 使用 "text" 键，禁止 Trainer 自动删除未使用列
        self.training_args.remove_unused_columns = False
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

    def _init_focal_loss(self):
        """若启用 Focal Loss，自动从 train_dataset 收集各类样本数并设置权重。"""
        if not getattr(self.model_args, "use_focal_loss", False):
            return

        # 遍历数据集收集所有标签
        labels_list = []
        for i in range(len(self.train_dataset)):
            item = self.train_dataset[i]
            if isinstance(item, dict):
                label = item.get("label") or item.get("labels")
            elif isinstance(item, (list, tuple)):
                label = item[1]
            else:
                continue
            if label is not None:
                labels_list.append(int(label))

        if not labels_list:
            self.logger.warning(
                "⚠️ Focal Loss 已启用，但无法从 train_dataset 中提取标签，"
                "将使用默认等权重（仅 gamma 聚焦效果）。"
            )
            return

        labels_tensor = torch.tensor(labels_list)
        num_classes = labels_tensor.max().item() + 1
        counts = torch.zeros(num_classes, dtype=torch.float)
        for c in range(num_classes):
            counts[c] = (labels_tensor == c).sum().float()

        self.logger.info(
            f"📊 Class distribution (for Focal Loss): {counts.tolist()}"
        )
        self.model.set_class_weights_from_counts(counts)
        self.logger.info(
            f"⚖️  Focal Loss α weights: "
            f"{self.model.focal_loss.alpha.tolist()}"
        )

    def save(self, save_model_dir):
        self.tokenizer.save_pretrained(save_model_dir)
        self.model.save_pretrained(save_model_dir)

    def train(self):
        # 检查是否有上次的 checkpoint
        self.last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and not self.training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(self.training_args.output_dir)

            if self.last_checkpoint is not None:
                self.logger.info(
                    f"⚡ Found checkpoint at {self.last_checkpoint}. Resuming training."
                )
            else:
                self.logger.warning("❌ No checkpoint found. Starting fresh training.")

        else:
            self.logger.info(
                "➡️ No previous output_dir or overwrite_output_dir=True, starting from scratch."
            )

        self.trainer.train(
            resume_from_checkpoint=(
                self.last_checkpoint if self.last_checkpoint else None
            )
        )
        self.save(self.best_model_dir)

    def evaluate(self, dataset: Union[Literal["eval", "test"], datasets.Dataset]):
        eval_logger = logging.getLogger("evaluate")
        file_handler = logging.FileHandler(
            os.path.join(self.training_args.output_dir, "eval.log"),
            encoding="utf-8",
            mode="a+",
        )
        file_handler.setFormatter(self.general_formatter)
        eval_logger.addHandler(file_handler)
        eval_logger.setLevel(logging.INFO)

        if isinstance(dataset, str):
            if dataset not in ("eval", "test"):
                raise ValueError(
                    f"字符串类型的dataset只能是'eval'或'test'，但收到：{dataset}"
                )
            if dataset == "eval":
                metric = self.trainer.evaluate(self.eval_dataset)
                eval_logger.info(f"<<<<< Eval on eval_dataset *****")
            elif dataset == "test":
                metric = self.trainer.evaluate(self.test_dataset)
                eval_logger.info(f"<<<<< Eval on test_dataset *****")
        else:
            metric = self.trainer.evaluate(dataset)

        eval_logger.info(f"Model: {self.model_args.hf_name}")
        eval_logger.info(f"Metrics: {metric}")
        eval_logger.info(f"***** End Eval >>>>>\n")
        return metric

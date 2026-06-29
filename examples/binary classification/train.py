"""
BERT 多分类微调训练脚本

基于 llm_parsed_results.csv 中的 LLM 解析结果训练文本分类模型。

用法：
    python train.py \
        --output_dir ./output \
        --hf_name google-bert/bert-base-uncased \
        --num_label 4 \
        --do_train

或通过 JSON 配置文件：
    python examples/multi_classification/train.py \
        --config ./config.json
"""

import os
import sys
from pathlib import Path

import logging

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer

import torch
from sklearn.model_selection import train_test_split

# 将项目根目录加入 sys.path，便于直接运行（非 pip 安装模式）
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm2bert.bert_finetune.trainer import TrainerUtil
from llm2bert.bert_finetune.data import (
    ClassificationDataset,
    DiyDataCollator,
    PredictDataset,
)
from llm2bert.bert_finetune.model import SequenceClassificationModel
from llm2bert.bert_finetune.arguments import ModelArguments, DataArguments

from settings import BERT_FORMAT, MULTICLASS_CONFIG


class TrainerUtilForMultiClass(TrainerUtil):
    """多分类 BERT 微调训练器。

    从本地 CSV 读取 LLM 解析结果数据，使用 BERT_FORMAT 格式化输入文本，
    基于 AutoModelForSequenceClassification 进行序列分类微调。

    需要重写的方法：
    - set_model
    - set_dataset
    - set_datacollator
    """

    def __init__(self, bert_format=None):
        self._bert_format = bert_format or BERT_FORMAT
        super().__init__()

    def start_up(self):
        super().start_up()
        # 从 settings.py 的 MULTICLASS_CONFIG 中获取类别数量
        self.logger.info(
            f"num_label = {len(MULTICLASS_CONFIG["classes"])}, "
            f"classes = {MULTICLASS_CONFIG['classes']}"
        )
        if self.data_args.train_val_csv:
            self.logger.info(f"train_val_csv = {self.data_args.train_val_csv}")
        else:
            self.logger.warning(
                "train_val_csv 未指定，请通过 --train_val_csv 参数传入 CSV 文件路径"
            )

    def set_model(self):
        """使用 AutoModelForSequenceClassification 构建模型。"""
        self.model = SequenceClassificationModel(
            self.model_args, num_labels=len(MULTICLASS_CONFIG["classes"])
        )
        self.logger.info(
            f"Model loaded: {self.model_args.hf_name}, "
            f"num_labels = {len(MULTICLASS_CONFIG['classes'])}"
        )

    def set_dataset(self):
        """从 CSV 读取数据集，划分为训练集和评估集。"""
        csv_file = self.data_args.train_val_csv
        if csv_file is None:
            raise ValueError(
                "请通过 --train_val_csv 参数指定训练/验证 CSV 文件路径"
            )
        full_dataset = ClassificationDataset(
            csv_file,
            self._bert_format,
            label_column="llm_pred_label",
        )
        self.logger.info(f"Loaded {len(full_dataset)} samples from {csv_file}")

        indices = list(range(len(full_dataset)))
        train_idx, eval_idx = train_test_split(indices, test_size=0.2, random_state=42)

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.eval_dataset = torch.utils.data.Subset(full_dataset, eval_idx)
        self.test_dataset = self.eval_dataset  # 兼容基类的 evaluate 方法

    def set_datacollator(self):
        """使用 DiyDataCollator 作为 DataCollator。"""
        self.datacollator_class = DiyDataCollator

    @classmethod
    def predict(cls, csv_file, model_dir, batch_size=32, text_max_length=512):
        """使用训练好的模型对 CSV 数据进行预测。

        内部使用 HuggingFace Trainer.predict()，通过 DataLoader 进行
        高效批量推理（支持多进程数据加载、自动 device 分配等）。

        从 model_dir/best_model/ 加载 SequenceClassificationModel 和 tokenizer，
        对 csv_file 中的文本进行批量推理，返回带 bert_pred_label 和 bert_pred_answer 的 DataFrame。

        Args:
            csv_file: 待预测的 CSV 文件路径。
            model_dir: 训练时 --output_dir 指定的目录（需包含 best_model/ 子目录）。
            batch_size: 预测批大小，默认 32。
            text_max_length: 输入文本最大 token 长度，默认 512。

        Returns:
            pd.DataFrame: 原始 CSV 所有列 + bert_pred_label(int) + bert_pred_answer(str)。
        """

        classes = MULTICLASS_CONFIG["classes"]
        num_labels = len(classes)
        best_model_path = os.path.join(model_dir, "best_model")

        if not os.path.isdir(best_model_path):
            raise FileNotFoundError(
                f"模型目录不存在: {best_model_path}\n"
                f"请确认 model_dir 指向训练时 --output_dir 指定的目录"
            )

        logger = logging.getLogger("TrainerUtilForMultiClass.predict")
        logger.info(f"📦 加载模型: {best_model_path}")

        # 使用项目内的 SequenceClassificationModel
        model_args = ModelArguments(hf_name=best_model_path)
        model = SequenceClassificationModel.from_pretrained(
            model_args,
            num_labels,
            best_model_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)

        predict_dataset = PredictDataset(csv_file, BERT_FORMAT)
        logger.info(
            f"📄 加载 {len(predict_dataset)} 条数据，开始预测 "
            f"(batch_size={batch_size}, max_length={text_max_length})..."
        )

        # 配置 Trainer（关闭训练相关功能，仅用于 predict）
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            report_to="none",
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
            logging_first_step=False,
        )
        data_args = DataArguments(text_max_length=text_max_length)
        collator = DiyDataCollator(data_args, tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        # Trainer.predict() 自动处理：DataLoader 分批 → device 分配 → 推理 → 结果拼接
        logger.info("🚀 开始推理...")
        predictions = trainer.predict(predict_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        logger.info(f"✅ 推理完成，共 {len(preds)} 条")

        # 添加预测结果列
        df = pd.read_csv(csv_file)
        df["bert_pred_label"] = preds.tolist()
        df["bert_pred_answer"] = df["bert_pred_label"].apply(lambda x: classes[x])

        # 统计分布
        logger.info(f"\n📊 预测分布（bert_pred_answer）:")
        for cat, cnt in df["bert_pred_answer"].value_counts().items():
            logger.info(f"  {cat}: {cnt}")

        return df


if __name__ == "__main__":
    trainer_util = TrainerUtilForMultiClass()
    trainer_util.train()


# 参考 train.sh 里面的参数

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

import torch
from sklearn.model_selection import train_test_split

# 将项目根目录加入 sys.path，便于直接运行（非 pip 安装模式）
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

print(os.path.exists(Path(__file__).parent.parent.parent / "src"))

from llm2bert.bert_finetune.trainer import TrainerUtil
from llm2bert.bert_finetune.data import ClassificationDataset, DiyDataCollator
from llm2bert.bert_finetune.model import SequenceClassificationModel

from settings import CSV_FILE, BERT_FORMAT, MULTICLASS_CONFIG


class TrainerUtilForMultiClass(TrainerUtil):
    """多分类 BERT 微调训练器。

    从本地 CSV 读取 LLM 解析结果数据，使用 BERT_FORMAT 格式化输入文本，
    基于 AutoModelForSequenceClassification 进行序列分类微调。

    需要重写的方法：
    - set_model
    - set_dataset
    - set_datacollator
    """

    def __init__(self, csv_file=None, bert_format=None):
        self._csv_file = csv_file or CSV_FILE
        self._bert_format = bert_format or BERT_FORMAT
        super().__init__()

    def start_up(self):
        super().start_up()
        # 从 settings.py 的 MULTICLASS_CONFIG 中获取类别数量
        self.logger.info(
            f"num_label = {len(MULTICLASS_CONFIG["classes"])}, "
            f"classes = {MULTICLASS_CONFIG['classes']}"
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
        full_dataset = ClassificationDataset(
            self._csv_file, self._bert_format,
            label_column="true_label",
        )
        self.logger.info(f"Loaded {len(full_dataset)} samples from {self._csv_file}")

        indices = list(range(len(full_dataset)))
        train_idx, eval_idx = train_test_split(indices, test_size=0.2, random_state=42)

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.eval_dataset = torch.utils.data.Subset(full_dataset, eval_idx)
        self.test_dataset = self.eval_dataset  # 兼容基类的 evaluate 方法

    def set_datacollator(self):
        """使用 DiyDataCollator 作为 DataCollator。"""
        self.datacollator_class = DiyDataCollator


if __name__ == "__main__":
    trainer_util = TrainerUtilForMultiClass()
    trainer_util.train()


# python train.py --do_train --output_dir ./output --hf_name google-bert/bert-base-uncased --

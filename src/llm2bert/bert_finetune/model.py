## 文本分类模型

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


class FocalLoss(nn.Module):
    """Focal Loss，用于缓解类别不平衡问题。

    Formula:
        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    其中 p_t = exp(-CE) 是真实类别的模型预估概率。

    α（类别权重）自动从各类样本数计算：weight_c = N_total / (C * N_c)，
    严格反比于样本数，数学上均衡各类的原始贡献。
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape (C,) 或 None，每个类别的权重
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 每个样本的标准 CE loss（无 reduction，无 weight）
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        # p_t = exp(-CE) — 真实类别的预估概率
        p_t = torch.exp(-ce_loss)

        # Focal loss：(1 - p_t)^γ * CE
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # 应用类别权重（若提供）
        if self.alpha is not None:
            alpha_t = self.alpha.to(labels.device).gather(0, labels)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    @staticmethod
    def compute_weight_from_counts(class_counts: torch.Tensor) -> torch.Tensor:
        """根据各类样本数量计算初始权重（严格反比）。

        weight_c = N_total / (C * N_c)

        使得 sum(weight_c) = C，即各类权重的均值为 1，
        每个类别对 loss 的原始贡献被数学均衡。
        """
        total = class_counts.sum()
        C = len(class_counts)
        # 防止除零
        counts = class_counts.float().clamp(min=1)
        weights = total / (C * counts)
        return weights


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

        self.use_focal_loss = getattr(model_args, "use_focal_loss", False)
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(gamma=2.0)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def set_class_weights_from_counts(self, class_counts: torch.Tensor):
        """从各类样本数自动计算并设置类别权重（仅 Focal Loss 模式有效）。"""
        if not self.use_focal_loss:
            return
        weights = FocalLoss.compute_weight_from_counts(class_counts)
        self.focal_loss.alpha = weights

    def forward(self, text_tokens, labels=None):
        output = self.model(**text_tokens)
        logits = output.logits

        if labels is not None:
            if self.use_focal_loss:
                loss = self.focal_loss(logits, labels)
            else:
                loss = self.ce_loss(logits, labels)
            return {
                "logits": logits,
                "loss": loss,
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

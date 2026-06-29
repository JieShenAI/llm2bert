"""
BERT 多分类预测脚本

基于训练好的模型对 CSV 数据进行预测，输出 bert_predict_{csv_file}.csv。
在原始 CSV 所有字段基础上增加 bert_pred_label（int，类别索引）和 bert_pred_answer（str，类别名）。

内部使用 TrainerUtilForMultiClass.predict() 和 SequenceClassificationModel，
保持与训练代码一致的模型加载和数据格式化逻辑。

用法：
    bash predict.sh test.csv ./output

或：
    python predict.py \
        --csv_file test.csv \
        --model_dir ./output
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# 将项目根目录和脚本所在目录加入 sys.path
_SCRIPT_DIR = str(Path(__file__).parent)
_SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
for p in [_SRC_DIR, _SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train import TrainerUtilForMultiClass


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 多分类预测")
    parser.add_argument(
        "--csv_file", type=str, required=True,
        help="待预测的 CSV 文件路径",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="训练输出目录（需包含 best_model/ 子目录）",
    )
    parser.add_argument(
        "--text_max_length", type=int, default=512,
        help="输入文本的最大 token 长度（默认 512）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="预测时的批大小（默认 32）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志（TrainerUtilForMultiClass.predict 内部使用 logging.getLogger）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        stream=sys.stdout,
    )

    # 调用 TrainerUtilForMultiClass 的预测方法
    df = TrainerUtilForMultiClass.predict(
        csv_file=args.csv_file,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        text_max_length=args.text_max_length,
    )

    # 保存结果（文件名：bert_predict_{原文件名}）
    csv_basename = os.path.basename(args.csv_file)
    output_file = f"bert_predict_{csv_basename}"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 预测完成，结果已保存至 {output_file}")

    # 前几条预览
    print(f"👀 预览（前 3 条）：")
    print(df[["bert_pred_label", "bert_pred_answer"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()

# python predict.py --csv_file=data/test.csv --model_dir=output
    
    
"""
从大模型预测结果 (llm_parsed_results.csv) 中，按类别切分出训练集和测试集。

用法:
    python llm_pred_split2test.py --per_num_cls 50

输出:
    ./data/train.csv  - 训练集（每类除去测试样本后的剩余样本）
    ./data/test.csv   - 测试集（每类随机抽取 per_num_cls 条，均衡采样）
"""

import csv
import os
import random
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 LLM 预测结果中按类别切分训练集和测试集"
    )
    parser.add_argument(
        "--per_num_cls",
        type=int,
        required=True,
        help="每个类别抽取的测试样本数量",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="llm_parsed_results.csv",
        help="输入 CSV 文件路径 (默认: llm_parsed_results.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出目录 (默认: ./data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label_name",
        help="用作类别分组的列名 (默认: label_name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # 读取 CSV
    csv_path = args.input_csv
    if not os.path.exists(csv_path):
        print(f"错误: 输入文件不存在: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"读取到 {len(all_rows)} 条记录")

    # 按类别分组
    label_col = args.label_col
    if label_col not in all_rows[0]:
        print(f"错误: 列 '{label_col}' 不存在，可用列: {list(all_rows[0].keys())}")
        return

    class_groups = defaultdict(list)
    for row in all_rows:
        cls_name = row[label_col]
        class_groups[cls_name].append(row)

    print(f"类别分布:")
    for cls_name, rows in sorted(class_groups.items()):
        print(f"  {cls_name}: {len(rows)} 条")

    # 每类随机抽取 per_num_cls 条作为测试集，不足则全部作为测试集
    train_rows = []
    test_rows = []

    per_num_cls = args.per_num_cls
    for cls_name, rows in sorted(class_groups.items()):
        # 打乱顺序以确保随机性
        random.shuffle(rows)

        if len(rows) <= per_num_cls:
            # 样本不足：全部放入测试集，训练集为空
            test_rows.extend(rows)
            print(f"  警告: 类别 '{cls_name}' 仅有 {len(rows)} 条 (< {per_num_cls})，全部放入测试集")
        else:
            test_rows.extend(rows[:per_num_cls])
            train_rows.extend(rows[per_num_cls:])

    # 再次打乱整体顺序
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    # 确保输出目录存在
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # 获取字段名
    fieldnames = all_rows[0].keys()

    # 写入 train.csv
    with open(train_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    # 写入 test.csv
    with open(test_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"\n输出完成!")
    print(f"  训练集: {train_path} ({len(train_rows)} 条)")
    print(f"  测试集: {test_path} ({len(test_rows)} 条)")

    # 打印训练集类别分布
    print(f"\n训练集类别分布:")
    train_counts = defaultdict(int)
    for row in train_rows:
        train_counts[row[label_col]] += 1
    for cls_name in sorted(train_counts.keys()):
        print(f"  {cls_name}: {train_counts[cls_name]} 条")

    # 打印测试集类别分布
    print(f"\n测试集类别分布:")
    test_counts = defaultdict(int)
    for row in test_rows:
        test_counts[row[label_col]] += 1
    for cls_name in sorted(test_counts.keys()):
        print(f"  {cls_name}: {test_counts[cls_name]} 条")


if __name__ == "__main__":
    main()


# python llm_pred_split2test.py --per_num_cls 50 --input_csv llm_parsed_results.csv --output_dir data --label_col label_name
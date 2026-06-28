from datasets import load_dataset, concatenate_datasets
import pandas as pd

ds = load_dataset("fancyzhx/ag_news")

label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ============================================================
# 1. 训练集：不均衡采样，模拟真实场景
# ============================================================
train_ds = ds["train"]

# 每个类别按不同数量采样（类别0最多，类别3最少 → 模拟不均衡分布）
label_0 = train_ds.filter(lambda x: x["label"] == 0).select(range(1000))
label_1 = train_ds.filter(lambda x: x["label"] == 1).select(range(800))
label_2 = train_ds.filter(lambda x: x["label"] == 2).select(range(700))
label_3 = train_ds.filter(lambda x: x["label"] == 3).select(range(300))

train_ds_subset = concatenate_datasets([label_0, label_1, label_2, label_3])

train_df = pd.DataFrame(train_ds_subset)
train_df = train_df.rename(columns={"label": "true_label"})
train_df["label_name"] = train_df["true_label"].map(label_names)

# 打乱顺序
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_output = "train.csv"
train_df.to_csv(train_output, index=False, encoding="utf-8-sig")
print(f"已导出训练集（不均衡，共 {len(train_df)} 条）到 {train_output}")
print("训练集类别分布:")
print(train_df["label_name"].value_counts().sort_index())
print()

# ============================================================
# 2. 测试集：均衡采样，用于公平评估
# ============================================================
test_ds = ds["test"]  # 7600 条，每类 1900 条

# 每个类别取 200 条（共 800 条），均衡
per_class = 200
test_label_0 = test_ds.filter(lambda x: x["label"] == 0).select(range(per_class))
test_label_1 = test_ds.filter(lambda x: x["label"] == 1).select(range(per_class))
test_label_2 = test_ds.filter(lambda x: x["label"] == 2).select(range(per_class))
test_label_3 = test_ds.filter(lambda x: x["label"] == 3).select(range(per_class))

test_ds_subset = concatenate_datasets([test_label_0, test_label_1, test_label_2, test_label_3])

test_df = pd.DataFrame(test_ds_subset)
test_df = test_df.rename(columns={"label": "true_label"})
test_df["label_name"] = test_df["true_label"].map(label_names)

# 打乱顺序
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_output = "test.csv"
test_df.to_csv(test_output, index=False, encoding="utf-8-sig")
print(f"已导出测试集（均衡，共 {len(test_df)} 条）到 {test_output}")
print("测试集类别分布:")
print(test_df["label_name"].value_counts().sort_index())
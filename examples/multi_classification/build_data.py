from datasets import load_dataset
import pandas as pd

ds = load_dataset("fancyzhx/ag_news")

# print(ds)
# 如何输出 ds label 对应的类别名？

# 导出训练集50条数据，每个类别均衡选取
train_ds = ds["train"]

# 按标签分组，每个类别选取12-13条（共50条）
label_0 = train_ds.filter(lambda x: x["label"] == 0).select(range(13))
label_1 = train_ds.filter(lambda x: x["label"] == 1).select(range(13))
label_2 = train_ds.filter(lambda x: x["label"] == 2).select(range(12))
label_3 = train_ds.filter(lambda x: x["label"] == 3).select(range(12))

# 合并所有数据
from datasets import concatenate_datasets
balanced_ds = concatenate_datasets([label_0, label_1, label_2, label_3])

# 转换为DataFrame
df = pd.DataFrame(balanced_ds)

# 添加类别名 (ag_news 数据集的标签: 0=World, 1=Sports, 2=Business, 3=Sci/Tech)

#  'label': ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech'])}
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
df["label_name"] = df["label"].map(label_names)

# 打乱顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存为CSV
output_path = "train_first_50.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"已导出训练集50条均衡数据到 {output_path}")
print("类别分布:")
print(df["label_name"].value_counts().sort_index())
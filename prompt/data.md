
## 数据库字段调整

在 test_async_api.py 构建提示词，也需要把 PromptBuilder.placeholders 对应的键值对，以json格式保存到数据库中。数据库增加一个 attr 字段存储该json数据。

在 export_csv.py 把数据库的数据导出来时，需要把 attr 转换为字典，其key作为csv的属性，从而把attr的数据也保存到csv中。


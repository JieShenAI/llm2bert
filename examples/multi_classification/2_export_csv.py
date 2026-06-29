"""
从数据库导出解析结果为 CSV

使用方式:
    python export_csv.py --db api_cache.db --output parsed_results.csv
"""

from pathlib import Path

from settings import DB_PATH, MULTICLASS_CONFIG, TASK_TYPE

import sys

sys.path.append("../../src")
from llm2bert.llm_api.parser import parse_and_export_from_db, LLMParser
from llm2bert.llm_api.api_with_cache import CacheDB


def main():

    # 检查数据库文件是否存在
    if not Path(DB_PATH).exists():
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return

    # 先读取数据库统计
    db = CacheDB(DB_PATH)
    stats = db.get_stats()
    print(f"数据库中共有 {stats['total_cached']} 条记录")

    # 使用 settings.py 中的配置
    print(f"\n开始解析...")
    try:
        output_path = parse_and_export_from_db(
            db_path=DB_PATH,
            output_filename="llm_parsed_results.csv",
            task_type=TASK_TYPE,
            multiclass_config=MULTICLASS_CONFIG,
            # limit=10,
            include_reason=True,
            include_errors=True,
        )

        print(f"\n成功! 结果已导出到: {output_path}")

        # 显示一些统计信息
        import pandas as pd

        df = pd.read_csv(output_path)
        print(f"\n导出数据统计:")
        print(f"  总记录数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        if "llm_pred_label" in df.columns:
            print(f"  标签分布:")
            label_counts = df["llm_pred_label"].value_counts().sort_index()
            for label, count in label_counts.items():
                print(f"    标签 {label}: {count} 条")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# python debug/LLM_API/export_csv.py --db api_cache.db --output parsed_results.csv

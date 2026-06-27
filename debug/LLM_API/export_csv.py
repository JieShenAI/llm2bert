"""
从数据库导出解析结果为 CSV

使用方式:
    python export_csv.py --db api_cache.db --output parsed_results.csv
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
from llm2bert.llm_api.parser import parse_and_export_from_db, LLMParser
from llm2bert.llm_api.api_with_cache import CacheDB


def main():
    parser = argparse.ArgumentParser(description="从数据库导出解析结果为 CSV")
    parser.add_argument("--db", type=str, default="api_cache.db", help="数据库文件路径")
    parser.add_argument("--output", type=str, default="parsed_results.csv", help="输出 CSV 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制读取的条数")
    parser.add_argument("--task-type", type=str, default=None, choices=["binary", "multiclass"], help="任务类型")
    parser.add_argument("--include-reason", action="store_true", default=True, help="包含 reason 列")
    parser.add_argument("--include-errors", action="store_true", help="包含解析失败的记录")

    args = parser.parse_args()

    # 检查数据库文件是否存在
    if not Path(args.db).exists():
        print(f"错误: 数据库文件不存在: {args.db}")
        return

    # 先读取数据库统计
    db = CacheDB(args.db)
    stats = db.get_stats()
    print(f"数据库中共有 {stats['total_cached']} 条记录")

    # 使用 settings.py 中的配置
    print(f"\n开始解析...")
    try:
        output_path = parse_and_export_from_db(
            db_path=args.db,
            output_path=args.output,
            task_type=args.task_type,
            limit=args.limit,
            include_reason=args.include_reason,
            include_errors=args.include_errors,
        )

        print(f"\n成功! 结果已导出到: {output_path}")

        # 显示一些统计信息
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"\n导出数据统计:")
        print(f"  总记录数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        if 'label' in df.columns:
            print(f"  标签分布:")
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                print(f"    标签 {label}: {count} 条")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# python debug/LLM_API/export_csv.py --db api_cache.db --output parsed_results.csv

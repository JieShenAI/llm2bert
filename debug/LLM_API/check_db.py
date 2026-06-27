"""检查数据库实际数据条数"""
import sqlite3

db_path = "api_cache.db"

# 直接查询数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取总条数
cursor.execute("SELECT COUNT(*) FROM cache")
total = cursor.fetchone()[0]
print(f"数据库实际条数: {total}")

# 获取所有数据
cursor.execute("SELECT id, prompt, model FROM cache ORDER BY id")
rows = cursor.fetchall()
print(f"\n实际查询到的条数: {len(rows)}")

print("\n所有数据的 ID:")
for row in rows:
    print(f"  ID {row[0]}: {row[2]} - {row[1][:50]}...")

conn.close()

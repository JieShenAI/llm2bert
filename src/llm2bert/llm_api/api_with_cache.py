"""
带 SQLite 缓存的 API 调用

功能：
1. 调用前先检查数据库，如果已有结果则直接返回
2. 新的调用结果自动保存到数据库
3. 支持失败重试，不会导致程序崩溃
4. 使用异步安全的 SQLite 操作
"""

import os
import asyncio
import sys
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from tqdm import tqdm

# 修复 Windows 编码
# if sys.platform == "win32":
#     import io
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


load_dotenv()

# ============================================
# SQLite 缓存管理
# ============================================


class CacheDB:
    """SQLite 缓存数据库"""

    def __init__(self, db_path: str = "api_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT UNIQUE NOT NULL,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_prompt_tokens INTEGER,
                    usage_completion_tokens INTEGER,
                    usage_total_tokens INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache (prompt_hash)
            """)
            conn.commit()

    @staticmethod
    def _hash_prompt(prompt: str, model: str) -> str:
        """生成 prompt 的哈希值"""
        combined = f"{model}:{prompt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """从缓存获取结果"""
        prompt_hash = self._hash_prompt(prompt, model)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM cache WHERE prompt_hash = ?", (prompt_hash,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "response": row["response"],
                    "usage_prompt_tokens": row["usage_prompt_tokens"],
                    "usage_completion_tokens": row["usage_completion_tokens"],
                    "usage_total_tokens": row["usage_total_tokens"],
                }
        return None

    def put(
        self,
        prompt: str,
        model: str,
        response: str,
        usage_prompt_tokens: int = 0,
        usage_completion_tokens: int = 0,
        usage_total_tokens: int = 0,
    ):
        """保存结果到缓存"""
        prompt_hash = self._hash_prompt(prompt, model)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO cache
                    (prompt_hash, prompt, model, response,
                     usage_prompt_tokens, usage_completion_tokens, usage_total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prompt_hash,
                        prompt,
                        model,
                        response,
                        usage_prompt_tokens,
                        usage_completion_tokens,
                        usage_total_tokens,
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            # 并发情况下可能的冲突，忽略
            pass

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            total = cursor.fetchone()[0]
            return {"total_cached": total}

    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        读取数据库中的所有数据

        Args:
            limit: 限制读取的条数，None 表示读取所有

        Returns:
            List[Dict]: 包含所有缓存数据的列表，每条数据格式为:
                {
                    'id': int,
                    'prompt_hash': str,
                    'prompt': str,
                    'model': str,
                    'response': str,
                    'created_at': str,
                    'usage_prompt_tokens': int,
                    'usage_completion_tokens': int,
                    'usage_total_tokens': int
                }
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if limit is not None:
                cursor = conn.execute(
                    "SELECT * FROM cache ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor = conn.execute("SELECT * FROM cache ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# ============================================
# 带缓存的 API 客户端
# ============================================


class CachedAPIClient:
    """带缓存的 API 客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        db_path: str,
        max_concurrent: int = 10,
        max_retries: int = 3,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache = CacheDB(db_path)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries

        # 统计
        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "failures": 0,
        }

    async def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，带缓存

        Returns:
            dict: {
                'response': str,
                'success': bool,
                'error': Optional[str],
            }
        """
        model = model or self.model

        # 1. 先查缓存
        cached = self.cache.get(prompt, model)
        if cached:
            self.stats["cache_hits"] += 1
            return {
                "response": cached["response"],
                "success": True,
                "error": None,
            }

        # 2. 缓存未命中，调用 API
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.choices[0].message.content

                    # 保存到缓存
                    usage = response.usage
                    self.cache.put(
                        prompt=prompt,
                        model=model,
                        response=content,
                        usage_prompt_tokens=usage.prompt_tokens if usage else 0,
                        usage_completion_tokens=usage.completion_tokens if usage else 0,
                        usage_total_tokens=usage.total_tokens if usage else 0,
                    )

                    self.stats["api_calls"] += 1
                    return {
                        "response": content,
                        "success": True,
                        "error": None,
                    }

                except (APIError, APIConnectionError, RateLimitError) as e:
                    if attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        await asyncio.sleep(wait_time)
                    else:
                        self.stats["failures"] += 1
                        return {
                            "response": None,
                            "success": False,
                            "error": str(e),
                        }
                except Exception as e:
                    self.stats["failures"] += 1
                    return {
                        "response": None,
                        "success": False,
                        "error": str(e),
                    }


# ============================================
# 主程序
# ============================================


async def main():
    """主程序"""
    # 初始化客户端
    if not os.getenv("api_key") or not os.getenv("base_url") or not os.getenv("model"):
        print("请在 .env 文件中设置 api_key, base_url, model")
        return

    client = CachedAPIClient(
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url"),
        model=os.getenv("model"),
        db_path=os.getenv("db_path"),
        max_concurrent=10,
        max_retries=3,
    )

    # 准备测试数据
    prompts = [f"Question {i}: {i} x 2 = ?" for i in range(50)]

    # 添加一些重复的 prompt 来测试缓存
    prompts.extend([f"Question {i}: {i} x 2 = ?" for i in range(5)])

    print(f"准备处理 {len(prompts)} 个请求...")
    print(f"当前缓存: {client.cache.get_stats()['total_cached']} 条")
    print("-" * 60)

    # 创建进度条
    pbar = tqdm(total=len(prompts), desc="Processing")

    # 并发处理
    async def process_prompt(prompt: str):
        result = await client.chat(prompt)
        pbar.update(1)
        return result

    tasks = [process_prompt(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    pbar.close()

    # 输出结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    success_count = sum(1 for r in results if r["success"])

    print(f"总请求数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")
    print(f"缓存命中: {client.stats['cache_hits']}")
    print(f"API 调用: {client.stats['api_calls']}")
    print(f"当前缓存总数: {client.cache.get_stats()['total_cached']}")

    print("\n详细结果:")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        if result["success"]:
            # 只显示前 50 个字符
            resp_preview = str(result["response"])[:50].replace("\n", " ")
            print(f"{i:2d} {prompt:20} -> {resp_preview}...")
        else:
            print(f"{i:2d} [FAIL] {prompt:20} -> ERROR: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())

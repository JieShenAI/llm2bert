"""
缓存数据库 Web 可视化服务器

提供 REST API 和 Web 界面来查看和管理 API 缓存
"""
import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterator

from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS


@dataclass
class CacheItem:
    """单个缓存项"""
    prompt: str
    response: str
    model: str
    id: Optional[int] = None
    prompt_hash: Optional[str] = None
    created_at: Optional[str] = None


class CacheDatabase:
    """缓存数据库"""

    def __init__(self, db_path: str = "api_cache.db"):
        self.db_path = db_path

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_all_items(self, limit: Optional[int] = None, offset: int = 0) -> List[CacheItem]:
        """获取所有缓存项"""
        with self._get_conn() as conn:
            query = "SELECT * FROM cache ORDER BY id DESC"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            rows = conn.execute(query).fetchall()
            items = []
            for row in rows:
                item = CacheItem(
                    id=row["id"],
                    prompt_hash=row["prompt_hash"],
                    prompt=row["prompt"],
                    model=row["model"],
                    response=row["response"],
                    created_at=row["created_at"],
                )
                items.append(item)
            return items

    def get_count(self) -> int:
        """获取总数量"""
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def search_items(self, keyword: str) -> List[CacheItem]:
        """搜索缓存项"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cache WHERE prompt LIKE ? OR response LIKE ? ORDER BY id DESC",
                (f"%{keyword}%", f"%{keyword}%")
            ).fetchall()
            items = []
            for row in rows:
                item = CacheItem(
                    id=row["id"],
                    prompt_hash=row["prompt_hash"],
                    prompt=row["prompt"],
                    model=row["model"],
                    response=row["response"],
                    created_at=row["created_at"],
                )
                items.append(item)
            return items

    def delete_item(self, item_id: int) -> None:
        """删除缓存项"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM cache WHERE id = ?", (item_id,))

    def delete_all(self) -> None:
        """清空所有缓存"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM cache")

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            # 按模型统计
            model_counts = {}
            rows = conn.execute("SELECT model, COUNT(*) as cnt FROM cache GROUP BY model").fetchall()
            for row in rows:
                model_counts[row["model"]] = row["cnt"]
        return {
            "total": total,
            "models": model_counts,
        }


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API 缓存可视化</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white; border-radius: 16px; padding: 30px;
            margin-bottom: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .header h1 { color: #1a202c; font-size: 28px; margin-bottom: 10px; }
        .header p { color: #718096; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }
        .stat-card {
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
        .stat-card .label { color: #718096; font-size: 14px; margin-bottom: 8px; }
        .stat-card .value { font-size: 32px; font-weight: 700; color: #4a5568; }
        .stat-card.model .value { font-size: 24px; }
        .stat-card.model .label { word-break: break-all; }
        .content { display: grid; grid-template-columns: 1fr; gap: 20px; }
        .card {
            background: white; border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); overflow: hidden;
        }
        .card-header {
            padding: 20px 24px; border-bottom: 1px solid #e2e8f0;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 15px;
        }
        .card-title { font-size: 18px; font-weight: 600; color: #2d3748; }
        .card-actions { display: flex; gap: 10px; flex-wrap: wrap; }
        .search-box { position: relative; }
        .search-box input {
            padding: 10px 16px; padding-left: 40px; border: 2px solid #e2e8f0;
            border-radius: 8px; font-size: 14px; width: 300px; transition: border-color 0.2s;
        }
        .search-box input:focus { outline: none; border-color: #667eea; }
        .search-icon {
            position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
            color: #a0aec0;
        }
        .btn {
            padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;
            font-size: 14px; font-weight: 500; transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
        .btn-danger { background: linear-gradient(135deg, #f56565, #e53e3e); color: white; }
        .btn-small { padding: 8px 16px; font-size: 13px; }
        .table-container { max-height: 700px; overflow: auto; }
        table { width: 100%; border-collapse: collapse; }
        thead { position: sticky; top: 0; background: #f7fafc; z-index: 10; }
        th, td { padding: 14px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { font-weight: 600; color: #4a5568; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
        td { color: #2d3748; font-size: 14px; }
        tbody tr { transition: background 0.2s; }
        tbody tr:hover { background: #f7fafc; }
        .text-cell { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
        .text-cell.expanded { white-space: pre-wrap; word-break: break-word; }
        .model-badge {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 12px; font-weight: 600; background: #e2e8f0; color: #4a5568;
        }
        .timestamp { font-size: 12px; color: #718096; }
        .db-info {
            background: #f7fafc; border-radius: 8px; padding: 15px 20px;
            margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;
        }
        .db-path { color: #4a5568; font-family: monospace; }
        .last-refresh { color: #718096; font-size: 14px; }
        .empty-state { padding: 60px 20px; text-align: center; color: #a0aec0; }
        .empty-state-icon { font-size: 48px; margin-bottom: 16px; }
        .empty-state-text { font-size: 16px; }
        .pagination {
            display: flex; justify-content: center; align-items: center; gap: 10px;
            padding: 20px; flex-wrap: wrap;
        }
        .page-btn {
            padding: 8px 16px; border: 2px solid #e2e8f0; background: white;
            border-radius: 8px; cursor: pointer; font-size: 14px; color: #4a5568;
            transition: all 0.2s;
        }
        .page-btn:hover:not(:disabled) { border-color: #667eea; color: #667eea; }
        .page-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .page-btn.active { background: #667eea; border-color: #667eea; color: white; }
        .refresh-btn {
            position: fixed; bottom: 30px; right: 30px; width: 60px; height: 60px;
            border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; border: none; cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            font-size: 24px; transition: transform 0.2s, box-shadow 0.2s; z-index: 100;
        }
        .refresh-btn:hover { transform: scale(1.1); box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5); }
        .refresh-btn.spinning { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .modal {
            display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;
            padding: 20px;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: white; border-radius: 16px; max-width: 900px; width: 100%;
            max-height: 90vh; overflow: auto; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .modal-header {
            padding: 24px; border-bottom: 1px solid #e2e8f0;
            display: flex; justify-content: space-between; align-items: center;
        }
        .modal-title { font-size: 20px; font-weight: 600; color: #2d3748; }
        .modal-close {
            background: none; border: none; font-size: 28px; cursor: pointer;
            color: #a0aec0; line-height: 1;
        }
        .modal-close:hover { color: #4a5568; }
        .modal-body { padding: 24px; }
        .detail-row { margin-bottom: 20px; }
        .detail-label {
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
            color: #718096; margin-bottom: 8px; font-weight: 600;
        }
        .detail-value {
            background: #f7fafc; padding: 12px 16px; border-radius: 8px;
            color: #2d3748; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
        }
        .copy-btn { margin-top: 10px; }
        @media (max-width: 768px) {
            .card-header { flex-direction: column; align-items: stretch; }
            .search-box input { width: 100%; }
            table { font-size: 12px; }
            th, td { padding: 10px 12px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💾 API 缓存可视化</h1>
            <p>查看和管理已缓存的 API 调用记录</p>
        </div>
        <div class="db-info">
            <div>
                <span style="color: #718096; font-size: 14px;">数据库:</span>
                <span class="db-path" id="dbPath"></span>
            </div>
            <div class="last-refresh" id="lastRefresh"></div>
        </div>
        <!-- 统计卡片 -->
        <div class="stats-grid" id="statsGrid"></div>
        <div class="content">
            <!-- 缓存列表 -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">缓存列表</span>
                    <div class="card-actions">
                        <div class="search-box">
                            <span class="search-icon">🔍</span>
                            <input type="text" id="searchInput" placeholder="搜索 prompt 或 response...">
                        </div>
                        <button class="btn btn-danger btn-small" id="clearAllBtn">清空缓存</button>
                    </div>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>模型</th>
                                <th>Prompt</th>
                                <th>Response</th>
                                <th>创建时间</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody id="itemTableBody"></tbody>
                    </table>
                </div>
                <div class="pagination" id="pagination"></div>
            </div>
        </div>
    </div>
    <!-- 刷新按钮 -->
    <button class="refresh-btn" id="refreshBtn" title="刷新数据">↻</button>
    <!-- 详情模态框 -->
    <div class="modal" id="detailModal">
        <div class="modal-content">
            <div class="modal-header">
                <span class="modal-title">缓存详情</span>
                <button class="modal-close" id="closeModal">&times;</button>
            </div>
            <div class="modal-body" id="modalBody"></div>
        </div>
    </div>
    <script>
        let currentPage = 1;
        const pageSize = 20;
        let allItems = [];
        let stats = {};
        let searchMode = false;

        // 格式化时间
        function formatTime(isoString) {
            if (!isoString) return '-';
            const date = new Date(isoString);
            return date.toLocaleString('zh-CN');
        }

        // 转义 HTML
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // 获取数据
        async function fetchStats() {
            const response = await fetch('/api/stats');
            stats = await response.json();
            return stats;
        }

        async function fetchItems() {
            const response = await fetch('/api/items');
            allItems = await response.json();
            return allItems;
        }

        // 渲染统计卡片
        function renderStats() {
            const grid = document.getElementById('statsGrid');
            let html = `
                <div class="stat-card">
                    <div class="label">总缓存数</div>
                    <div class="value">${stats.total || 0}</div>
                </div>
            `;
            if (stats.models) {
                for (const [model, count] of Object.entries(stats.models)) {
                    html += `
                        <div class="stat-card model">
                            <div class="label">${escapeHtml(model)}</div>
                            <div class="value">${count}</div>
                        </div>
                    `;
                }
            }
            grid.innerHTML = html;
        }

        // 获取过滤后的项
        function getFilteredItems() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            if (!searchTerm) return allItems;
            return allItems.filter(item =>
                (item.prompt && item.prompt.toLowerCase().includes(searchTerm)) ||
                (item.response && item.response.toLowerCase().includes(searchTerm))
            );
        }

        // 渲染表格
        function renderItems() {
            const filtered = getFilteredItems();
            const totalPages = Math.ceil(filtered.length / pageSize);
            const start = (currentPage - 1) * pageSize;
            const pageItems = filtered.slice(start, start + pageSize);

            const tbody = document.getElementById('itemTableBody');

            if (pageItems.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6">
                            <div class="empty-state">
                                <div class="empty-state-icon">📭</div>
                                <div class="empty-state-text">暂无缓存数据</div>
                            </div>
                        </td>
                    </tr>
                `;
            } else {
                tbody.innerHTML = pageItems.map(item => `
                    <tr>
                        <td><strong>#${item.id}</strong></td>
                        <td><span class="model-badge">${escapeHtml(item.model)}</span></td>
                        <td>
                            <div class="text-cell" id="prompt-${item.id}"
                                onclick="toggleExpand('prompt-${item.id}')"
                                title="点击展开/收起">${escapeHtml(item.prompt || '-')}</div>
                        </td>
                        <td>
                            <div class="text-cell" id="response-${item.id}"
                                onclick="toggleExpand('response-${item.id}')"
                                title="点击展开/收起">${escapeHtml(item.response || '-')}</div>
                        </td>
                        <td class="timestamp">${formatTime(item.created_at)}</td>
                        <td>
                            <button class="btn btn-primary btn-small" onclick="showDetail(${item.id})">详情</button>
                            <button class="btn btn-danger btn-small" onclick="deleteItem(${item.id})">删除</button>
                        </td>
                    </tr>
                `).join('');
            }

            renderPagination(totalPages, filtered.length);
        }

        // 渲染分页
        function renderPagination(totalPages, totalItems) {
            const pagination = document.getElementById('pagination');
            if (totalPages <= 1) {
                pagination.innerHTML = `<span style="color:#718096;">共 ${totalItems} 条</span>`;
                return;
            }
            let html = `
                <button class="page-btn" onclick="goToPage(1)" ${currentPage === 1 ? 'disabled' : ''}>首页</button>
                <button class="page-btn" onclick="goToPage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>上一页</button>
            `;
            const startPage = Math.max(1, currentPage - 2);
            const endPage = Math.min(totalPages, currentPage + 2);
            for (let i = startPage; i <= endPage; i++) {
                html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
            }
            html += `
                <button class="page-btn" onclick="goToPage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>下一页</button>
                <button class="page-btn" onclick="goToPage(${totalPages})" ${currentPage === totalPages ? 'disabled' : ''}>末页</button>
                <span style="color:#718096; margin-left:10px;">第 ${currentPage}/${totalPages} 页，共 ${totalItems} 条</span>
            `;
            pagination.innerHTML = html;
        }

        function toggleExpand(id) {
            const el = document.getElementById(id);
            el.classList.toggle('expanded');
        }

        function goToPage(page) {
            currentPage = page;
            renderItems();
        }

        // 显示详情
        function showDetail(itemId) {
            const item = allItems.find(i => i.id === itemId);
            if (!item) return;
            const modalBody = document.getElementById('modalBody');
            modalBody.innerHTML = `
                <div class="detail-row">
                    <div class="detail-label">ID</div>
                    <div class="detail-value">#${item.id}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">模型</div>
                    <div class="detail-value">${escapeHtml(item.model)}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Prompt</div>
                    <div class="detail-value">${escapeHtml(item.prompt || '-')}</div>
                    <button class="btn btn-primary btn-small copy-btn" onclick="copyText('${item.prompt.replace(/'/g, "\\'")}')">复制 Prompt</button>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Response</div>
                    <div class="detail-value">${escapeHtml(item.response || '-')}</div>
                    <button class="btn btn-primary btn-small copy-btn" onclick="copyText('${(item.response || '').replace(/'/g, "\\'")}')">复制 Response</button>
                </div>
                <div class="detail-row">
                    <div class="detail-label">创建时间</div>
                    <div class="detail-value">${formatTime(item.created_at)}</div>
                </div>
            `;
            document.getElementById('detailModal').classList.add('show');
        }

        // 复制文本
        function copyText(text) {
            navigator.clipboard.writeText(text).then(() => {
                alert('已复制到剪贴板！');
            }).catch(() => {
                alert('复制失败');
            });
        }

        // 删除项
        async function deleteItem(itemId) {
            if (!confirm('确定要删除这个缓存项吗？')) return;
            try {
                const response = await fetch(`/api/items/${itemId}`, { method: 'DELETE' });
                if (response.ok) await refreshData();
                else alert('删除失败');
            } catch (e) { alert('删除失败: ' + e); }
        }

        // 清空所有
        async function clearAll() {
            if (!confirm('确定要清空所有缓存吗？此操作不可恢复！')) return;
            try {
                const response = await fetch('/api/items', { method: 'DELETE' });
                if (response.ok) await refreshData();
                else alert('清空失败');
            } catch (e) { alert('清空失败: ' + e); }
        }

        // 刷新数据
        async function refreshData() {
            const btn = document.getElementById('refreshBtn');
            btn.classList.add('spinning');
            try {
                await Promise.all([fetchStats(), fetchItems()]);
                renderStats();
                renderItems();
                document.getElementById('lastRefresh').textContent = '最后刷新: ' + new Date().toLocaleTimeString('zh-CN');
            } finally {
                btn.classList.remove('spinning');
            }
        }

        // 初始化
        async function init() {
            const dbResponse = await fetch('/api/db-path');
            const dbData = await dbResponse.json();
            document.getElementById('dbPath').textContent = dbData.path;

            await refreshData();

            document.getElementById('searchInput').addEventListener('input', () => {
                currentPage = 1;
                renderItems();
            });

            document.getElementById('refreshBtn').addEventListener('click', refreshData);
            document.getElementById('closeModal').addEventListener('click', () => {
                document.getElementById('detailModal').classList.remove('show');
            });
            document.getElementById('detailModal').addEventListener('click', (e) => {
                if (e.target.id === 'detailModal') {
                    document.getElementById('detailModal').classList.remove('show');
                }
            });
            document.getElementById('clearAllBtn').addEventListener('click', clearAll);

            setInterval(refreshData, 10000);
        }

        init();
    </script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)
db: Optional[CacheDatabase] = None


def init_database(db_path: str = "api_cache.db"):
    global db
    db = CacheDatabase(db_path)
    return db


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def get_stats():
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    return jsonify(db.get_stats())


@app.route("/api/items")
def get_items():
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    items = db.get_all_items()
    return jsonify([asdict(i) for i in items])


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id: int):
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    db.delete_item(item_id)
    return jsonify({"success": True})


@app.route("/api/items", methods=["DELETE"])
def delete_all_items():
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    db.delete_all()
    return jsonify({"success": True})


@app.route("/api/db-path")
def get_db_path():
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    return jsonify({"path": str(Path(db.db_path).absolute())})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="API 缓存 Web 可视化")
    parser.add_argument("--db", default="api_cache.db", help="数据库文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5001, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    init_database(args.db)

    print("\n" + "=" * 60)
    print("  API 缓存 Web 可视化服务器")
    print("=" * 60)
    print(f"  数据库: {Path(args.db).absolute()}")
    print(f"  访问地址: http://{args.host}:{args.port}")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()

# python cache_web_server.py --db api_cache.db --port 8080

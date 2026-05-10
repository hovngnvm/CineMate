import logging
import sqlite3
import pickle
from rank_bm25 import BM25Okapi
from pathlib import Path
import chromadb

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "cinemate.db"
CHROMA_PATH = BASE_DIR / "db" / "chroma_storage"
BM25_PATH = BASE_DIR / "db" / "bm25_index.pkl"

_chroma_collection: chromadb.Collection | None = None
_bm25_data: dict | None = None


def _get_bm25_data() -> dict:
    """Nạp chỉ mục BM25 từ file pickle một lần và cache lại cho các lần gọi sau."""
    global _bm25_data
    if _bm25_data is None:
        logger.info("Loading BM25 index (one-time)…")
        with open(BM25_PATH, 'rb') as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _get_chroma_collection() -> chromadb.Collection:
    """Mở collection ChromaDB một lần và giữ handle singleton."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_collection(name="movies")
    return _chroma_collection


def run_sql(query_string: str, params: tuple = ()) -> list | str:
    """
    Thực thi truy vấn đọc trên SQLite. Dùng placeholder ? để tham số hóa
    và tránh SQL injection. Trả về danh sách dòng hoặc chuỗi lỗi.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query_string, params)
            return cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("SQL query failed: %s | query: %s", exc, query_string)
        return f"Lỗi truy vấn: {exc}"


def search_vector(query_embedding: list[float], top_k: int = 3) -> list[tuple]:
    """Tìm kiếm ngữ nghĩa trên vector store, trả về top K phim gần nhất theo embedding."""
    collection = _get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return [
        {
            "id":           meta.get("id", ""),
            "title":        meta.get("title", ""),
            "year":         meta.get("year", ""),
            "genres":       meta.get("genres", ""),
            "overview":     meta.get("overview", ""),
            "vote_average": meta.get("vote_average", "N/A"),
            "vote_count":   meta.get("vote_count", "N/A"),
            "poster_url":   meta.get("poster_url", ""),
        }
        for meta in results["metadatas"][0]
    ]


def search_index(text_query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm từ khóa trên chỉ mục BM25, trả về top K phim có điểm khớp cao nhất."""
    data = _get_bm25_data()
    bm25: BM25Okapi = data['bm25']
    records = data['records']

    tokens = text_query.lower().split()
    scores = bm25.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Chỉ giữ kết quả có điểm dương, tức khớp ít nhất một từ khóa
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            rec = records[idx]
            results.append({
                "id": str(rec.get("id", "")),
                "title": rec.get("title", ""),
                "year": rec.get("year", ""),
                "genres": rec.get("genres", ""),
                "overview": rec.get("overview", ""),
                "vote_average": rec.get("vote_average", "N/A"),
                "vote_count": rec.get("vote_count", "N/A"),
                "poster_url": rec.get("poster_url", ""),
            })
    return results
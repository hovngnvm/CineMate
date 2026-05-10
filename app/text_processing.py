"""
text_processing.py

Các hàm thuần túy xử lý văn bản: trích xuất SQL, làm sạch phản hồi LLM,
chuẩn hóa kết quả cơ sở dữ liệu và tách từ khóa cho BM25.
Tất cả đều không có trạng thái nội bộ nên dễ dàng kiểm thử độc lập.
"""

import re

from config import EN_STOPWORDS

# Biên dịch sẵn toàn bộ regex lúc import để tránh phân tích lặp lại mỗi lần gọi hàm
_RE_SQL_EXTRACT      = re.compile(r"\[SQL_QUERY\](.*?)(\[RESPONSE\]|$)", re.DOTALL | re.IGNORECASE)
_RE_STRIP_SQL_TAG    = re.compile(r'\[SQL_QUERY\].*?(?=\[|$)', re.DOTALL | re.IGNORECASE)
_RE_STRIP_THOUGHT    = re.compile(r'\[THOUGHT\].*?(?=\[|$)', re.DOTALL | re.IGNORECASE)
_RE_STRIP_RESPONSE   = re.compile(r'\[RESPONSE\]', re.IGNORECASE)
_RE_STRIP_ANY_TAG    = re.compile(r'\[.*?\]')

_RE_SELECT_FROM      = re.compile(r'(?i)^SELECT\s+.*?\s+FROM')
_RE_LOWER_CAST       = re.compile(r'(?i)\bLOWER\s*\(\s*cast\s*\)')
_RE_LOWER_CREW       = re.compile(r'(?i)\bLOWER\s*\(\s*crew\s*\)')
_RE_BARE_CAST        = re.compile(r'(?i)\bcast\b(?!\s*`)')
_RE_BARE_CREW        = re.compile(r'(?i)\bcrew\b(?!\s*`)')
_RE_LIMIT            = re.compile(r'(?i)LIMIT\s+\d+')

_RE_ALPHA_TOKENS     = re.compile(r"[A-Za-z']+")

_EMPTY_POSTER_VALUES = {"0", "0.0", "None", "nan", ""}


def extract_sql_query(llm_output: str) -> str | None:
    """Trích xuất câu truy vấn SQL từ phản hồi LLM nằm giữa cặp tag SQL_QUERY và RESPONSE."""
    if not llm_output:
        return None
    match = _RE_SQL_EXTRACT.search(llm_output)
    if match:
        sql = match.group(1).replace("```sql", "").replace("```", "").strip()
        return sql if sql.upper() != "NONE" else None
    return None


def clean_response(text: str) -> str:
    """Loại bỏ toàn bộ tag nội bộ mà LLM có thể vô tình xuất ra trong câu trả lời."""
    if not text:
        return ""
    text = _RE_STRIP_SQL_TAG.sub('', text)
    text = _RE_STRIP_THOUGHT.sub('', text)
    text = _RE_STRIP_RESPONSE.sub('', text)
    return _RE_STRIP_ANY_TAG.sub('', text).strip()


def format_db_results_to_dicts(raw_results: list) -> list[dict]:
    """Chuẩn hóa kết quả thô từ cơ sở dữ liệu (tuple hoặc dict) thành danh sách dict đồng nhất."""
    formatted_movies: list[dict] = []
    for row in raw_results:
        if isinstance(row, dict):
            poster = str(row.get("poster_url", ""))
            if poster.strip() in _EMPTY_POSTER_VALUES:
                poster = ""
            # Sao chép nông để không làm thay đổi dict gốc của lời gọi bên ngoài
            movie = {**row, "poster_url": poster}
            formatted_movies.append(movie)
            continue

        title  = str(row[1]) if len(row) > 1 and row[1] else "Unknown"
        poster = str(row[9]) if len(row) > 9 and row[9] else ""
        if poster.strip() in _EMPTY_POSTER_VALUES:
            poster = ""

        # Ánh xạ cứng theo thứ tự cột của bảng Movies trong SQLite
        formatted_movies.append({
            "id":           row[0] if len(row) > 0 else None,
            "title":        title,
            "year":         row[2] if len(row) > 2 and row[2] else "N/A",
            "genres":       row[3] if len(row) > 3 and row[3] else "N/A",
            "overview":     row[4] if len(row) > 4 and row[4] else "Chưa có nội dung tóm tắt.",
            "vote_average": row[5] if len(row) > 5 and row[5] else "N/A",
            "vote_count":   row[6] if len(row) > 6 and row[6] else "N/A",
            "poster_url":   poster,
        })
    return formatted_movies


def sanitise_sql(sql: str, num_recs: int) -> str:
    """Chuẩn hóa SQL do LLM sinh ra: ép SELECT *, bọc backtick cho từ khóa đặc biệt, giới hạn LIMIT."""
    sql = _RE_SELECT_FROM.sub('SELECT * FROM', sql.strip())
    sql = _RE_LOWER_CAST.sub('LOWER(`cast`)', sql)
    sql = _RE_LOWER_CREW.sub('LOWER(`crew`)', sql)
    sql = _RE_BARE_CAST.sub('`cast`', sql)
    sql = _RE_BARE_CREW.sub('`crew`', sql)
    if "LIMIT" not in sql.upper():
        sql += f" LIMIT {num_recs}"
    else:
        sql = _RE_LIMIT.sub(f'LIMIT {num_recs}', sql)
    return sql


def extract_keywords_for_bm25(text: str) -> str:
    """Loại bỏ stop words và giữ lại các từ mang tín hiệu cao cho BM25."""
    tokens = _RE_ALPHA_TOKENS.findall(text)
    keywords = [t for t in tokens if t.lower() not in EN_STOPWORDS and len(t) > 2]
    result = " ".join(keywords)
    print(f"- BM25 KEYWORDS: '{result}'")
    return result if result else text

"""
rag.py

Pipeline Retrieval Augmented Generation. Bao gồm mở rộng truy vấn,
fallback SQL theo tên riêng, hợp nhất RRF và điều phối hybrid RAG
khi truy vấn SQL có cấu trúc không đủ hiệu quả.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import ollama

from config import OLLAMA_MODEL
from text_processing import (
    extract_keywords_for_bm25,
    format_db_results_to_dicts,
)
from prompts import build_query_expansion_prompt
from services import translate_vi_to_en, query_embedding, reranking

try:
    from retriever import run_sql, search_vector, search_index
except ImportError:
    print("SafeMode active.")
    run_sql = lambda x, params=(): []
    search_vector = lambda x, top_k=3: []
    search_index = lambda x, top_k=10: []

# Regex nhận diện tên riêng có dấu Unicode, biên dịch một lần vì pattern phức tạp
_RE_PROPER_NOUNS = re.compile(
    r'\b([A-ZÁÀẢÃẠĂẮẶẴẰẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]'
    r'[a-záàảãạăắặẵằấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+'
    r'(?:\s+[A-ZÁÀẢÃẠĂẮẶẴẰẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]'
    r'[a-záàảãạăắặẵằấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+)+)\b'
)


def expand_semantic_query(user_message: str, eng_query: str) -> str:
    """Nhờ LLM sinh thêm từ khóa chuyên ngành phim để làm giàu truy vấn tìm kiếm."""
    prompt = build_query_expansion_prompt(user_message, eng_query)
    try:
        res = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = res['message']['content'].strip()
        expanded = re.sub(r'[^\w\s]', ' ', raw).strip()
        combined = f"{eng_query} {expanded}"
        print(f"- QUERY EXPANSION: '{expanded}'")
        print(f"- COMBINED QUERY:  '{combined}'")
        return combined
    except Exception as e:
        print(f"- Query expansion failed: {e}")
        return eng_query


def rrf_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: gộp hai danh sách xếp hạng thành một danh sách điểm duy nhất."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    for rank, doc in enumerate(vector_results):
        doc_id = str(doc.get('id'))
        doc_map[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(bm25_results):
        doc_id = str(doc.get('id'))
        doc_map[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def try_named_entity_sql(user_message: str, num_recs: int) -> list:
    """Tìm kiếm trực tiếp trong cột cast, keywords, title theo các cụm tên riêng phát hiện được."""
    proper_nouns = _RE_PROPER_NOUNS.findall(user_message)
    if not proper_nouns:
        return []

    results = []
    for name in proper_nouns:
        name_lower = name.lower()
        # Dùng tham số hóa để chống SQL injection
        pattern = f"%{name_lower}%"
        sql = (
            "SELECT * FROM Movies WHERE "
            "LOWER(`cast`) LIKE ? OR "
            "LOWER(keywords) LIKE ? OR "
            "LOWER(title) LIKE ? "
            "ORDER BY vote_count DESC LIMIT ?"
        )
        params = (pattern, pattern, pattern, num_recs * 3)
        print(f"- NAMED-ENTITY SQL: {sql}  params={params}")
        rows = run_sql(sql, params)
        if rows and not isinstance(rows, str):
            results.extend(rows)

    # Loại bỏ kết quả trùng lặp dựa trên ID phim
    seen, deduped = set(), []
    for row in results:
        rid = row[0] if not isinstance(row, dict) else row.get('id')
        if rid not in seen:
            seen.add(rid)
            deduped.append(row)
    return deduped[:num_recs]


def run_hybrid_rag(user_message: str, num_recs: int) -> list:
    """
    Pipeline hybrid RAG đầy đủ: dịch sang tiếng Anh, mở rộng từ khóa,
    truy xuất song song vector và BM25, hợp nhất RRF, xếp hạng lại bằng Cross Encoder.
    """
    print("- Falling back to full Hybrid RAG pipeline")
    eng_query = translate_vi_to_en(user_message)
    print(f"- HYBRID PATH — translated: '{eng_query}'")

    expanded_query = expand_semantic_query(user_message, eng_query)

    embedded_query = query_embedding(expanded_query)
    bm25_query = extract_keywords_for_bm25(expanded_query)

    # Truy xuất song song vì vector search và BM25 là hai tác vụ I/O độc lập
    oversample = max(num_recs * 6, 20)
    with ThreadPoolExecutor(max_workers=2) as pool:
        vec_future  = pool.submit(search_vector, embedded_query, top_k=oversample)
        bm25_future = pool.submit(search_index, bm25_query, top_k=oversample)
        vector_candidates = vec_future.result()
        bm25_candidates   = bm25_future.result()

    vector_list = format_db_results_to_dicts(vector_candidates)
    bm25_list   = format_db_results_to_dicts(bm25_candidates)

    pool_titles = (
        [d.get('title', '?') for d in vector_list] +
        [d.get('title', '?') for d in bm25_list]
    )
    print(f"- POOL: {pool_titles}")

    combined_candidates = rrf_fusion(vector_list, bm25_list)
    print(f"- RRFUSION: {len(combined_candidates)} unique candidates.")
    print(f"- TOP 10 RRF: {[m['title'] for m in combined_candidates[:10]]}")

    if len(combined_candidates) > num_recs:
        # Xếp hạng lại dựa trên bản dịch gốc để đảm bảo độ chính xác ngữ nghĩa
        raw_db_results = reranking(eng_query, combined_candidates, num_recs)
        print(f"- RERANKED: {[m['title'] for m in raw_db_results]}")
    else:
        raw_db_results = combined_candidates

    return raw_db_results

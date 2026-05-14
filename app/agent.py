"""
agent.py

Bộ điều phối trung tâm của CineMate. Xử lý phân loại ý định người dùng
và ủy quyền cho các module chuyên biệt: config, text_processing, prompts, rag.
"""

import re
import ollama

from config import OLLAMA_MODEL, CHAT_KEYWORDS, VAGUE_PICK_PATTERNS
from text_processing import (
    extract_sql_query,
    clean_response,
    format_db_results_to_dicts,
    sanitise_sql,
)
from prompts import (
    build_router_prompt,
    build_chat_prompt,
    build_sql_prompt,
    build_response_prompt,
)
from rag import try_named_entity_sql, run_hybrid_rag

try:
    from retriever import run_sql
except ImportError:
    print("SafeMode active.")
    run_sql = lambda x: []


def _keyword_is_chat(msg: str) -> bool | None:
    """Kiểm tra nhanh bằng từ khóa: trả về True nếu là xã giao, None nếu không rõ."""
    lower = msg.lower()
    if any(kw in lower for kw in CHAT_KEYWORDS):
        return True
    if any(p.search(lower) for p in VAGUE_PICK_PATTERNS):
        return True
    return None


def get_agent_response(
    user_message: str,
    num_recs: int,
    chat_history: list[dict],
) -> tuple[str, list[dict] | None]:
    """
    Xử lý tin nhắn và trả về (văn bản phản hồi, danh sách phim hoặc None).

    Luồng quyết định: từ khóa nhanh, LLM phân loại, rẽ nhánh CHAT hoặc SEARCH
    (SQL, named entity fallback, hybrid RAG).
    """
    try:
        clean_history = [
            {'role': m['role'], 'content': m['content']}
            for m in chat_history[-4:]
        ]
        print(f"\n📋 CLEAN HISTORY: {clean_history}")

        # Phân loại ý định: ưu tiên từ khóa, chỉ gọi LLM khi không chắc chắn
        fast_result = _keyword_is_chat(user_message)
        if fast_result is True:
            is_search = False
            intent_raw = "FAST-KEYWORD → CHAT"
        else:
            router_prompt = build_router_prompt(clean_history, user_message)
            intent_res = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': router_prompt}]
            )
            intent_raw = intent_res['message']['content'].strip().upper()

            if "SEARCH" in intent_raw:
                is_search = True
            elif "CHAT" in intent_raw:
                is_search = False
            else:
                is_search = not any(kw in user_message.lower() for kw in CHAT_KEYWORDS)

        intent = "SEARCH" if is_search else "CHAT"
        print(f"- INTENT: {intent} (raw: {intent_raw})")

        # Nhánh xã giao: phản hồi hội thoại, không trả danh sách phim
        if not is_search:
            chat_prompt = build_chat_prompt()
            chat_res = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'system', 'content': chat_prompt}] + clean_history +
                          [{'role': 'user', 'content': user_message}]
            )
            return clean_response(chat_res['message']['content']), None

        # Nhánh tìm kiếm: thử SQL từ LLM trước
        sql_prompt = build_sql_prompt(user_message, num_recs)
        response_1 = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': sql_prompt}]
        )
        sql_query = extract_sql_query(response_1['message']['content'])
        print(f"- RAW SQL QUERY: {sql_query}")

        raw_db_results = []
        used_sql = False

        if sql_query and sql_query.strip().upper() not in ("NONE", "") and \
                sql_query.strip().upper().startswith("SELECT"):
            sql_query = sanitise_sql(sql_query, num_recs)
            print(f"- SQL PATH: {sql_query}")
            raw_db_results = run_sql(sql_query)
            if raw_db_results and not isinstance(raw_db_results, str):
                used_sql = True
                print(f"- SQL returned {len(raw_db_results)} results.")

        # Fallback named entity: tìm theo tên riêng trong câu hỏi
        if not used_sql:
            ne_results = try_named_entity_sql(user_message, num_recs)
            if ne_results:
                raw_db_results = ne_results
                used_sql = True
                print(f"- NAMED-ENTITY FALLBACK: {len(ne_results)} results.")

        # Fallback cuối: hybrid RAG kết hợp ngữ nghĩa và BM25
        if not used_sql:
            raw_db_results = run_hybrid_rag(user_message, num_recs)

        movies_list = format_db_results_to_dicts(raw_db_results)
        movie_titles = ", ".join([m['title'] for m in movies_list if m['title'] != 'Unknown'])
        if not movie_titles:
            return "Mình chưa tìm được phim phù hợp. Bạn thử mô tả thêm nhé?", []

        response_prompt = build_response_prompt(user_message, movie_titles)
        response_2 = ollama.chat(
            model=OLLAMA_MODEL,
            messages=clean_history + [{'role': 'user', 'content': response_prompt}]
        )
        return clean_response(response_2['message']['content']), movies_list

    except Exception as e:
        return f"System Error: {str(e)}", []
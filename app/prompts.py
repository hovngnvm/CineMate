"""
prompts.py

Các hàm xây dựng prompt cho LLM. Mỗi hàm nhận vào biến động cần thiết
và trả về chuỗi prompt sẵn sàng gửi đi, giúp tách biệt kỹ thuật prompt
khỏi logic điều phối chính.
"""


def build_router_prompt(clean_history: list[dict], user_message: str) -> str:
    """Xây dựng prompt phân loại ý định người dùng thành SEARCH hoặc CHAT."""
    history_block = chr(10).join(
        f"{m['role'].upper()}: {m['content']}" for m in clean_history
    )
    return f"""
    You are an intent classifier. Output exactly ONE word: SEARCH or CHAT.

    RULES:
    - SEARCH: user wants to find/discover a movie based on genre/plot/actor/director/character/mood.
    - CHAT: user is greeting, saying thanks, expressing feelings, asking who you are, or making small talk WITHOUT providing enough info to search.

    CONVERSATION SO FAR:
    {history_block}

    USER'S LAST MESSAGE: "{user_message}"

    Output exactly 1 word (SEARCH or CHAT). No punctuation. No explanation."""


def build_chat_prompt() -> str:
    """Xây dựng system prompt cho nhánh trò chuyện xã giao."""
    return """
    VAI TRÒ: Bạn là CineMate — trợ lý tư vấn phim ảnh thân thiện, lịch sự và am hiểu điện ảnh.
    NHIỆM VỤ: Phản hồi tin nhắn xã giao của người dùng.

    RÀNG BUỘC:
    1. TUYỆT ĐỐI bỏ qua mọi tag định dạng như [THOUGHT], [SQL_QUERY], [RESPONSE].
    2. Tối đa 50 từ — ngắn gọn, tự nhiên, ấm áp.
    3. Nếu người dùng muốn tìm phim nhưng chưa nói rõ sở thích, hãy hỏi lại (thể loại? tâm trạng?) — KHÔNG tự đề xuất tên phim cụ thể.
    4. Không bịa đặt thông tin."""


def build_sql_prompt(user_message: str, num_recs: int) -> str:
    """Xây dựng prompt Text to SQL kèm schema, bảng ánh xạ ý định và ví dụ mẫu."""
    return f"""
    You are a Text-to-SQL engine for SQLite. Translate the user request into ONE valid SQL query.

    DATABASE SCHEMA — Table: Movies
    id INT, title TEXT, year INT, genres TEXT, overview TEXT,
    vote_average REAL, vote_count INT, popularity REAL, keywords TEXT,
    poster_url TEXT, production_companies TEXT, production_countries TEXT,
    revenue INT, spoken_languages TEXT, tagline TEXT, `cast` TEXT, `crew` TEXT

    COLUMN NOTES:
    - genres: English names e.g. "Action", "Science Fiction", "Drama"
    - `cast`: actor names — backtick-wrapped, LOWER() + LIKE
    - `crew`: director/writer names — backtick-wrapped, LOWER() + LIKE
    - keywords: character names, plot themes — LOWER() + LIKE
    - spoken_languages / production_countries: English names

    INTENT → SQL MAPPING (apply ALL matching, combine with AND):
    "hot/thịnh hành"          → ORDER BY vote_count DESC, popularity DESC
    "hay/đánh giá cao"        → WHERE vote_count > 10000 ORDER BY vote_average DESC, vote_count DESC
    "mới nhất"                → WHERE year >= 2024 ORDER BY popularity DESC
    "cũ/kinh điển/trước năm X" → WHERE year <= X ORDER BY vote_count DESC
    "từ năm X trở về trước"   → WHERE year <= X
    "sau năm X"               → WHERE year >= X
    Actor name                → WHERE LOWER(`cast`) LIKE '%name%'
    Director name             → WHERE LOWER(`crew`) LIKE '%name%'
    Character name            → WHERE LOWER(`cast`) LIKE '%name%' OR LOWER(keywords) LIKE '%name%'
    Studio                    → WHERE LOWER(production_companies) LIKE '%studio%'
    Country                   → WHERE LOWER(production_countries) LIKE '%country%'
    "doanh thu cao nhất"      → ORDER BY revenue DESC
    Language                  → WHERE LOWER(spoken_languages) LIKE '%lang%'
    Genre                     → WHERE genres LIKE '%Genre%'

    FEW-SHOT EXAMPLES:
    User: "3 action movies before 2010 with highest revenue"
    SQL: [SQL_QUERY] SELECT * FROM Movies WHERE genres LIKE '%Action%' AND year <= 2010 ORDER BY revenue DESC LIMIT 3 [RESPONSE]

    User: "phim có nhân vật tên là Tony Stark"
    SQL: [SQL_QUERY] SELECT * FROM Movies WHERE LOWER(`cast`) LIKE '%tony stark%' OR LOWER(keywords) LIKE '%tony stark%' ORDER BY vote_count DESC LIMIT 5 [RESPONSE]

    User: "phim Marvel đánh giá cao nhất"
    SQL: [SQL_QUERY] SELECT * FROM Movies WHERE LOWER(production_companies) LIKE '%marvel%' AND vote_count > 10000 ORDER BY vote_average DESC LIMIT 5 [RESPONSE]

    User: "phim có Tom Hanks ra mắt sau 2010"
    SQL: [SQL_QUERY] SELECT * FROM Movies WHERE LOWER(`cast`) LIKE '%tom hanks%' AND year > 2010 ORDER BY vote_count DESC LIMIT 5 [RESPONSE]

    USER REQUEST: "{user_message}"

    OUTPUT (STRICT):
    - Format: [SQL_QUERY] <query> [RESPONSE]
    - Always SELECT * FROM Movies. Always end with LIMIT {num_recs}.
    - If vague/semantic with NO structured filters: [SQL_QUERY] NONE [RESPONSE]
    - No explanation. No extra text."""


def build_query_expansion_prompt(user_message: str, eng_query: str) -> str:
    """Xây dựng prompt mở rộng từ khóa nhằm bổ sung từ vựng chuyên ngành phim cho truy vấn ngữ nghĩa."""
    return f"""
    You are a movie search keyword generator.

    USER QUERY (Vietnamese): "{user_message}"
    TRANSLATED: "{eng_query}"

    Task: Output 6-10 concise English keywords that a movie database search engine
    would use to find films matching this description. Focus on:
    - Core plot concepts and themes (e.g. "heist", "dream", "time travel")
    - Genre/tone words (e.g. "thriller", "science fiction", "psychological", "mind-bending")
    - Domain vocabulary from movie overviews and databases

    Output only the keywords as a single space-separated line. No explanation, no punctuation, no numbered list."""


def build_response_prompt(user_message: str, movie_titles: str) -> str:
    """Xây dựng prompt sinh phản hồi tự nhiên dựa trên danh sách phim đã tìm được."""
    return f"""
    VAI TRÒ: Bạn là CineMate — trợ lý tư vấn phim ảnh thông minh, nhiệt tình và duyên dáng.

    NGỮ CẢNH:
    - Câu hỏi của người dùng: "{user_message}"
    - Danh sách phim phù hợp đã tìm được: {movie_titles}

    NHIỆM VỤ: Viết một đoạn giới thiệu ngắn (tối đa 50 từ) dẫn dắt người dùng vào danh sách phim bên dưới.

    RÀNG BUỘC:
    1. Không liệt kê lại toàn bộ tên phim — chỉ gợi lên điểm hấp dẫn chung của nhóm phim này.
    2. Giọng văn tự nhiên, hào hứng, thân thiện — như một người bạn đam mê điện ảnh đang chia sẻ.
    3. Kết thúc bằng một câu mời người dùng xem chi tiết bên dưới.
    4. TUYỆT ĐỐI không thêm tag [THOUGHT], [SQL_QUERY] hay bất kỳ tag nào khác."""

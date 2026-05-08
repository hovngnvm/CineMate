import re
import ollama
from transformers import MarianMTModel, MarianTokenizer

try:
    from retriever import run_sql, search_vector
except ImportError:
    print("SafeMode active.")
    run_sql = lambda x: []
    search_vector = lambda x, top_k=3: []

# TEXT PROCESSING
def extract_sql_query(llm_output):
    """Extract SQL query from LLM output"""
    pattern = r"\[SQL_QUERY\](.*?)(\[RESPONSE\]|$)"
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        sql = match.group(1).replace("```sql", "").replace("```", "").strip()
        return sql if sql.upper() != "NONE" else None
    
    return None

def clean_ai_text(text):
    if not text:
        return ""
    
    # Get text after [RESPONSE]
    if "[RESPONSE]" in text.upper():
        idx = text.upper().index("[RESPONSE]")
        return text[idx + len("[RESPONSE]"):].strip()
    
    # Remove [THOUGHT] tags
    cleaned = re.sub(r'\[THOUGHT\].*?(?=\[SQL_QUERY\]|\[RESPONSE\]|$)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    if not cleaned:
        # If cleaned text is empty, remove all tags
        return re.sub(r'\[.*?\]', '', text).strip()
    
    return cleaned

def format_db_results_to_dicts(raw_results):
    formatted_movies = []
    for row in raw_results:
        if isinstance(row, dict):
            poster = str(row.get("poster_url", ""))
            if poster.strip() in ["0", "0.0", "None", "nan", ""]: poster = "" 
            row["poster_url"] = poster
            formatted_movies.append(row)
            continue
        title = str(row[1]) if len(row) > 1 and row[1] else "Unknown"
        poster = str(row[9]) if len(row) > 9 and row[9] else ""
        if poster.strip() in ["0", "0.0", "None", "nan", ""]: poster = ""
        
        movie_dict = {
            "id": row[0] if len(row) > 0 else None,
            "title": title,
            "year": row[2] if len(row) > 2 and row[2] else "N/A",
            "genres": row[3] if len(row) > 3 and row[3] else "N/A",
            "overview": row[4] if len(row) > 4 and row[4] else "Chưa có nội dung tóm tắt.",
            "vote_average": row[5] if len(row) > 5 and row[5] else "N/A",
            "vote_count": row[6] if len(row) > 6 and row[6] else "N/A",
            "poster_url": poster
        }
        formatted_movies.append(movie_dict)
        
    return formatted_movies

# RAG Core System
_translator_model = None
_translator_tokenizer = None

def _get_translator():
    """
    Load translation model to RAM only once (Singleton)
    """
    global _translator_model, _translator_tokenizer
    if _translator_model is None:
        print("⏳ Loading Translation Model Helsinki-NLP (First time only)...")
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        # Load Tokenizer and Model
        _translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator_model = MarianMTModel.from_pretrained(model_name)
        print("✅ Loading Translation Model successfully!")
    return _translator_tokenizer, _translator_model

def get_english_query(vietnamese_query):
    """
    Translate Vietnamese -> English
    """
    try:
        tokenizer, model = _get_translator()
        
        # Tokenize
        inputs = tokenizer(vietnamese_query, return_tensors="pt", padding=True)
        # Translate
        translated_tokens = model.generate(**inputs)
        # Decode
        english_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return english_text.strip()
    except Exception as e:
        print(f"⚠️ Lỗi dịch thuật: {e}")
        return vietnamese_query # Fallback
    
def get_agent_response(user_message, num_recs, chat_history):
    try:
        # 1. Clean history (Only remember the last 4 messages)
        clean_history = [{'role': m['role'], 'content': m['content']} for m in chat_history[-4:]]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in clean_history])
        print(f"\n📋 CLEAN HISTORY: {clean_history}")
        print(f"📋 HISTORY TEXT: {history_text}")
        
        # 2. ROUTER (3-layer: LLM raw → LLM cleaned → keyword fallback)
        router_prompt = f"""
            Chat history:
            {history_text}
            
            TASK: CLASSIFY the user's intent based on their last message: '{user_message}'
            RULES:
            1. Return exactly the word "SEARCH" if the user is asking to find a movie, looking for movie recommendations, describing a movie, or talking about movie genres.
            2. Return exactly the word "CHAT" if the user is just greeting, thanking, asking who you are, making small talk, or not mentioning movies.
            WARNING: YOU ARE A ROUTER. DO NOT GREET. DO NOT EXPLAIN. RETURN EXACTLY 1 WORD ONLY.
            """

        intent_res = ollama.chat(model='cinemate_agent', messages=[{'role': 'user', 'content': router_prompt}])
        intent_raw = intent_res['message']['content'].upper()
        
        # Check RAW output
        if "SEARCH" in intent_raw:
            is_search = True
        elif "CHAT" in intent_raw or "CHÀO HỎI" in intent_raw or "KHÔNG LIÊN QUAN" in intent_raw:
            is_search = False
        else:
            # LLM didn't follow instructions — fallback to keyword detection
            _chat_keywords = ["chào", "hello", "hi ", "hey", "cảm ơn", "thanks", "thank",
                              "bạn là ai", "who are you", "tạm biệt", "bye", "ok", "oke"]
            msg_lower = user_message.lower().strip()
            is_search = not any(kw in msg_lower for kw in _chat_keywords)
        
        intent = "SEARCH" if is_search else "CHAT"
        print(f"- INTENT: {intent} (raw: {intent_raw[:]})")

        if not is_search:
            prompt_1 = f"""
            Bạn là CineMate, một trợ lý ảo, chuyên gia tư vấn phim ảnh.
            NHIỆM VỤ hiện tại: Trò chuyện giao tiếp cơ bản với người dùng.

            QUY TẮC NGHIÊM NGẶT:
            1. Trả lời bám sát câu hỏi, yêu cầu của người dùng bằng lời văn thân thiện, tự nhiên, nhẹ nhàng (tối đa 40 từ).
            2. TUYỆT ĐỐI KHÔNG tự bịa ra bất kỳ tên bộ phim nào. Nếu người dùng muốn xem phim, hãy nhẹ nhàng yêu cầu họ nói rõ hơn về thể loại hoặc nội dung muốn xem.
            """
            # 1: CHATTING
            system_prompt = {
                'role': 'user', 
                'content': prompt_1
            }
            chat_res = ollama.chat(model='cinemate_agent', messages=[system_prompt] + clean_history)
            
            clean_chat = clean_ai_text(chat_res['message']['content'])
            return clean_chat, None

        # 2: QUERYING/SEARCHING
        sql_rule_prompt = f"""
        You are a Text-to-SQL system.
        Database Schema: Movies(id INT, title TEXT, year INT, genres TEXT, overview TEXT, vote_average REAL, vote_count INT, popularity REAL, keywords TEXT, poster_url TEXT, production_companies TEXT, production_countries TEXT, revenue INT, spoken_languages TEXT, tagline TEXT, cast TEXT, crew TEXT)
        
        QUICK REFERENCE GUIDE:
        - Genres: Action, Adventure, Science Fiction, Comedy, Drama, Horror, Romance, Thriller, Animation, Fantasy, etc.
        - Statistics: vote_average, vote_count, popularity, revenue
        - temporal: year
        - Cast/Crew: names (TEXT)
        - Production: production_companies, production_countries (comma-separated or JSON)
        
        MUST apply the following RULES based on user intent:
        1. "hot nhất", "thịnh hành", "nổi tiếng", "nhiều người xem" → ORDER BY vote_count DESC, popularity DESC
        2. "hay nhất", "siêu phẩm", "đánh giá cao" → WHERE vote_count > 10000 ORDER BY vote_average DESC, vote_count DESC
        3. "mới nhất", "gần đây", "vừa ra mắt" → WHERE year >= 2024 ORDER BY vote_count DESC, popularity DESC
        4. "cũ", "kinh điển", "cổ điển" → WHERE year <= 2010 ORDER BY vote_count DESC, popularity DESC
        5. Actor ("có Tom Cruise", "do DiCaprio đóng") → Add WHERE LOWER(cast) LIKE '%tom cruise%'
        6. Director ("của đạo diễn Nolan", "do James Cameron đạo diễn") → Add WHERE LOWER(crew) LIKE '%nolan%'
        7. Production companies ("phim của Marvel", "do Disney sản xuất") → Add WHERE LOWER(production_companies) LIKE '%marvel%'
        8. Countries ("phim Hàn Quốc", "phim Pháp") → Add WHERE LOWER(production_countries) LIKE '%korea%' or '%france%'
        9. Revenue ("doanh thu cao nhất", "phá đảo phòng vé") → Add ORDER BY revenue DESC
        10. Language ("tiếng Pháp", "tiếng Nhật") → Add WHERE LOWER(spoken_languages) LIKE '%french%' or '%japanese%'
        
        User's Request: "{user_message}"
        Return the most suitable SQL query (Output SQL only, ignore RESPONSE, LIMIT {num_recs}).
        """
        response_1 = ollama.chat(model='cinemate_agent', messages=[{'role': 'system', 'content': sql_rule_prompt}])
        sql_query = extract_sql_query(response_1['message']['content'])
        print(f"🤖 RAW SQL QUERY: {sql_query}")
        
        raw_db_results = []
        if sql_query and sql_query.strip().upper() != "NONE" and sql_query.strip().upper().startswith("SELECT"):
            # SQL path
            sql_query = re.sub(r'(?i)^SELECT\s+.*?\s+FROM', 'SELECT * FROM', sql_query.strip())
            if "LIMIT" not in sql_query.upper():
                sql_query += f" LIMIT {num_recs}"
            else:
                sql_query = re.sub(r'(?i)LIMIT\s+\d+', f'LIMIT {num_recs}', sql_query)
            # Sanitize reserved SQLite keywords used as column names
            # `cast` is a reserved keyword (CAST() function) — must be backtick-quoted
            sql_query = re.sub(r'(?i)\bLOWER\s*\(\s*cast\s*\)', 'LOWER(`cast`)', sql_query)
            sql_query = re.sub(r'(?i)\bLOWER\s*\(\s*crew\s*\)', 'LOWER(`crew`)', sql_query)
            
            print(f"⚙️ SQL PATH: {sql_query}")
            raw_db_results = run_sql(sql_query)
            
            # If SQL failed or returned empty → fallback to vector
            if not raw_db_results or isinstance(raw_db_results, str):
                eng_query = get_english_query(user_message)
                print("⚠️ SQL returned nothing, falling back to vector search...")
                print(f"🔍 VECTOR PATH — translated: '{eng_query}'")
                raw_db_results = search_vector(eng_query, top_k =num_recs)
        else:
            print("SQL Path failed, falling back to vector search...")
            # Semantic/descriptive query → Vector search path with RERANK
            eng_query = get_english_query(user_message)
            print(f"🔍 VECTOR PATH — translated: '{eng_query}'")
            
            # Retrieve MORE candidates than needed (3x oversampling)
            candidates = search_vector(eng_query, top_k=num_recs * 3)
            candidates_list = format_db_results_to_dicts(candidates)
            
            # Rerank — LLM picks the most relevant movies from candidates
            if len(candidates_list) > num_recs:
                candidate_info = "\n".join([
                    f"{i+1}. {m['title']} ({m['year']}) [{m['genres']}] - {(m.get('overview') or '')[:120]}"
                    for i, m in enumerate(candidates_list)
                ])
                rerank_prompt = f"""
                User's request: "{user_message}"
                Translated: "{eng_query}"
                Here are {len(candidates_list)} candidate movies from vector search: {candidate_info}
                
                TASK: Pick the {num_recs} movies that BEST match the user's FULL description.
                RULE: Return ONLY the {num_recs} number (e.g. "1, 5, 2")."""

                rerank_res = ollama.chat(model='cinemate_agent', messages=[{'role': 'system', 'content': rerank_prompt}])
                rerank_raw = rerank_res['message']['content']
                
                # Parse the selected indices
                picked_indices = [int(n.strip()) - 1 for n in re.findall(r'\d+', rerank_raw) if 0 < int(n.strip()) <= len(candidates_list)]
                picked_indices = picked_indices[:num_recs]
                
                if picked_indices:
                    raw_db_results = [candidates[i] for i in picked_indices]
                    print(f"🏆 RERANKED: picked {[candidates_list[i]['title'] for i in picked_indices]}")
                else:
                    raw_db_results = candidates[:num_recs]
            else:
                raw_db_results = candidates
            
        movies_list = format_db_results_to_dicts(raw_db_results)
        
        # Trả lời dựa trên dữ liệu thật
        movie_titles = ", ".join([m['title'] for m in movies_list if m['title'] != 'Unknown'])
        if not movie_titles: 
            return
            
        prompt_2 = f"""
        Bạn là CineMate, một trợ lý ảo, chuyên gia tư vấn phim ảnh. 
        Câu hỏi của người dùng: "{user_message}"
        Các phim phù hợp đã tìm thấy: {movie_titles}

        NHIỆM VỤ: Viết một câu trả lời ngắn gọn, tự nhiên và hấp dẫn (tối đa 40 từ) để giới thiệu danh sách phim này. 
        QUY TẮC: Không cần liệt kê lại tất cả tên phim, chỉ cần tóm tắt sức hấp dẫn của chúng và mời người dùng xem chi tiết bên dưới."""
        response_2 = ollama.chat(model='cinemate_agent', messages=[{'role': 'user', 'content': prompt_2}] + clean_history)
        
        clean_chat = clean_ai_text(response_2['message']['content'])
        return clean_chat, movies_list
        
    except Exception as e:
        return f"System Error : {str(e)}", []
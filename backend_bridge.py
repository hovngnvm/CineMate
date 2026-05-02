# project/backend_bridge.py
import time

def get_agent_response(user_message, num_recs):
    """
    Hàm giả lập gọi AI Agent.
    Sau này sẽ tích hợp LangChain hoặc LlamaIndex tại đây.
    """
    time.sleep(1.5)  # Giả lập độ trễ của AI
    
    # Giả lập danh sách phim trả về từ RAG
    mock_movies = [
        {"title": f"Phim ví dụ {i+1}", "year": 2024 - i, "genre": "Hành động, Phiêu lưu"}
        for i in range(num_recs)
    ]
    
    message = f"Dựa trên yêu cầu '{user_message}', tôi gợi ý cho bạn {num_recs} bộ phim sau:"
    
    return message, mock_movies
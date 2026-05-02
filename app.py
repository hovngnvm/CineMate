# project/app.py
import streamlit as st
from backend_bridge import get_agent_response # Task 2: Import từ module ngoài

# --- CẤU HÌNH SIDEBAR ---
with st.sidebar:
    st.title("🎬 CineMate")
    st.markdown("---")
    st.write("**Thành viên nhóm:** Nguyễn Thành Lộc, Huỳnh Hoàng Nam, Lâm Phước Dị")
    num_recommendations = st.slider("Số lượng phim muốn gợi ý", 1, 5, 3)

# --- THIẾT KẾ COMPONENT THẺ PHIM (Task 1) ---
def display_movie_cards(movie_list):
    # Chia thành các cột tùy theo số lượng phim
    cols = st.columns(len(movie_list))
    for i, movie in enumerate(movie_list):
        with cols[i]:
            with st.container(border=True): # Tạo khung viền vuông vức
                st.subheader(movie['title'])
                st.caption(f"📅 Năm: {movie['year']}")
                st.write(f"🎭 {movie['genre']}")

# --- GIAO DIỆN CHAT ---
st.title("Hybrid-RAG Movie Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "movies" in message:
            display_movie_cards(message["movies"])

if prompt := st.chat_input("Bạn muốn tìm phim gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Task 2: Gọi hàm từ backend_bridge
        response_text, movies = get_agent_response(prompt, num_recommendations)
        st.markdown(response_text)
        display_movie_cards(movies) # Task 1: Hiển thị thẻ phim
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "movies": movies
    })
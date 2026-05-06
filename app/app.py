import streamlit as st
from agent import get_agent_response

st.set_page_config(
    page_title="CineMate",
    page_icon="🎬",
    layout="wide",
)

# --- CUSTOM SIDEBAR STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ── Sidebar container ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29 0%, #1a1040 50%, #24243e 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem;
}

/* ── Logo block ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}
.sidebar-logo .logo-icon {
    font-size: 2.2rem;
    filter: drop-shadow(0 0 10px rgba(189,100,255,0.7));
}
.sidebar-logo .logo-text {
    font-size: 1.65rem;
    font-weight: 700;
    background: linear-gradient(90deg, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.04em;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    padding-left: 0.1rem;
}

/* ── Divider ── */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192,132,252,0.4), transparent);
    margin: 1rem 0;
}

/* ── Section label ── */
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(192,132,252,0.7);
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Feature badges ── */
.feature-badges {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.feature-badge {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.45rem 0.75rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.75);
    transition: background 0.2s;
}
.feature-badge:hover {
    background: rgba(192,132,252,0.12);
    border-color: rgba(192,132,252,0.3);
}
.feature-badge .badge-icon { font-size: 1rem; }

/* ── Slider overrides ── */
[data-testid="stSlider"] label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
[data-testid="stSlider"] .stSliderThumb {
    background: #c084fc !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #c084fc !important;
    box-shadow: 0 0 0 4px rgba(192,132,252,0.25) !important;
}

/* ── Footer ── */
.sidebar-footer {
    position: absolute;
    bottom: 1.5rem;
    left: 1.25rem;
    right: 1.25rem;
    text-align: center;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.2);
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# --- CẤU HÌNH SIDEBAR ---
with st.sidebar:
    # Logo & branding
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🎬</span>
        <span class="logo-text">CineMate</span>
    </div>
    <div class="sidebar-tagline">Your AI Movie Expert</div>
    """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown('<div class="sidebar-section-label">✦ Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-badges">
        <div class="feature-badge"><span class="badge-icon">🔍</span> Semantic Search</div>
        <div class="feature-badge"><span class="badge-icon">🧠</span> AI-Powered Picks</div>
        <div class="feature-badge"><span class="badge-icon">🎭</span> Genre/Mood Aware</div>
        <div class="feature-badge"><span class="badge-icon">⭐</span> Rating-Based Ranking</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Recommendations slider
    st.markdown('<div class="sidebar-section-label">⚙ Settings</div>', unsafe_allow_html=True)
    num_recommendations = st.slider("Number of Movies:", 1, 5, 3)

# --- THIẾT KẾ COMPONENT THẺ PHIM (Task 1) ---
def display_movie_cards(movie_list):
    # 1. Nếu Backend trả về None -> Đang chat phiếm, im lặng và thoát luôn
    if movie_list is None:
        return

    # 2. Nếu Backend trả về mảng rỗng [] -> Có tìm phim nhưng không thấy
    if len(movie_list) == 0:
        return

    # Khống chế hiển thị tối đa 5 cột trên 1 hàng để UI không bị vỡ
    num_cols = min(len(movie_list), 5)
    cols = st.columns(num_cols)
    
    for i, movie in enumerate(movie_list[:num_cols]):
        with cols[i]:
            with st.container(border=True):
                # Nâng cấp 1: Hiển thị Poster phim cực mượt
                if movie.get('poster_url'):
                    st.image(movie['poster_url'], width='stretch')
                else:
                    # Ảnh placeholder nếu phim không có poster
                    st.image("https://placehold.co/500x750/png?text=No+Poster", width='stretch')
                
                st.subheader(movie.get('title', 'Unknown'))
                st.caption(f"📅 Năm: {movie.get('year', 'N/A')}")
                st.write(f"🎭 {movie.get('genres', 'N/A')}")
                st.write(f"⭐ {movie.get('vote_average', 'N/A')}" + f" ({movie.get('vote_count', 'N/A')})")
                
                # Nâng cấp 2: Dùng expander để giấu phần tóm tắt cho gọn UI
                if movie.get('overview'):
                    with st.expander("📖 Xem tóm tắt"):
                        st.write(movie['overview'])

# --- GIAO DIỆN CHAT ---
st.title("What do you want to watch?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "movies" in message:
            display_movie_cards(message["movies"])

if prompt := st.chat_input("Ask CineMate"):
    # Snapshot history BEFORE appending to avoid duplicate user message in LLM context
    history_snapshot = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Task 2: Gọi hàm từ backend_bridge
        response_text, movies = get_agent_response(prompt, num_recommendations, history_snapshot)
        st.markdown(response_text)
        display_movie_cards(movies) # Task 1: Hiển thị thẻ phim
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "movies": movies or []
    })
"""
config.py

Tập trung toàn bộ hằng số, danh sách từ khóa, biểu thức chính quy
và định danh mô hình để tách biệt cấu hình khỏi logic nghiệp vụ.
"""

import re

OLLAMA_MODEL = "cinemate_agent"

# Danh sách từ khóa dùng để phân loại nhanh ý định người dùng là trò chuyện xã giao
CHAT_KEYWORDS: list[str] = [
    "chào", "hello", "hi ", "hey", "cảm ơn", "thanks", "thank you",
    "bạn là ai", "who are you", "tạm biệt", "bye", "ok", "oke",
    "hôm nay", "mệt", "buồn", "vui", "chán", "bạn ơi",
]

# Biểu thức chính quy nhận diện các câu mơ hồ kiểu nhờ chọn phim hộ, biên dịch một lần để tối ưu tốc độ
VAGUE_PICK_PATTERNS: list[re.Pattern] = [
    re.compile(r"chọn bừa"),
    re.compile(r"chọn giúp"),
    re.compile(r"bạn chọn"),
    re.compile(r"tùy bạn"),
    re.compile(r"phim gì cũng được"),
    re.compile(r"pick.*for me"),
    re.compile(r"surprise me"),
    re.compile(r"anything.*watch"),
]

# Tập hợp stop words tiếng Anh dùng cho BM25, bao gồm cả từ vựng phổ biến trong ngữ cảnh tìm phim
EN_STOPWORDS: set[str] = {
    "i", "i'm", "im", "me", "my", "we", "you", "he", "she", "they", "it",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "for", "of", "to", "in", "on", "at", "by", "up", "as", "or", "and",
    "but", "if", "so", "yet", "nor", "not", "no", "with", "from", "into",
    "about", "that", "this", "these", "those", "there", "here", "where",
    "which", "who", "whom", "whose", "what", "when", "how", "why",
    "looking", "look", "find", "want", "like", "see", "get", "go", "going",
    "film", "movie", "movies", "films", "show", "watch", "watching",
    "recommend", "recommendation", "suggest", "suggestion",
    "character", "named", "called", "name", "person", "someone", "somebody",
    "any", "some", "one", "two", "three", "all", "both", "each",
    "more", "most", "other", "such", "than", "then", "too", "very",
    "just", "also", "only", "even", "still", "already", "again",
    "else", "another", "group", "people", "getting", "steal", "secrets",
    "there", "their", "them", "they", "its", "our", "us",
}

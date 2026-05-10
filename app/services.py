"""
services.py

Lớp dịch vụ ML: dịch thuật Vi Anh, tạo embedding và xếp hạng lại bằng Cross Encoder.
Mỗi mô hình được nạp theo kiểu singleton để tránh tải lại nhiều lần.
"""

import warnings
import transformers
import torch

warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
transformers.logging.set_verbosity_error()

from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# Chỉ embedding model dùng GPU, các mô hình còn lại chạy trên CPU để không vượt quá 6 GB VRAM
_DEVICE_CPU = "cpu"
_DEVICE_EMBED = "cuda" if torch.cuda.is_available() else "cpu"

_translator_model = None
_translator_tokenizer = None
_embedding_model = None
_reranker_model = None


def _get_translator():
    global _translator_model, _translator_tokenizer
    if _translator_model is None:
        print("⏳ Loading Translation Model (Helsinki-NLP)...")
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        _translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator_model = MarianMTModel.from_pretrained(model_name).to(_DEVICE_CPU)
    return _translator_tokenizer, _translator_model


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("⏳ Loading Embedding Model (paraphrase-multilingual-MiniLM-L12-v2)...")
        _embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device=_DEVICE_EMBED
        )
    return _embedding_model


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        print("⏳ Loading Reranker Model (BAAI/bge-reranker-v2-m3)...")
        _reranker_model = CrossEncoder(
            'BAAI/bge-reranker-v2-m3', max_length=512, device=_DEVICE_CPU
        )
    return _reranker_model


def translate_vi_to_en(vietnamese_query):
    """Dịch câu truy vấn tiếng Việt sang tiếng Anh, trả về bản gốc nếu dịch thất bại."""
    try:
        tokenizer, model = _get_translator()
        inputs = tokenizer(vietnamese_query, return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE_CPU) for k, v in inputs.items()}
        translated_tokens = model.generate(**inputs)
        english_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return english_text.strip()
    except Exception as e:
        print(f"- Translation Error: {e}")
        return vietnamese_query


def query_embedding(text):
    """Chuyển đổi văn bản thành vector embedding dạng danh sách số thực cho ChromaDB."""
    model = _get_embedding_model()
    return model.encode(text).tolist()


def reranking(eng_query, candidates_list, top_k):
    """Xếp hạng lại danh sách ứng viên bằng Cross Encoder dựa trên độ tương đồng ngữ nghĩa."""
    if not candidates_list:
        return []

    reranker = _get_reranker()

    # Ghép từng cặp truy vấn với ngữ cảnh phim để Cross Encoder chấm điểm
    pairs = []
    for movie in candidates_list:
        movie_context = f"{movie['title']} ({movie['year']}) - Genres: {movie['genres']} - Overview: {movie.get('overview', '')}"
        pairs.append([eng_query, movie_context])

    print(f"- Scoring {len(pairs)} candidates using Cross-Encoder...")
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        candidates_list[i]['rerank_score'] = float(score)

    candidates_list.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates_list[:top_k]
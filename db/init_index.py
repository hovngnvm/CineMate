import logging
import re
import pickle
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movie_db.csv"
BM25_PATH = BASE_DIR / "db" / "bm25_index.pkl"


def init_index():
    """
    Xây dựng chỉ mục BM25 từ file CSV phim.

    Gom các cột văn bản thành một tài liệu cho mỗi dòng, token hóa
    và xây dựng mô hình BM25Okapi. Lưu cả mô hình lẫn danh sách bản ghi
    vào file pickle để truy xuất nhanh khi chạy ứng dụng.
    """
    logger.info("Đang tải dữ liệu từ CSV...")
    df = pd.read_csv(DATA_PATH)

    tokenized_corpus = []
    records = []

    for _, row in df.iterrows():
        bm25_columns = [
            "title", "year", "genres", "overview", "keywords",
            "cast", "crew", "production_companies",
            "production_countries", "tagline"
        ]

        parts = []
        for col in bm25_columns:
            val = str(row.get(col, '')).strip()
            if val and val.lower() != 'nan':
                parts.append(val)

        # Ghép tất cả các cột thành một chuỗi rồi chuẩn hóa và token hóa
        text = " ".join(parts)
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        tokens = clean_text.lower().split()
        tokenized_corpus.append(tokens)

        records.append(row.to_dict())

    logger.info("Đang xây dựng Index BM25...")
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"Đang lưu xuống {BM25_PATH}...")
    with open(BM25_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'records': records}, f)

    logger.info("Xây dựng BM25 thành công!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_index()
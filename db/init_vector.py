import logging
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movie_db.csv"
CHROMA_PATH = BASE_DIR / "db" / "chroma_storage"

BATCH_SIZE = 5000

TEXT_COLUMNS = ["title", "genres", "overview", "keywords", "production_companies", "production_countries", "tagline", "spoken_languages", "cast", "crew"]
META_COLUMNS = ["title", "genres", "year", "overview", "popularity", "poster_url", "vote_average", "vote_count", "revenue"]


def _build_document(row: pd.Series) -> str:
    """Ghép các cột văn bản của một dòng CSV thành chuỗi tài liệu duy nhất để tạo embedding."""
    parts: list[str] = []
    for col in TEXT_COLUMNS:
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            parts.append(f"{col.capitalize()}: {value}")
    return ". ".join(parts)


def _build_metadata(row: pd.Series) -> dict:
    """Trích xuất metadata phẳng từ một dòng CSV để lưu kèm vector trong ChromaDB."""
    meta: dict = {}
    for col in META_COLUMNS:
        value = row.get(col, "")
        value = str(value).strip() if pd.notna(value) else ""
        meta[col] = value

    meta["id"] = str(row["id"])
    return meta


def init_vector() -> None:
    """
    Xây dựng vector store ChromaDB từ file CSV phim.

    Mỗi dòng được ghép thành tài liệu, nhúng bằng mô hình đa ngôn ngữ
    và lưu kèm metadata. Upsert theo batch để kiểm soát bộ nhớ đỉnh.
    An toàn khi chạy lại nhờ upsert cập nhật ID trùng.
    """
    logger.info("Loading SentenceTransformer model …")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    logger.info("Reading data from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    logger.info("Loaded %d rows.", total)

    logger.info("Building document strings …")
    documents: list[str] = df.apply(_build_document, axis=1).tolist()
    ids: list[str] = df["id"].astype(str).tolist()

    logger.info("Initializing ChromaDB at %s …", CHROMA_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="movies")

    logger.info(
        "Encoding & upserting %d documents (batch_size=%d) …",
        total, BATCH_SIZE,
    )

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]

        # Chỉ nhúng batch hiện tại để tránh giữ toàn bộ embedding trong RAM cùng lúc
        batch_embeddings = model.encode(
            batch_docs,
            show_progress_bar=True,
            batch_size=256,
        ).tolist()

        batch_metadata = [
            _build_metadata(df.iloc[i]) for i in range(start, end)
        ]

        collection.upsert(
            documents=batch_docs,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metadata,
        )

        logger.info("Upserted %d / %d vectors", end, total)

    logger.info("Vector DB created successfully — %d total vectors.", total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_vector()
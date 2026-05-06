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
    """
    Compose a single document string from a CSV row.

    Format:
        Title: <title>. Overview: <overview>.
        Genres: <genres>.

    Empty / NaN fields are silently skipped.
    """
    parts: list[str] = []
    for col in TEXT_COLUMNS:
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            parts.append(f"{col.capitalize()}: {value}")
    return ". ".join(parts)


def _build_metadata(row: pd.Series) -> dict:
    """Return a flat dict of metadata fields for ChromaDB storage."""
    meta: dict = {}
    for col in META_COLUMNS:
        value = row.get(col, "")
        value = str(value).strip() if pd.notna(value) else ""
        meta[col] = value
        
    meta["id"] = str(row["id"])
    return meta


def init_vector() -> None:
    """
    Build the ChromaDB vector store from ``movie_db.csv``.

    For every row the script:
    1. Combines *title, overview, genres* into one
       document string that is embedded with ``paraphrase-multilingual-MiniLM-L12-v2``.
    2. Stores additional metadata (*title, genres,
       release_date*) so downstream queries can filter/display
       results without a separate SQL lookup.
    3. Upserts in batches of ``BATCH_SIZE`` to keep peak memory bounded.

    Safe to re-run — uses ``upsert`` so duplicate IDs are updated.
    """
    # Load model
    logger.info("Loading SentenceTransformer model …")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Read data
    logger.info("Reading data from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    logger.info("Loaded %d rows.", total)

    # Build document strings & IDs up-front (lightweight — just text)
    logger.info("Building document strings …")
    documents: list[str] = df.apply(_build_document, axis=1).tolist()
    ids: list[str] = df["id"].astype(str).tolist()

    # Init ChromaDB
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

        # Encode only this batch — avoids holding all embeddings in RAM
        batch_embeddings = model.encode(
            batch_docs,
            show_progress_bar=True,
            batch_size=256,          # internal GPU/CPU micro-batch
        ).tolist()

        # Build per-row metadata
        batch_metadata = [
            _build_metadata(df.iloc[i]) for i in range(start, end)
        ]

        # Upsert instead of add → idempotent, safe to re-run
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
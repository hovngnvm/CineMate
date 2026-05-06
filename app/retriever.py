import logging
import sqlite3
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import transformers

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "cinemate.db"
CHROMA_PATH = BASE_DIR / "db" / "chroma_storage"

# Singleton holders
_embedding_model: SentenceTransformer | None = None
_chroma_collection: chromadb.Collection | None = None

# Embedding Model
def _get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for all subsequent calls."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model (one-time)…")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model

def _get_chroma_collection() -> chromadb.Collection:
    """Open the ChromaDB collection once and cache the handle."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_collection(name="movies")
    return _chroma_collection

# SQL Query
def run_sql(query_string: str, params: tuple = ()) -> list | str:
    """
    Execute a read query against the SQLite database.
    Parameters
    ----------
    query_string : str
        SQL query — use ``?`` placeholders for parameters.
    params : tuple
        Values to bind to the placeholders (prevents SQL injection).
    Returns
    -------
    list
        Rows returned by the query.
    str
        An error message if the query fails.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query_string, params)
            return cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("SQL query failed: %s | query: %s", exc, query_string)
        return f"Lỗi truy vấn: {exc}"

# Vector Search
def search_vector(text_query: str, top_k: int = 3) -> list[tuple]:
    """
    Semantic search over the movie vector store.
    Parameters
    ----------
    text_query : str
        Natural-language query to embed and search.
    top_k : int
        Number of nearest-neighbour results to return.
    Returns
    -------
    list[tuple]
        Each tuple contains (id, title, year, genres, overview, poster_url).
    """
    model = _get_embedding_model()
    collection = _get_chroma_collection()
    query_embedding = model.encode(text_query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return [
        {
            "id": results["metadatas"][0][i].get("id", ""),
            "title": results["metadatas"][0][i].get("title", ""),
            "year": results["metadatas"][0][i].get("year", ""),
            "genres": results["metadatas"][0][i].get("genres", ""),
            "overview": results["metadatas"][0][i].get("overview", ""),
            "vote_average": results["metadatas"][0][i].get("vote_average", "N/A"),
            "vote_count": results["metadatas"][0][i].get("vote_count", "N/A"),
            "poster_url": results["metadatas"][0][i].get("poster_url", ""),

        }
        for i in range(len(results["ids"][0]))
    ]
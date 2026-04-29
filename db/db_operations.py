import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "cinemate.db"
CHROMA_PATH = BASE_DIR / "db" / "chroma_storage"


def run_sql(query_string):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(query_string)
        results = cursor.fetchall()

        conn.close()
        return results

    except Exception:
        return "Lỗi truy vấn"


def search_vector(text_query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name="movies")

    query_embedding = model.encode([text_query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    movies = []

    for i in range(len(results["ids"][0])):
        movies.append({
            "id": results["metadatas"][0][i]["id"],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })

    return movies

if __name__ == "__main__":
    print(run_sql("SELECT * FROM Movies LIMIT 3;"))
    print(search_vector("animated movie about toys", top_k=3))
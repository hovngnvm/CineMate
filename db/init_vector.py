import pandas as pd
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_movies.csv"
CHROMA_PATH = BASE_DIR / "db" / "chroma_storage"


def init_vector():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Reading data...")
    df = pd.read_csv(DATA_PATH)

    documents = df["document_context"].fillna("").astype(str).tolist()
    ids = df["movieId"].astype(str).tolist()

    print("Creating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="movies")

    batch_size = 5000

    print("Saving to ChromaDB...")
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))

        batch_docs = documents[start:end]
        batch_embeddings = embeddings[start:end]
        batch_ids = ids[start:end]
        batch_metadata = [{"id": movie_id} for movie_id in batch_ids]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metadata
        )

        print(f"Saved {end}/{len(documents)} vectors")

    print("Vector DB created successfully")
    print(f"Total vectors: {len(documents)}")


if __name__ == "__main__":
    init_vector()
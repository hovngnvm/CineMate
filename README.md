# 🎬 CineMate

An AI movie recommendation agent built on a Hybrid RAG architecture. Users chat naturally in Vietnamese to discover films — the system decides whether to answer with structured SQL, semantic vector search, or a blend of both, all running locally without external API calls.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4F00?style=flat&logo=chroma&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

## Architecture

```text
                          ┌──────────────────────┐
                          │   Streamlit Web UI    │
                          │      (app.py)         │
                          └──────────┬───────────┘
                                     │ Vietnamese query
                          ┌──────────▼───────────┐
                          │    Intent Router      │
                          │  Keyword → LLM gate   │
                          │     (agent.py)        │
                          └─────┬──────────┬─────┘
                                │          │
                     ┌──────────▼──┐  ┌────▼──────────┐
                     │    CHAT     │  │    SEARCH      │
                     │  Ollama LLM │  │                │
                     └──────┬──────┘  └────┬───────────┘
                            │              │
                            ▼              ▼
                      Direct reply    ┌────────────────┐
                      (no movies)     │ Text-to-SQL    │
                                      │ LLM → SQLite   │
                                      └───┬────────────┘
                                          │ results?
                                    ┌─────┤
                                  yes     no
                                    │     │
                                    │     ▼
                                    │  ┌────────────────────┐
                                    │  │ Named-Entity SQL   │
                                    │  │ Regex → LIKE query │
                                    │  └───┬────────────────┘
                                    │      │ results?
                                    │  ┌───┤
                                    │ yes  no
                                    │  │   │
                                    │  │   ▼
                                    │  │ ┌──────────────────────────────────┐
                                    │  │ │        Hybrid RAG Pipeline       │
                                    │  │ │                                  │
                                    │  │ │  VI → EN Translation (MarianMT)  │
                                    │  │ │  Query Expansion (LLM keywords)  │
                                    │  │ │          ┌─────┴─────┐           │
                                    │  │ │    Vector Search   BM25 Search   │
                                    │  │ │     (ChromaDB)    (rank_bm25)    │
                                    │  │ │          └─────┬─────┘           │
                                    │  │ │       RRF Fusion (k=60)          │
                                    │  │ │    Cross-Encoder Reranking       │
                                    │  │ │      (BAAI/bge-reranker)         │
                                    │  │ └────────────┬─────────────────────┘
                                    │  │              │
                                    ▼  ▼              ▼
                               ┌──────────────────────────┐
                               │  Response Generation     │
                               │  LLM → Vietnamese reply  │
                               └────────────┬─────────────┘
                                            ▼
                                    Movie Cards & Chat
```

## Project Structure

```text
CINEMATE/
├── app/
│   ├── app.py               Streamlit UI, movie card grid, chat interface
│   ├── agent.py              Orchestrator: intent routing, branch delegation
│   ├── config.py             Constants, keyword lists, regex patterns
│   ├── prompts.py            LLM prompt templates (router, SQL, chat, response)
│   ├── rag.py                Hybrid RAG: query expansion, RRF fusion, named-entity fallback
│   ├── retriever.py          Data access: SQLite queries, ChromaDB search, BM25 lookup
│   ├── services.py           ML services: translation, embedding, Cross-Encoder reranking
│   ├── text_processing.py    SQL extraction, response cleaning, result formatting
│   └── .streamlit/
│       └── config.toml       Streamlit server configuration
│
├── db/
│   ├── init_sqlite.py        Populate SQLite from CSV
│   ├── init_vector.py        Generate embeddings and populate ChromaDB
│   ├── init_index.py         Build BM25 index and serialize to pickle
│   ├── cinemate.db           SQLite database (generated)
│   ├── chroma_storage/       ChromaDB persistent store (generated)
│   └── bm25_index.pkl        BM25 index (generated)
│
├── engine/
│   ├── Modelfile             Ollama model config and system prompt
│   └── cinemate_agent.gguf   Quantized model weights (Qwen3 8B Q4_K_M)
│
├── data/
│   └── movie_db.csv          Source dataset
│
├── requirements.txt
└── README.md
```

## Pipeline

### Intent Routing

Every user message first passes through a two-stage classifier. A fast keyword check catches obvious greetings and small talk without calling the LLM. Ambiguous messages go to the Ollama model, which outputs a single word — `SEARCH` or `CHAT`. The chat branch returns a conversational reply with no database access; the search branch activates the retrieval cascade below.

### Text-to-SQL

The LLM receives the user query along with the full `Movies` table schema, a mapping of Vietnamese intent phrases to SQL patterns, and few-shot examples. It generates a single `SELECT` query. The output is sanitized — columns like `cast` and `crew` are backtick-wrapped, `SELECT *` is enforced, and `LIMIT` is always applied. If the query returns rows, those are used directly.

### Named-Entity Fallback

When SQL generation fails or returns nothing, a regex scanner extracts proper-noun sequences (including Vietnamese Unicode names) from the original query. Each name is searched against the `cast`, `keywords`, and `title` columns using parameterized `LIKE` queries. Results are deduplicated by movie ID.

### Hybrid RAG

If neither SQL path produces results, the full retrieval pipeline activates:

1. **Translation** — The Vietnamese query is translated to English via Helsinki-NLP's MarianMT model (running on CPU to stay within 6 GB VRAM).
2. **Query Expansion** — The LLM generates 6–10 domain-specific English keywords (plot themes, genre terms) to enrich the search.
3. **Parallel Retrieval** — The expanded query is embedded with `paraphrase-multilingual-MiniLM-L12-v2` for ChromaDB vector search, and simultaneously tokenized (with stop-word removal) for BM25 keyword matching. Both run concurrently via `ThreadPoolExecutor`.
4. **RRF Fusion** — Reciprocal Rank Fusion (k=60) merges the two ranked lists into a single score.
5. **Cross-Encoder Reranking** — The top candidates are re-scored by `BAAI/bge-reranker-v2-m3` against the original English translation. The final top-K are returned.

### Response Generation

The LLM receives the matched movie titles and composes a short, engaging Vietnamese introduction (≤ 50 words). The Streamlit frontend displays this alongside poster cards with year, rating, genres, and an expandable plot summary.

## Models

| Role                               | Model                                 | Device |
| ---------------------------------- | ------------------------------------- | ------ |
| LLM (routing, SQL, chat, response) | Qwen3 8B (Q4_K_M via Ollama)          | GPU    |
| Translation (VI → EN)              | Helsinki-NLP/opus-mt-vi-en            | CPU    |
| Embedding                          | paraphrase-multilingual-MiniLM-L12-v2 | GPU    |
| Reranking                          | BAAI/bge-reranker-v2-m3               | CPU    |

## Database Schema

### Movies (SQLite)

| Column                 | Type | Notes                                                       |
| ---------------------- | ---- | ----------------------------------------------------------- |
| `id`                   | INT  | Primary key                                                 |
| `title`                | TEXT | Movie title                                                 |
| `year`                 | INT  | Release year                                                |
| `genres`               | TEXT | English genre names, comma-separated                        |
| `overview`             | TEXT | Plot summary                                                |
| `vote_average`         | REAL | Rating score                                                |
| `vote_count`           | INT  | Number of votes                                             |
| `popularity`           | REAL | Popularity score                                            |
| `keywords`             | TEXT | Plot themes and character names                             |
| `poster_url`           | TEXT | Poster image URL                                            |
| `production_companies` | TEXT | Studio names                                                |
| `production_countries` | TEXT | Country names in English                                    |
| `revenue`              | INT  | Box office revenue                                          |
| `spoken_languages`     | TEXT | Language names in English                                   |
| `tagline`              | TEXT | Marketing tagline                                           |
| `cast`                 | TEXT | Actor names (reserved word, backtick-wrapped in queries)    |
| `crew`                 | TEXT | Director and writer names (reserved word, backtick-wrapped) |

### ChromaDB Collection: `movies`

Each document is a concatenation of all text columns. Embeddings are generated with the multilingual MiniLM model. Metadata mirrors the SQLite columns used for display.

### BM25 Index

Serialized as `bm25_index.pkl` — contains the `BM25Okapi` model and the full list of record dicts. Tokenized from the same text columns as ChromaDB documents, with punctuation stripped and lowercased.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- A GPU with ≥ 6 GB VRAM (CPU-only is possible but slower)
- `movie_db.csv` placed in the `data/` directory

### Installation

```bash
git clone https://github.com/your-username/cinemate.git
cd cinemate

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the LLM

```bash
cd engine
ollama create cinemate_agent -f Modelfile
cd ..
```

### Initialize Databases

```bash
python db/init_sqlite.py
python db/init_vector.py
python db/init_index.py
```

The vector and BM25 initialization will download embedding models on first run and may take several minutes depending on dataset size.

### Run

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` to start chatting with CineMate.

## License

This project is licensed under the MIT License.

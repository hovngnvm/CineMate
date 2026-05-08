# 🎬 CineMate - AI-Powered Movie Advisor (Hybrid RAG System)

## 📌 Project Overview

An intelligent **Movie Recommendation Agent** leveraging a Hybrid Retrieval-Augmented Generation (RAG) architecture. It allows users to chat naturally to find movies, combining deterministic SQL filtering for structured queries with semantic vector search for descriptive, abstract requests.

**Core Goal:** Provide accurate, conversational movie recommendations by seamlessly routing between Text-to-SQL generation and Vector Similarity Search, all powered by a local LLM.

## 🏗️ Architecture & Tech Stack

```text
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Web Interface                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ User Query (Vietnamese)
          ┌────────────────────▼────────────────────┐
          │             Intent Router (LLM)         │
          │         Classifies: CHAT vs SEARCH      │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │          Translation Model (NMT)        │
          │      VI → EN (Helsinki-NLP MarianMT)    │
          └────────────────────┬────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌──────────────────┐                       ┌──────────────────┐
│ Text-to-SQL (LLM)│──(Fails/Empty)───────►│  Vector Search   │
│   SQLite Query   │                       │    (ChromaDB)    │
└────────┬─────────┘                       └────────┬─────────┘
         │                                          │
         └───────────────────┬──────────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │ LLM Reranking & Gen   │
                 │ Synthesizes response  │
                 └───────────┬───────────┘
                             │
                             ▼
                    Movie Cards & Chat
```

- **Frontend / UI:** ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
- **Local LLM Engine:** ![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white)
- **Vector Database:** ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4F00?style=flat&logo=chroma&logoColor=white)
- **Relational DB:** ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
- **Embeddings & Translation:** ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

## 🗂️ Data Source

**Movie Dataset:** Contains detailed metadata including genres, overview, release year, cast, crew, production companies, and ratings. Ingested from `data/movie_db.csv`.

## 📁 Project Structure

```text
cinemate/
│
├── app/
│   ├── app.py                 # Streamlit web application & UI components
│   ├── agent.py               # Core LLM orchestration, routing, and translation
│   └── retriever.py           # DB connection layer (SQLite & ChromaDB)
│
├── db/
│   ├── init_sqlite.py         # Script to populate SQLite from CSV
│   ├── init_vector.py         # Script to generate embeddings & populate ChromaDB
│   └── chroma_storage/        # Persistent vector store (generated)
│
├── engine/
│   └── Modelfile              # Ollama system prompt & model configuration
│
├── data/                      # Raw datasets
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline Workflow

### 1. Intent Routing
- User input is sent to the LLM.
- The router categorizes the intent as either `CHAT` (casual conversation) or `SEARCH` (movie discovery).

### 2. Query Translation
- Queries are translated from Vietnamese to English using `Helsinki-NLP/opus-mt-vi-en` to match the English dataset.

### 3. Text-to-SQL (Structured Path)
- The LLM generates an SQL query based on strict rules (e.g., matching genres, actors, sorting by ratings).
- Executes against **SQLite** to retrieve precise matches.

### 4. Vector Search (Semantic Fallback)
- If the SQL query fails or returns no results, the system falls back to semantic search using **ChromaDB**.
- The query is embedded using `paraphrase-multilingual-MiniLM-L12-v2` to find conceptually similar movies based on their overviews and metadata.
- Candidate movies are passed to the LLM for **Reranking** to select the best matches.

### 5. Final Generation
- The LLM synthesizes a friendly, natural language response presenting the final movie selections to the user.

## 🚀 Key Engineering Highlights

| Feature               | Details                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| **Hybrid RAG**        | Combines deterministic SQL lookups with fuzzy semantic vector search.                  |
| **Local Execution**   | Runs entirely locally using Ollama and HuggingFace models, ensuring privacy and zero API costs. |
| **Dynamic Routing**   | LLM intelligently decides when to just chat vs. when to query the databases.           |
| **Multi-lingual**     | Built-in translation allows Vietnamese users to query English metadata seamlessly.     |
| **Reranking Phase**   | Oversamples vector results and uses the LLM to pick the absolute best candidates.      |

## 🛠️ How to Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cinemate.git
cd cinemate
```

### 2. Setup Python Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the Local LLM Model
Ensure Ollama is running, then create the custom agent:
```bash
cd engine
ollama create cinemate_agent -f Modelfile
cd ..
```

### 4. Initialize Databases
Make sure `movie_db.csv` is present in the `data/` folder, then run the initialization scripts:
```bash
# Populate SQLite
python db/init_sqlite.py

# Populate ChromaDB (Generates embeddings, may take a few minutes)
python db/init_vector.py
```

### 5. Start the Application
```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501` to start chatting with CineMate!

## 📊 Database Schema

### `Movies` (SQLite)
Stores structured metadata for Text-to-SQL queries.

| Column | Type | Description |
|---|---|---|
| `id` | INT | Unique Movie ID |
| `title` | TEXT | Movie Title |
| `year` | INT | Release Year |
| `genres` | TEXT | Comma-separated genres |
| `overview` | TEXT | Plot summary |
| `vote_average` | REAL | Rating (e.g., 8.5) |
| `vote_count` | INT | Number of votes |
| `popularity` | REAL | Popularity score |
| `poster_url` | TEXT | Image URL for the UI |
| `cast` / `crew` | TEXT | Actors and Directors |

## 📄 License
This project is licensed under the MIT License.

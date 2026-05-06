import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movie_db.csv"
DB_PATH = BASE_DIR / "db" / "cinemate.db"


def init_sqlite() -> None:
    """
    Read the cleaned movie CSV and load it into an SQLite database.

    Creates a ``Movies`` table with columns ``id``, ``title``, ``year``,
    ``genres`` and adds indexes on ``id`` and ``genres`` for fast lookups.
    Safe to re-run — uses ``IF NOT EXISTS`` for indexes and
    ``if_exists='replace'`` for the table.
    """
    # Read CSV
    logger.info("Reading CSV from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    required_cols = {"id", "title", "year", "genres", "overview", "vote_average", "vote_count", "popularity", "keywords", "poster_url", "production_companies", "production_countries", "revenue", "spoken_languages", "tagline", "cast", "crew"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    
    df = df[["id", "title", "year", "genres", "overview", "vote_average", "vote_count", "popularity", "keywords", "poster_url", "production_companies", "production_countries", "revenue", "spoken_languages", "tagline", "cast", "crew"]]

    # Write to SQLite
    logger.info("Writing %d rows to %s …", len(df), DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            name="Movies",
            con=conn,
            if_exists="replace",
            index=False,
        )

        # Create indexes
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_id ON Movies ("id");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_genres ON Movies ("genres");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_keywords ON Movies ("keywords");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_production_companies ON Movies ("production_companies");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_spoken_languages ON Movies ("spoken_languages");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_cast ON Movies ("cast");'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_movies_crew ON Movies ("crew");'
        )

    logger.info("Đã tạo cinemate.db thành công — %d rows inserted.", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_sqlite()
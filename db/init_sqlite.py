import sqlite3
from pathlib import Path
import pandas as pd


# Đường dẫn tới project
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_movies.csv"
DB_PATH = BASE_DIR / "db" / "cinemate.db"


def init_sqlite():
    # Đọc file CSV
    df = pd.read_csv(DATA_PATH)

    # Debug: xem cột thật trong file
    print("Columns trong CSV:", df.columns.tolist())

    # Lấy đúng cột từ CSV (chú ý là movieId, không phải id)
    df = df[["movieId", "title", "year", "genres"]]

    # Đổi tên movieId -> id (để đúng schema đề)
    df = df.rename(columns={"movieId": "id"})

    # Kết nối SQLite
    conn = sqlite3.connect(DB_PATH)

    # Ghi vào DB
    df.to_sql(
        name="Movies",
        con=conn,
        if_exists="replace",
        index=False
    )

    # Đóng kết nối
    conn.close()

    print("Đã tạo cinemate.db thành công.")
    print(f"Số dòng đã thêm: {len(df)}")


if __name__ == "__main__":
    init_sqlite()
"""Load the player-level extract produced by sql/01_player_features.sql."""

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
PARQUET_PATH = DATA_DIR / "player_features.parquet"
CSV_PATH = DATA_DIR / "player_features.csv"


def csv_to_parquet(csv_path: Path = CSV_PATH, parquet_path: Path = PARQUET_PATH) -> Path:
    """Convert the BigQuery CSV export to Parquet (smaller, typed, fast to read)."""
    duckdb.sql(
        f"COPY (SELECT * FROM read_csv_auto('{csv_path}')) "
        f"TO '{parquet_path}' (FORMAT parquet)"
    )
    return parquet_path


def load_players(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """One row per player: the churn label plus first-day features."""
    if not path.exists():
        csv_to_parquet(CSV_PATH, path)
    return duckdb.sql(f"SELECT * FROM '{path}'").df()

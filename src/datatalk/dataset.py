"""CSV loading and bounded metadata construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings


def validate_dataframe(df: pd.DataFrame, settings: Settings) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Dataset must contain at least one row and one column")
    if len(df) > settings.max_rows:
        raise ValueError(f"Dataset exceeds the {settings.max_rows} row limit")
    if len(df.columns) > 200:
        raise ValueError("Dataset exceeds the 200 column limit")
    if not all(isinstance(column, str) for column in df.columns):
        raise ValueError("DataFrame column names must be strings")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique")
    if df.memory_usage(deep=True).sum() > settings.max_csv_mb * 1024 * 1024 * 10:
        raise ValueError("DataFrame exceeds the in-memory size limit")


def load_csv(path: str | Path, settings: Settings, encoding: str = "utf-8-sig") -> pd.DataFrame:
    csv_path = Path(path).resolve(strict=True)
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise ValueError("Input must be a CSV file")
    if csv_path.stat().st_size > settings.max_csv_mb * 1024 * 1024:
        raise ValueError(f"CSV exceeds the {settings.max_csv_mb} MB file limit")
    df = pd.read_csv(csv_path, encoding=encoding, nrows=settings.max_rows + 1)
    validate_dataframe(df, settings)
    return df


def describe_dataframe(df: pd.DataFrame, sample_rows: int = 0) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "row_count": len(df),
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in df.dtypes.items()],
    }
    if sample_rows:
        metadata["sample_rows"] = json.loads(
            df.head(sample_rows).to_json(orient="records", date_format="iso")
        )
    if len(json.dumps(metadata, ensure_ascii=False)) > 40_000:
        raise ValueError("Dataset metadata exceeds the prompt size limit")
    return metadata

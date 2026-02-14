# SPDX-License-Identifier: Apache-2.0
"""Dataset-related models and enums."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class DatasetFormat(str, Enum):
    """Supported dataset file formats."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AUTO = "auto"  # Auto-detect from file extension


class DatasetLoadRequest(BaseModel):
    """Dataset loading request with pandas support."""

    data_files: str
    file_format: DatasetFormat = DatasetFormat.AUTO
    num_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    # CSV-specific options
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    # Columns to add (for missing required columns)
    added_columns: Optional[Dict[str, str]] = None

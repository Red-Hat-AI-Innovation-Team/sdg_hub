# SPDX-License-Identifier: Apache-2.0
"""File handling utilities: sanitization, format detection, dataset loading."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from config import FILENAME_SANITIZER, SUPPORTED_EXTENSIONS
from models.datasets import DatasetFormat

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename."""
    if not filename:
        return ""
    basename = os.path.basename(filename)
    sanitized = FILENAME_SANITIZER.sub("_", basename)
    return sanitized.strip("._")


def slugify_name(name: str, prefix: str = "flow") -> str:
    """Generate a directory-safe slug for flows/prompts."""
    base = name or ""
    slug = FILENAME_SANITIZER.sub("_", base.lower())
    slug = slug.strip("_") or f"{prefix}_{int(time.time())}"
    return slug


def detect_file_format(file_path: Path):
    """Auto-detect file format from extension.

    Raises HTTPException if format is not supported.
    Returns the DatasetFormat enum value.
    """
    suffix = file_path.suffix.lower()
    format_map = {
        ".jsonl": DatasetFormat.JSONL,
        ".json": DatasetFormat.JSON,
        ".csv": DatasetFormat.CSV,
        ".parquet": DatasetFormat.PARQUET,
        ".pq": DatasetFormat.PARQUET,
    }

    if suffix not in format_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{suffix}'. Please upload a dataset in one of these formats: JSONL (.jsonl), JSON (.json), CSV (.csv), or Parquet (.parquet)",
        )

    return format_map[suffix]


def is_supported_format(file_path: Path) -> bool:
    """Check if file has a supported format."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_dataset_as_pandas(
    file_path: Path,
    file_format,
    csv_delimiter: str = ",",
    csv_encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load dataset file as pandas DataFrame.

    Supports JSONL, JSON, CSV, and Parquet formats for optimal performance.
    """
    if file_format == DatasetFormat.AUTO:
        file_format = detect_file_format(file_path)

    logger.info(f"Loading dataset as pandas DataFrame (format: {file_format.value})")

    if file_format == DatasetFormat.JSONL:
        df = pd.read_json(file_path, lines=True)
    elif file_format == DatasetFormat.JSON:
        df = pd.read_json(file_path)
    elif file_format == DatasetFormat.CSV:
        df = pd.read_csv(file_path, delimiter=csv_delimiter, encoding=csv_encoding)
    elif file_format == DatasetFormat.PARQUET:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_json(file_path, lines=True)

    return df

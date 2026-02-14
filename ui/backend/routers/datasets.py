# SPDX-License-Identifier: Apache-2.0
"""Dataset upload, loading, deduplication, schema, and preview endpoints."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from config import (
    UPLOADS_DIR,
    SUPPORTED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    MAX_UPLOAD_SIZE_BYTES,
)
from models.datasets import DatasetLoadRequest
from state import current_config
from utils.dataset_utils import _get_hashable_duplicate_mask
from utils.file_handling import sanitize_filename, load_dataset_as_pandas
from utils.security import ensure_within_directory, resolve_dataset_file

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/dataset/upload")
async def upload_dataset_file(request: Request):
    """Upload a dataset file and save it temporarily.

    Only accepts supported formats: JSONL, JSON, CSV, Parquet.
    Manually parses multipart form data to avoid FastAPI parsing issues.
    """
    try:
        # Log request details for debugging
        content_type = request.headers.get("content-type", "")
        logger.info("=== UPLOAD REQUEST DEBUG ===")
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Content-Length: {request.headers.get('content-length')}")

        # Read the entire body first to avoid async context issues with large files
        body = await request.body()
        logger.info(f"Body size: {len(body)} bytes")

        # Parse the multipart data manually
        # Extract boundary from content-type
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip('"')
                break

        if not boundary:
            raise HTTPException(
                status_code=400, detail="Missing boundary in content-type"
            )

        # Simple multipart parser for single file upload
        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)

        file_data = None
        filename = None

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            # Split headers from content
            if b"\r\n\r\n" in part:
                headers_section, content = part.split(b"\r\n\r\n", 1)
            elif b"\n\n" in part:
                headers_section, content = part.split(b"\n\n", 1)
            else:
                continue

            headers_str = headers_section.decode("utf-8", errors="ignore")

            # Check if this is the file field
            if 'name="file"' in headers_str:
                # Extract filename
                import re as regex

                filename_match = regex.search(r'filename="([^"]+)"', headers_str)
                if filename_match:
                    filename = filename_match.group(1)

                # Remove trailing boundary markers
                if content.endswith(b"--\r\n"):
                    content = content[:-4]
                elif content.endswith(b"--\n"):
                    content = content[:-3]
                elif content.endswith(b"\r\n"):
                    content = content[:-2]
                elif content.endswith(b"\n"):
                    content = content[:-1]

                file_data = content
                break

        if file_data is None or filename is None:
            raise HTTPException(status_code=400, detail="No file found in upload")

        logger.info(f"Parsed filename: {filename}")
        logger.info(f"File data size: {len(file_data)} bytes")
        logger.info("=== END DEBUG ===")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_filename = sanitize_filename(filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename provided.")

        # Validate file format
        file_extension = Path(safe_filename).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file_extension}'. Please upload a dataset in one of these formats: JSONL (.jsonl), JSON (.json), CSV (.csv), or Parquet (.parquet)",
            )

        file_path = ensure_within_directory(UPLOADS_DIR, UPLOADS_DIR / safe_filename)
        bytes_written = len(file_data)

        if bytes_written > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds max upload size of {MAX_UPLOAD_SIZE_MB} MB.",
            )

        try:
            with open(file_path, "wb") as destination:
                destination.write(file_data)
        except Exception:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            raise

        relative_path = Path("uploads") / safe_filename
        logger.info(
            f"Uploaded dataset file: {file_path} ({bytes_written} bytes, format: {file_extension})"
        )

        return {
            "status": "success",
            "message": f"File '{filename}' uploaded successfully",
            "file_path": str(relative_path),
            "file_size": bytes_written,
            "format": file_extension[1:],  # Remove the dot
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/api/dataset/load")
async def load_dataset_from_file(request: DatasetLoadRequest):
    """Load dataset from file using pandas for optimal performance.

    Supports multiple formats: JSONL, JSON, CSV, Parquet.
    """
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        safe_dataset_path = resolve_dataset_file(request.data_files)

        # Load dataset as pandas DataFrame
        df = load_dataset_as_pandas(
            safe_dataset_path,
            request.file_format,
            request.csv_delimiter,
            request.csv_encoding,
        )

        # Apply shuffle if requested
        if request.shuffle:
            df = df.sample(frac=1, random_state=request.seed).reset_index(drop=True)

        # Limit samples if specified
        if request.num_samples:
            df = df.head(min(request.num_samples, len(df)))

        # Add any missing columns with provided values
        if request.added_columns:
            for col_name, col_value in request.added_columns.items():
                if col_name not in df.columns:
                    df[col_name] = col_value
                    logger.info(f"Added missing column '{col_name}' with value: {col_value[:50]}..." if len(str(col_value)) > 50 else f"Added missing column '{col_name}' with value: {col_value}")

        # Store dataset as pandas DataFrame
        current_config["dataset"] = df
        current_config["dataset_info"] = {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        # Store loading parameters for worker process reconstruction
        current_config["dataset_load_params"] = {
            **request.model_dump(),
            "data_files": str(safe_dataset_path),
        }

        logger.info(
            f"Loaded dataset from {safe_dataset_path}: {len(df)} samples, {len(df.columns)} columns (pandas DataFrame)"
        )

        return {
            "status": "success",
            "message": f"Dataset loaded with {len(df)} samples",
            "dataset_info": current_config["dataset_info"],
            "format": request.file_format.value,
        }

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/api/dataset/check-duplicates")
async def check_dataset_duplicates():
    """Check for duplicate rows in the currently loaded dataset.
    
    Returns information about duplicates found, if any.
    Uses the same hashable-comparison logic as the sdg_hub library's
    validate_no_duplicates() so results are consistent.
    """
    try:
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        df = current_config["dataset"]
        
        # Check for duplicates using the same hashable logic as sdg_hub
        duplicate_mask = _get_hashable_duplicate_mask(df)
        num_duplicates = int(duplicate_mask.sum())
        
        if num_duplicates > 0:
            # Get sample of duplicate rows for preview (max 5)
            duplicate_indices = df[duplicate_mask].index.tolist()[:5]
            duplicate_samples = df.loc[duplicate_indices].to_dict(orient='records')
            
            return {
                "has_duplicates": True,
                "num_duplicates": num_duplicates,
                "total_rows": len(df),
                "unique_rows": len(df) - num_duplicates,
                "duplicate_samples": duplicate_samples,
            }
        else:
            return {
                "has_duplicates": False,
                "num_duplicates": 0,
                "total_rows": len(df),
                "unique_rows": len(df),
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dataset/remove-duplicates")
async def remove_dataset_duplicates():
    """Remove duplicate rows from the currently loaded dataset.
    
    Keeps the first occurrence of each duplicate.
    Saves the deduplicated dataset to a temp file so the generation worker
    (which runs in a separate process) uses the clean data instead of
    re-reading the original file.
    
    Uses the same hashable-comparison logic as the sdg_hub library's
    validate_no_duplicates() so that the deduped file will pass the
    library's pre-generation duplicate check.
    """
    try:
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        df = current_config["dataset"]
        original_count = len(df)
        
        # Remove duplicates using the same hashable logic as sdg_hub's
        # validate_no_duplicates().  Plain df.drop_duplicates() can miss
        # duplicates when cells contain unhashable types (lists, dicts,
        # numpy arrays) because pandas may compare them differently than
        # the library does.
        duplicate_mask = _get_hashable_duplicate_mask(df)
        df_deduplicated = df[~duplicate_mask].reset_index(drop=True)
        removed_count = original_count - len(df_deduplicated)
        
        # Update the in-memory dataset
        current_config["dataset"] = df_deduplicated
        current_config["dataset_info"] = {
            "num_samples": len(df_deduplicated),
            "columns": df_deduplicated.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df_deduplicated.dtypes.items()},
        }
        
        # Save the deduplicated dataset to a temp file so that the generation
        # worker process (which reloads from disk) gets the clean version.
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dedup_filename = f"dedup_{uuid.uuid4().hex[:12]}.jsonl"
        dedup_path = UPLOADS_DIR / dedup_filename
        df_deduplicated.to_json(str(dedup_path), orient='records', lines=True, force_ascii=False)
        
        # Update dataset_load_params to point at the deduplicated file.
        # The worker will now load this file instead of the original.
        if current_config.get("dataset_load_params"):
            current_config["dataset_load_params"]["data_files"] = str(dedup_path)
            current_config["dataset_load_params"]["file_format"] = "jsonl"
            # Clear num_samples/shuffle since the file already has the exact
            # rows we want (already shuffled and sliced before dedup).
            current_config["dataset_load_params"]["num_samples"] = None
            current_config["dataset_load_params"]["shuffle"] = False
        
        logger.info(f"Removed {removed_count} duplicate rows from dataset. {len(df_deduplicated)} rows remaining. Saved to {dedup_path}")
        
        return {
            "status": "success",
            "original_count": original_count,
            "removed_count": removed_count,
            "new_count": len(df_deduplicated),
            "message": f"Removed {removed_count} duplicate rows. {len(df_deduplicated)} rows remaining.",
            "dedup_data_files": str(dedup_path),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dataset/schema")
async def get_dataset_schema():
    """Get the required dataset schema for the selected flow.

    Note: In latest sdg_hub, get_dataset_schema() returns pd.DataFrame.
    """
    try:
        if not current_config["flow"]:
            raise HTTPException(status_code=400, detail="No flow selected")

        flow = current_config["flow"]
        schema_df = flow.get_dataset_schema()  # Returns pd.DataFrame in latest sdg_hub

        requirements = flow.get_dataset_requirements()

        # Handle pandas DataFrame return type
        return {
            "columns": schema_df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in schema_df.dtypes.items()},
            "requirements": requirements.model_dump() if requirements else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dataset/preview")
async def preview_dataset():
    """Get a preview of the loaded dataset (pandas DataFrame)."""
    try:
        if current_config["dataset"] is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        df = current_config["dataset"]

        # Get first 5 samples for preview
        preview_size = min(5, len(df))
        preview_df = df.head(preview_size)
        # Convert to dict format compatible with frontend (orient='list' for column-based)
        preview_data = preview_df.to_dict(orient="list")

        return {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": preview_data,
            "preview_size": preview_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

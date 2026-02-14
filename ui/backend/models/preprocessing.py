# SPDX-License-Identifier: Apache-2.0
"""Preprocessing-related models for document chunking and dataset creation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FileChunkConfig(BaseModel):
    """Per-file chunking configuration."""
    chunk_size: int = 1000
    overlap: int = 100


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = 1000  # Max tokens/words per chunk
    overlap: int = 100  # Overlap between chunks
    method: str = "word"  # 'word' or 'token'
    selected_files: Optional[List[str]] = None  # Optional: specific files to chunk
    file_configs: Optional[Dict[str, FileChunkConfig]] = None  # Optional: per-file configs


class ICLTemplate(BaseModel):
    """ICL (In-Context Learning) template structure."""
    icl_document: str
    icl_query_1: str
    icl_response_1: str = ""
    icl_query_2: str
    icl_response_2: str = ""
    icl_query_3: str
    icl_response_3: str = ""


class PreprocessingDatasetRequest(BaseModel):
    """Request to create a dataset from preprocessed documents.
    
    Flow-aware parameters allow the preprocessing to create only the columns
    that the selected flow actually requires.
    """
    job_id: str
    chunk_config: ChunkingConfig
    additional_columns: Dict[str, str] = {}  # Static columns to add
    icl_template: Optional[ICLTemplate] = None
    domain: str = ""
    document_outline: str = ""
    dataset_name: Optional[str] = None  # Custom name for the dataset (without extension)
    # Flow-aware column configuration
    content_column_name: str = "document"  # Name of the content column ('text' for text analysis, 'document' for others)
    include_domain: bool = True  # Whether to include the domain column
    include_document_outline: bool = True  # Whether to include the document_outline column

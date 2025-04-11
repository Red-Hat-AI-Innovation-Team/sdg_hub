#!/usr/bin/env python3
"""
PDF Document Processing and Export Tool

This script processes PDF documents using the docling library, performing OCR,
table detection, and exporting to multiple formats. Configuration is handled
through a YAML file, allowing flexible control over processing options.

Example Usage:
    # Using defaults
    python docparser_v2.py -i ./pdfs -o ./output

    # Using custom config
    python docparser_v2.py -i ./pdfs -o ./output -c config.yaml

See README.md for detailed configuration options and examples.
"""

# Standard
from pathlib import Path
import json
import time
from typing import Dict, Optional
import yaml

# Third Party
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.ocr_mac_model import OcrMacOptions
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions
from logger_config import setup_logger
import click

logger = setup_logger(__name__)

# Constants and type definitions
EXPORT_FORMATS = {
    "json": ("json", "export_to_dict"),      # Deep Search JSON format
    "text": ("txt", "export_to_text"),       # Plain text
    "markdown": ("md", "export_to_markdown"), # Markdown with structure
    "html": ("html", "export_to_html"),      # HTML with styling
    "doctags": ("doctags", "export_to_document_tokens")  # Document tokens
}

DEFAULT_CONFIG = {
    "pipeline": {
        "ocr": {
            "enabled": True,      # Enable/disable OCR processing
            "languages": ["es"],  # List of language codes (e.g., eng, fra, deu)
        },
        "tables": {
            "enabled": True,        # Enable/disable table detection
            "cell_matching": True,  # Enable/disable cell matching in tables
        },
        "performance": {
            "threads": 4,          # Number of processing threads
            "device": "auto"       # Device selection (auto, cpu, gpu)
        }
    },
    "export": {
        "formats": {
            "json": True,      # Deep Search JSON format
            "text": True,      # Plain text
            "markdown": True,  # Markdown with structure
            "html": True,     # HTML with styling
            "doctags": True   # Document tokens
        }
    }
}

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from file or return defaults."""
    if not config_path:
        return DEFAULT_CONFIG
        
    try:
        with config_path.open("r") as f:
            user_config = yaml.safe_load(f)
            # Merge with defaults to ensure all required fields exist
            return {**DEFAULT_CONFIG, **user_config}
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}. Using defaults.")
        return DEFAULT_CONFIG

def setup_pipeline_options(config: dict) -> PdfPipelineOptions:
    """Configure pipeline options from config dictionary."""
    pipeline_config = config["pipeline"]
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = pipeline_config["ocr"]["enabled"]
    pipeline_options.do_table_structure = pipeline_config["tables"]["enabled"]
    pipeline_options.table_structure_options.do_cell_matching = pipeline_config["tables"]["cell_matching"]
    pipeline_options.ocr_options.lang = pipeline_config["ocr"]["languages"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=pipeline_config["performance"]["threads"],
        device=getattr(AcceleratorDevice, pipeline_config["performance"]["device"].upper())
    )
    return pipeline_options

def export_document(conv_result, doc_filename: str, output_dir: Path, config: dict) -> None:
    """Export document in configured formats."""
    enabled_formats = {
        k: v for k, v in EXPORT_FORMATS.items() 
        if config["export"]["formats"].get(k, True)
    }
    
    for format_name, (extension, export_method) in enabled_formats.items():
        try:
            content = getattr(conv_result.document, export_method)()
            output_path = output_dir / f"{doc_filename}.{extension}"
            
            with output_path.open("w", encoding="utf-8") as fp:
                if isinstance(content, (dict, list)):
                    json.dump(content, fp, ensure_ascii=False, indent=2)
                else:
                    fp.write(content)
                    
            logger.debug(f"Successfully exported {format_name} format to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export {format_name} format: {str(e)}")
            raise

@click.command()
@click.option(
    "--input-dir", "-i",
    type=click.Path(path_type=Path),
    help="Directory containing the documents to convert",
    required=True,
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Directory to save the converted documents",
    required=True,
)
@click.option(
    "--config", "-c",
    type=click.Path(path_type=Path),
    help="Path to YAML configuration file",
    default=None,
)
def export_document_new_docling(
    input_dir: Path,
    output_dir: Path,
    config: Optional[Path],
):
    """Convert PDF documents and export them in multiple formats."""
    config_data = load_config(config)
    
    file_paths = list(input_dir.glob("*.pdf"))
    if not file_paths:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(file_paths)} PDF files to process")
    
    pipeline_options = setup_pipeline_options(config_data)
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = failure_count = 0
    start_time = time.time()
    
    for file_path in file_paths:
        logger.info(f"Processing {file_path}")
        try:
            conv_result = doc_converter.convert(file_path)
            doc_filename = conv_result.input.file.stem
            
            export_document(conv_result, doc_filename, output_dir, config_data)
            success_count += 1
            logger.info(f"Successfully processed {file_path}")
            
        except Exception as e:
            failure_count += 1
            logger.error(f"Failed to process {file_path}: {str(e)}")
            continue

    processing_time = time.time() - start_time
    
    logger.info(
        f"Processed {success_count + failure_count} docs in {processing_time:.2f} seconds"
        f"\n  Successful: {success_count}"
        f"\n  Failed: {failure_count}"
    )

if __name__ == "__main__":
    try:
        export_document_new_docling()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

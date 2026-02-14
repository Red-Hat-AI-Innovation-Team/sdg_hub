# SPDX-License-Identifier: Apache-2.0
"""PDF Preprocessing router – all /api/preprocessing/* endpoints."""

from pathlib import Path
from typing import Optional
import json
import logging
import os
import re
import shutil
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.responses import Response
import pandas as pd

from config import (
    UPLOADS_DIR,
    PDF_UPLOADS_DIR,
    PDF_CONVERTED_DIR,
)
from models.preprocessing import (
    ChunkingConfig,
    PreprocessingDatasetRequest,
)
from state import preprocessing_jobs
from utils.file_handling import sanitize_filename
from utils.preprocessing_utils import save_preprocessing_jobs as _save_preprocessing_jobs
from utils.security import ensure_within_directory, safe_join
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


def save_preprocessing_jobs():
    """Save preprocessing jobs to disk using extracted util."""
    _save_preprocessing_jobs(preprocessing_jobs)


@router.post("/api/preprocessing/upload-pdf")
async def upload_pdf_files(request: Request, existing_job_id: Optional[str] = None):
    """Upload one or more PDF or Markdown files for preprocessing.
    
    Returns a job_id to track the preprocessing progress.
    PDF files will need conversion, MD files are treated as pre-converted.
    
    Args:
        existing_job_id: Optional job ID to add files to an existing job.
                        If not provided, creates a new job.
    """
    try:
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        
        # Parse multipart data
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip('"')
                break
        
        if not boundary:
            raise HTTPException(status_code=400, detail="Missing boundary in content-type")
        
        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)
        
        uploaded_files = []
        pre_converted_files = []  # MD files that don't need conversion
        
        # Use existing job or create new one
        if existing_job_id and existing_job_id in preprocessing_jobs:
            job_id = existing_job_id
            # Sanitize job_id before any path construction (CWE-23)
            safe_job_id = os.path.basename(job_id)
            if not safe_job_id or not re.fullmatch(r'[A-Za-z0-9_.-]+', safe_job_id):
                raise HTTPException(status_code=400, detail="Invalid job ID format")
            job_dir = Path(preprocessing_jobs[safe_job_id]["job_dir"])
            # Reset status to allow re-conversion/re-chunking if needed
            # This allows adding new files even after dataset was created
            if preprocessing_jobs[safe_job_id]["status"] in ["converted", "dataset_created", "complete"]:
                preprocessing_jobs[safe_job_id]["status"] = "uploaded"
        else:
            # Server-generated ID — no user input
            safe_job_id = f"pdf_{int(time.time())}_{os.urandom(4).hex()}"
            job_id = safe_job_id
            job_dir = PDF_UPLOADS_DIR / safe_job_id
            job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create converted directory using sanitized job ID
        converted_dir = PDF_CONVERTED_DIR / safe_job_id
        converted_dir.mkdir(parents=True, exist_ok=True)
        
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            
            if b"\r\n\r\n" in part:
                headers_section, content = part.split(b"\r\n\r\n", 1)
            elif b"\n\n" in part:
                headers_section, content = part.split(b"\n\n", 1)
            else:
                continue
            
            headers_str = headers_section.decode("utf-8", errors="ignore")
            
            if 'name="files"' in headers_str or 'name="file"' in headers_str:
                import re as regex
                filename_match = regex.search(r'filename="([^"]+)"', headers_str)
                if filename_match:
                    filename = filename_match.group(1)
                    filename_lower = filename.lower()
                    
                    # Accept PDF and MD files
                    if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.md')):
                        continue
                    
                    # Clean trailing boundary markers
                    if content.endswith(b"--\r\n"):
                        content = content[:-4]
                    elif content.endswith(b"--\n"):
                        content = content[:-3]
                    elif content.endswith(b"\r\n"):
                        content = content[:-2]
                    elif content.endswith(b"\n"):
                        content = content[:-1]
                    
                    # Sanitize filename: basename strips dirs, sanitize strips to [A-Za-z0-9_.-]
                    safe_filename = sanitize_filename(os.path.basename(filename))
                    if not safe_filename:
                        continue
                    # Explicit allowlist validation - reject if not safe chars only
                    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', safe_filename):
                        logger.warning(f"Rejected unsafe filename after sanitization: {safe_filename}")
                        continue
                    
                    if filename_lower.endswith('.md'):
                        # MD file: save directly to converted directory
                        # CWE-23 hardening: generate server-side filename using UUID
                        # to completely eliminate user input from file system paths.
                        # Original name kept in metadata for display only.
                        disk_name = uuid.uuid4().hex[:16] + ".md"
                        dest_resolved = (converted_dir / disk_name).resolve()
                        with open(str(dest_resolved), "wb") as f:
                            f.write(content)
                        
                        uploaded_files.append({
                            "filename": safe_filename,
                            "size": len(content),
                            "path": str(dest_resolved),
                            "type": "markdown"
                        })
                        
                        # Add to pre-converted list
                        pre_converted_files.append({
                            "original": safe_filename,
                            "markdown": disk_name,
                            "path": str(dest_resolved)
                        })
                    else:
                        # PDF file: save to job directory for later conversion
                        # CWE-23 hardening: server-generated filename via UUID
                        disk_name = uuid.uuid4().hex[:16] + ".pdf"
                        dest_resolved = (job_dir / disk_name).resolve()
                        with open(str(dest_resolved), "wb") as f:
                            f.write(content)
                        
                        uploaded_files.append({
                            "filename": safe_filename,
                            "disk_name": disk_name,
                            "size": len(content),
                            "path": str(dest_resolved),
                            "type": "pdf"
                        })
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No PDF or Markdown files found in upload")
        
        # Initialize or update job tracking (use safe_job_id for all keys)
        if existing_job_id and existing_job_id in preprocessing_jobs:
            # Add files to existing job
            existing_files = preprocessing_jobs[safe_job_id]["files"]
            existing_filenames = {f["filename"] for f in existing_files}
            # Only add files that don't already exist
            new_files = [f for f in uploaded_files if f["filename"] not in existing_filenames]
            preprocessing_jobs[safe_job_id]["files"].extend(new_files)
            all_files = preprocessing_jobs[safe_job_id]["files"]
            
            # Add pre-converted MD files to markdown_files
            existing_md_originals = {mf.get("original") for mf in preprocessing_jobs[safe_job_id].get("markdown_files", [])}
            new_md_files = [f for f in pre_converted_files if f["original"] not in existing_md_originals]
            if new_md_files:
                if "markdown_files" not in preprocessing_jobs[safe_job_id]:
                    preprocessing_jobs[safe_job_id]["markdown_files"] = []
                preprocessing_jobs[safe_job_id]["markdown_files"].extend(new_md_files)
            
            # Update converted_dir if not set
            if not preprocessing_jobs[safe_job_id].get("converted_dir"):
                preprocessing_jobs[safe_job_id]["converted_dir"] = str(converted_dir)
            
            logger.info(f"📄 Added {len(new_files)} file(s) to existing job {safe_job_id} (total: {len(all_files)}, {len(new_md_files)} MD files pre-converted)")
        else:
            # Create new job
            preprocessing_jobs[safe_job_id] = {
                "status": "uploaded",
                "files": uploaded_files,
                "job_dir": str(job_dir),
                "converted_dir": str(converted_dir),
                "markdown_files": pre_converted_files,  # MD files are already "converted"
                "chunks": [],
                "created_at": time.time()
            }
            all_files = uploaded_files
            logger.info(f"📄 Uploaded {len(uploaded_files)} file(s) for new job {safe_job_id} ({len(pre_converted_files)} MD files pre-converted)")
        
        # Persist to disk
        save_preprocessing_jobs()
        
        # Determine message based on file types
        pdf_count = len([f for f in all_files if f.get("type") == "pdf"])
        md_count = len([f for f in all_files if f.get("type") == "markdown"])
        
        if pdf_count > 0 and md_count > 0:
            message = f"Job has {pdf_count} PDF file(s) ready for conversion and {md_count} Markdown file(s) ready for chunking."
        elif pdf_count > 0:
            message = f"Job has {pdf_count} PDF file(s). Ready for conversion."
        else:
            message = f"Job has {md_count} Markdown file(s). Ready for chunking."
        
        return {
            "status": "success",
            "job_id": safe_job_id,
            "files": all_files,  # Return all files in the job
            "pre_converted_files": pre_converted_files,  # MD files that are already "converted"
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/preprocessing/convert/{job_id}")
async def convert_pdfs_to_markdown(job_id: str, selected_files: Optional[str] = None):
    """Convert uploaded PDFs to markdown using docling.
    
    This is a streaming endpoint that returns progress updates.
    
    Args:
        job_id: The job ID
        selected_files: Optional comma-separated list of filenames to convert.
                       If not provided, converts all files.
    """
    job_id = os.path.basename(job_id)
    # Parse selected files from query param
    files_to_convert = None
    if selected_files:
        files_to_convert = set(f.strip() for f in selected_files.split(",") if f.strip())
    
    async def generate_conversion_events():
        nonlocal files_to_convert
        try:
            if job_id not in preprocessing_jobs:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                return
            
            job = preprocessing_jobs[job_id]
            
            # Allow conversion from uploaded, conversion_failed, converted, or chunked (for reconversion/additional files)
            if job["status"] not in ["uploaded", "conversion_failed", "converted", "chunked"]:
                status = job["status"]
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid job status: {status}'})}\n\n"
                return
            
            # Determine which files to convert
            if files_to_convert:
                files_list = [f for f in job["files"] if f["filename"] in files_to_convert]
            else:
                files_list = job["files"]
            
            total_files = len(files_list)
            if total_files == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No files to convert'})}\n\n"
                return
                
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting PDF conversion...', 'total_files': total_files})}\n\n"
            
            # Import docling components
            try:
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'docling not installed: {str(e)}. Please run: pip install docling'})}\n\n"
                return
            
            # Configure docling pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.AUTO,
            )
            
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Create output directory
            converted_dir = PDF_CONVERTED_DIR / job_id
            converted_dir.mkdir(parents=True, exist_ok=True)
            job["converted_dir"] = str(converted_dir)
            job["status"] = "converting"
            
            # Preserve existing converted files (for partial conversion)
            existing_markdown_files = job.get("markdown_files", []) or []
            existing_originals = set(f.get("original", "") for f in existing_markdown_files if isinstance(f, dict))
            
            # New files converted in this batch
            new_markdown_files = []
            
            for idx, file_info in enumerate(files_list):
                file_path = Path(file_info["path"])
                filename = file_info["filename"]
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Converting {filename}...', 'current': idx, 'total': total_files})}\n\n"
                
                try:
                    conv_result = doc_converter.convert(file_path)
                    
                    # Export to markdown
                    md_content = conv_result.document.export_to_markdown()
                    # Sanitize output filename: basename strips dirs, sanitize strips special chars
                    md_filename = sanitize_filename(os.path.basename(file_path.stem)) + ".md"
                    md_path = safe_join(converted_dir, md_filename)
                    
                    with open(str(md_path.resolve()), "w", encoding="utf-8") as f:
                        f.write(md_content)
                    
                    original_filename = file_info["filename"]
                    file_result = {
                        "original": original_filename,
                        "markdown": md_filename,
                        "path": str(md_path),
                        "size": len(md_content),
                        "preview": md_content[:500] + "..." if len(md_content) > 500 else md_content
                    }
                    new_markdown_files.append(file_result)
                    
                    yield f"data: {json.dumps({'type': 'file_complete', 'message': f'Converted {original_filename}', 'file': file_result, 'original': original_filename})}\n\n"
                    
                except Exception as e:
                    err_filename = file_info["filename"]
                    logger.error(f"Error converting {file_path}: {e}")
                    yield f"data: {json.dumps({'type': 'file_error', 'message': f'Error converting {err_filename}: {str(e)}'})}\n\n"
            
            # Merge new files with existing (remove duplicates by replacing with new version)
            merged_markdown_files = [f for f in existing_markdown_files if f.get("original") not in {nf["original"] for nf in new_markdown_files}]
            merged_markdown_files.extend(new_markdown_files)
            
            job["markdown_files"] = merged_markdown_files
            job["status"] = "converted" if merged_markdown_files else "conversion_failed"
            
            # Persist to disk after conversion
            save_preprocessing_jobs()
            
            yield f"data: {json.dumps({'type': 'complete', 'message': f'Conversion complete. {len(new_markdown_files)} file(s) converted.', 'markdown_files': new_markdown_files, 'total_converted': len(merged_markdown_files)})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in PDF conversion: {e}")
            if job_id in preprocessing_jobs:
                preprocessing_jobs[job_id]["status"] = "conversion_failed"
                save_preprocessing_jobs()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_conversion_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/api/preprocessing/download/{job_id}/{filename}")
async def download_converted_file(job_id: str, filename: str):
    """Download a converted markdown file."""
    job_id = os.path.basename(job_id)
    filename = os.path.basename(filename)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = preprocessing_jobs[job_id]
    converted_dir = job.get("converted_dir")
    
    if not converted_dir:
        raise HTTPException(status_code=404, detail="No converted files for this job")
    
    converted_path = Path(converted_dir).resolve()
    file_path = safe_join(converted_path, filename)
    
    if not file_path.exists():
        if not filename.endswith('.md'):
            file_path = safe_join(converted_path, os.path.basename(filename) + '.md')
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=str(file_path.resolve()),
        filename=os.path.basename(filename),
        media_type="text/markdown"
    )


@router.get("/api/preprocessing/pdf/{job_id}/{filename}")
async def serve_pdf_file(job_id: str, filename: str):
    """Serve an uploaded PDF file for viewing."""
    job_id = os.path.basename(job_id)
    filename = os.path.basename(filename)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = preprocessing_jobs[job_id]
    files = job.get("files", [])
    
    # Find the file metadata (filename is only used for matching, not path construction)
    file_info = next((f for f in files if f.get("filename") == filename), None)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"PDF file not found: {filename}")
    
    # Determine trusted base directory
    job_dir_path = job.get("job_dir", "")
    if job_dir_path:
        trusted_base = Path(job_dir_path).resolve()
    else:
        trusted_base = (PDF_UPLOADS_DIR / os.path.basename(job_id)).resolve()
    
    if not trusted_base.is_dir():
        raise HTTPException(status_code=404, detail="Job directory not found")
    
    # CWE-23 hardening: serve the file using ONLY server-controlled paths.
    for entry in trusted_base.iterdir():
        if not entry.is_file():
            continue
        # Only serve files matching UUID pattern (server-generated names)
        if not re.fullmatch(r'[a-f0-9]{16}\.[a-z]+', entry.name):
            continue
        # Check if this disk file matches the requested display name in metadata
        is_match = any(
            f.get("filename") == filename
            and entry.name == os.path.basename(str(f.get("path", "")))
            for f in files
        )
        if is_match:
            with open(str(entry.resolve()), "rb") as fh:
                content = fh.read()
            display_name = sanitize_filename(os.path.basename(filename)) or "document.pdf"
            return Response(
                content=content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"inline; filename={display_name}",
                    "X-Content-Type-Options": "nosniff",
                }
            )
    
    raise HTTPException(status_code=404, detail="PDF file not found on disk")


@router.get("/api/preprocessing/markdown-content/{job_id}/{filename}")
async def get_markdown_content(job_id: str, filename: str):
    """Get markdown file content as text (for comparison view)."""
    job_id = os.path.basename(job_id)
    filename = os.path.basename(filename)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = preprocessing_jobs[job_id]
    converted_dir = job.get("converted_dir")
    
    if not converted_dir:
        raise HTTPException(status_code=404, detail="No converted files for this job")
    
    converted_path = Path(converted_dir).resolve()
    file_path = safe_join(converted_path, filename)
    
    if not file_path.exists():
        if not filename.endswith('.md'):
            file_path = safe_join(converted_path, os.path.basename(filename) + '.md')
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Markdown file not found: {filename}")
    
    try:
        with open(str(file_path.resolve()), 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content, "filename": file_path.name}
    except Exception as e:
        logger.error(f"Error reading markdown file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/preprocessing/chunk/{job_id}")
async def chunk_markdown_documents(job_id: str, config: ChunkingConfig):
    """Chunk the converted markdown documents."""
    job_id = os.path.basename(job_id)
    try:
        if job_id not in preprocessing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = preprocessing_jobs[job_id]
        
        if job["status"] not in ["converted", "chunked", "uploaded", "dataset_created", "complete"]:
            current_status = job["status"]
            raise HTTPException(status_code=400, detail=f"Invalid job status: {current_status}. Documents must be converted first.")
        
        if not job["markdown_files"]:
            raise HTTPException(status_code=400, detail="No markdown files to chunk")
        
        # Import chunking utilities
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        
        # Determine which files to process
        files_to_process = job["markdown_files"]
        if config.selected_files:
            files_to_process = [
                md_file for md_file in job["markdown_files"]
                if md_file["original"] in config.selected_files
            ]
            logger.info(f"Processing selected files: {[f['original'] for f in files_to_process]}")
        
        # Get existing chunks for files not being re-chunked
        existing_chunks = []
        if config.selected_files and job.get("chunks"):
            existing_chunks = [
                chunk for chunk in job["chunks"]
                if chunk["source_file"] not in config.selected_files
            ]
        
        new_chunks = []
        
        for md_file in files_to_process:
            original_name = md_file["original"]
            
            # Get chunk config for this file
            if config.file_configs and original_name in config.file_configs:
                file_config = config.file_configs[original_name]
                chunk_size = file_config.chunk_size
                overlap = file_config.overlap
            else:
                chunk_size = config.chunk_size
                overlap = config.overlap
            
            # Calculate chunk size in characters (approximate)
            chunk_size_chars = chunk_size * 4
            overlap_chars = overlap * 4
            
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=chunk_size_chars,
                chunk_overlap=overlap_chars,
            )
            
            # Reconstruct path from trusted base + sanitized filename
            md_safe_name = sanitize_filename(os.path.basename(md_file.get("markdown", "")))
            if not md_safe_name:
                continue
            converted_base = Path(job.get("converted_dir", "")).resolve()
            md_validated_path = ensure_within_directory(converted_base, converted_base / md_safe_name)
            with open(str(md_validated_path.resolve()), "r", encoding="utf-8") as f:
                content = f.read()
            
            # Clean up markdown tables
            content = re.sub(r"-{2,}\|", "-|", content)
            content = re.sub(r"\  +\|", " |", content)
            
            # Split into chunks
            docs = text_splitter.create_documents([content])
            
            for idx, doc in enumerate(docs):
                new_chunks.append({
                    "source_file": original_name,
                    "chunk_index": idx,
                    "document": doc.page_content,
                    "char_count": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "chunk_config": {
                        "chunk_size": chunk_size,
                        "overlap": overlap
                    }
                })
        
        # Combine existing chunks with new chunks
        all_chunks = existing_chunks + new_chunks
        
        job["chunks"] = all_chunks
        job["chunk_config"] = config.model_dump()
        job["status"] = "chunked"
        
        # Persist to disk after chunking
        save_preprocessing_jobs()
        
        # Store per-file chunk configs for reference
        if "file_chunk_configs" not in job:
            job["file_chunk_configs"] = {}
        for md_file in files_to_process:
            original_name = md_file["original"]
            if config.file_configs and original_name in config.file_configs:
                file_config = config.file_configs[original_name]
                job["file_chunk_configs"][original_name] = {
                    "chunk_size": file_config.chunk_size,
                    "overlap": file_config.overlap
                }
            else:
                job["file_chunk_configs"][original_name] = {
                    "chunk_size": config.chunk_size,
                    "overlap": config.overlap
                }
        
        processed_files = [f["original"] for f in files_to_process]
        logger.info(f"📑 Chunked {len(files_to_process)} files into {len(new_chunks)} new chunks for job {job_id}")
        logger.info(f"📑 Total chunks: {len(all_chunks)} (including {len(existing_chunks)} existing chunks from other files)")
        
        # Calculate per-file chunk counts
        per_file_chunk_counts = {}
        for chunk in all_chunks:
            source = chunk.get("source_file", "unknown")
            per_file_chunk_counts[source] = per_file_chunk_counts.get(source, 0) + 1
        
        return {
            "status": "success",
            "job_id": job_id,
            "total_chunks": len(all_chunks),
            "new_chunks": len(new_chunks),
            "chunks_preview": new_chunks[:10],
            "processed_files": processed_files,
            "per_file_chunk_counts": per_file_chunk_counts,
            "config": config.model_dump()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/preprocessing/jobs")
async def list_preprocessing_jobs():
    """List all preprocessing jobs with their converted files."""
    jobs_list = []
    
    for job_id, job in preprocessing_jobs.items():
        markdown_files = job.get("markdown_files", [])
        if not markdown_files and job.get("status") not in ["converted", "chunked", "dataset_created"]:
            continue
        
        jobs_list.append({
            "job_id": job_id,
            "status": job.get("status"),
            "files": job.get("files", []),
            "markdown_files": markdown_files,
            "created_at": job.get("created_at"),
            "chunks_count": len(job.get("chunks", [])),
        })
    
    jobs_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return {
        "jobs": jobs_list,
        "total": len(jobs_list)
    }


@router.get("/api/preprocessing/datasets")
async def list_preprocessed_datasets():
    """List all preprocessed datasets (final JSONL files from PDF pipeline)."""
    datasets_list = []

    for job_id, job in preprocessing_jobs.items():
        if job.get("status") != "dataset_created":
            continue

        output_file = job.get("output_file")
        if not output_file:
            continue

        output_path = Path(output_file)
        if not output_path.exists():
            continue

        file_size = output_path.stat().st_size
        file_mtime = output_path.stat().st_mtime

        sample_count = 0
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                sample_count = sum(1 for line in f if line.strip())
        except Exception as e:
            logger.warning(f"Could not count samples in {output_path}: {e}")

        source_files = [f.get("filename", "Unknown") for f in job.get("files", [])]
        source_file_str = ", ".join(source_files) if source_files else "Unknown source"

        created_at = job.get("created_at")
        if not created_at or created_at == 0:
            created_at = file_mtime

        try:
            created_at_iso = datetime.fromtimestamp(created_at).isoformat()
        except (ValueError, TypeError, OSError):
            created_at_iso = datetime.now().isoformat()

        display_name = job.get("dataset_name", output_path.stem)
        
        datasets_list.append(
            {
                "job_id": job_id,
                "name": output_path.name,
                "display_name": display_name,
                "file_path": str(output_file),
                "file_size": file_size,
                "source_file": source_file_str,
                "source_files": source_files,
                "sample_count": sample_count,
                "num_chunks": len(job.get("chunks", [])),
                "created_at": created_at_iso,
                "status": "ready",
            }
        )

    datasets_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {"datasets": datasets_list, "total": len(datasets_list)}


@router.delete("/api/preprocessing/datasets/{job_id}")
async def delete_preprocessed_dataset(job_id: str):
    """Delete a preprocessed dataset and its associated job data."""
    job_id = os.path.basename(job_id)
    try:
        if job_id not in preprocessing_jobs:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        job = preprocessing_jobs[job_id]
        
        output_file = job.get("output_file")
        if output_file:
            output_path = Path(output_file).resolve()
            ensure_within_directory(UPLOADS_DIR, output_path)
            if output_path.exists():
                output_path.unlink()
                logger.info(f"🗑️ Deleted dataset file: {output_file}")
        
        if job.get("job_dir") and Path(job["job_dir"]).exists():
            job_dir_path = Path(job["job_dir"]).resolve()
            ensure_within_directory(PDF_UPLOADS_DIR, job_dir_path)
            shutil.rmtree(str(job_dir_path))
        
        if job.get("converted_dir") and Path(job["converted_dir"]).exists():
            converted_dir_path = Path(job["converted_dir"]).resolve()
            ensure_within_directory(PDF_CONVERTED_DIR, converted_dir_path)
            shutil.rmtree(str(converted_dir_path))
        
        del preprocessing_jobs[job_id]
        save_preprocessing_jobs()
        
        logger.info(f"🗑️ Deleted preprocessed dataset for job {job_id}")
        
        return {"status": "success", "message": f"Dataset {job_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/preprocessing/datasets/{job_id}/download")
async def download_preprocessed_dataset(job_id: str):
    """Download a preprocessed dataset file."""
    job_id = os.path.basename(job_id)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    job = preprocessing_jobs[job_id]
    output_file = job.get("output_file")
    
    if not output_file:
        raise HTTPException(status_code=404, detail="No dataset file found for this job")
    
    output_path = Path(output_file)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    
    return FileResponse(
        path=str(output_path.resolve()),
        filename=output_path.name,
        media_type="application/x-jsonlines"
    )


@router.get("/api/preprocessing/status/{job_id}")
async def get_preprocessing_status(job_id: str):
    """Get the current status of a preprocessing job."""
    job_id = os.path.basename(job_id)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = preprocessing_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "files_count": len(job.get("files", [])),
        "markdown_files_count": len(job.get("markdown_files", [])),
        "chunks_count": len(job.get("chunks", [])),
        "chunk_config": job.get("chunk_config"),
        "created_at": job.get("created_at")
    }


@router.get("/api/preprocessing/chunks/{job_id}")
async def get_preprocessing_chunks(job_id: str, offset: int = 0, limit: int = 10):
    """Get chunks from a preprocessing job with pagination."""
    job_id = os.path.basename(job_id)
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = preprocessing_jobs[job_id]
    chunks = job.get("chunks", [])
    
    return {
        "job_id": job_id,
        "total": len(chunks),
        "offset": offset,
        "limit": limit,
        "chunks": chunks[offset:offset + limit]
    }


@router.get("/api/preprocessing/icl-templates")
async def get_icl_templates():
    """Get example ICL (In-Context Learning) templates."""
    templates = [
        {
            "name": "IBM Software Revenue (Financial)",
            "description": "Real example from IBM Annual Report - Software and business performance analysis",
            "domain": "Corporate Finance",
            "template": {
                "icl_document": """## Management Discussion

International Business Machines Corporation and Subsidiary Companies

Software revenue of $27,085 million increased 8.3 percent as reported (9.0 percent adjusted for currency) in 2024 compared to the prior year, reflecting growth across all lines of business with double-digit growth in Red Hat and Automation and high single-digit growth in Transaction Processing. This revenue performance reflects the investments we have been making in Software, both organically and through acquisitions. Our investments in generative AI are contributing to growth, as we had strong demand for our generative AI products such as watsonx, Concert and our AI assistants. We also launched new products in the fourth quarter of 2024 including our next generation of watsonx Code Assistant and Guardium Quantum Safe.

Hybrid Platform & Solutions revenue of $18,808 million increased 8.1 percent as reported (8.7 percent adjusted for currency) in 2024 compared to the prior year. Within Hybrid Platform & Solutions, Red Hat revenue increased 11.4 percent as reported (12.0 percent adjusted for currency), which reflects the continued demand for our hybrid cloud solutions as clients are prioritizing application modernization on OpenShift containers and Ansible automation to optimize their IT spending and reduce operational complexity.""",
                "icl_query_1": "What contributed to IBM's 8.3% increase in Software revenue in 2024?",
                "icl_response_1": "IBM's 8.3% increase in Software revenue in 2024 was driven by growth across all lines of business, including double-digit growth in Red Hat and Automation, high single-digit growth in Transaction Processing, strong demand for generative AI products like watsonx and Concert, new product launches such as watsonx Code Assistant and Guardium Quantum Safe, and increased revenue from acquisitions including Apptio, StreamSets, and webMethods.",
                "icl_query_2": "How did IBM's Hybrid Platform & Solutions segment perform in 2024 and what fueled this growth?",
                "icl_response_2": "The Hybrid Platform & Solutions segment generated $18,808 million in revenue in 2024, an increase of 8.1% as reported. Growth was fueled by strong performance in Red Hat (up 11.4%), especially OpenShift and Ansible with double-digit growth, increased revenue from OpenShift Virtualization, and strong demand for hybrid cloud solutions enabling application modernization and IT optimization.",
                "icl_query_3": "What were the key AI-related products that contributed to IBM's revenue growth?",
                "icl_response_3": "The key AI-related products that contributed to IBM's revenue growth were the watsonx platform, Concert, AI assistants, and the new watsonx Code Assistant. Strong demand for these generative AI products contributed significantly to the Software segment's growth in 2024."
            }
        },
        {
            "name": "Environmental Conservation (Articles)",
            "description": "Example about environmental challenges and technological solutions",
            "domain": "Environmental Science",
            "template": {
                "icl_document": """The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution. Plastic debris and oil spills have devastated marine life, prompting a decline in tourism and fishing industries. Residents have organized weekly clean-up initiatives, but the scale of the problem overwhelms their efforts.

Technologists at the local university have developed an AI-powered buoy system to combat this. The buoys, equipped with solar panels and filtration technology, can identify and absorb oil spills while collecting microplastics. Data from the buoys is shared publicly, raising awareness and pressuring corporations to adopt sustainable practices. Though costly, the project has sparked hope for revitalizing the ecosystem and economy.""",
                "icl_query_1": "How does the technological solution address the economic and environmental challenges highlighted in the document?",
                "icl_response_1": "The AI-powered buoy system addresses environmental challenges by identifying and absorbing oil spills while collecting microplastics. It tackles economic challenges by helping revitalize the ecosystem, which could restore tourism and fishing industries. The public data sharing also pressures corporations to adopt sustainable practices, creating broader economic incentives for environmental protection.",
                "icl_query_2": "What implicit values or priorities do the community's actions and the technologists' project reflect?",
                "icl_response_2": "The community's weekly clean-up initiatives reflect values of collective responsibility, environmental stewardship, and grassroots activism. The technologists' project reflects values of innovation, scientific problem-solving, transparency (through public data sharing), and the belief that technology can address environmental crises. Both approaches share a commitment to restoring the local ecosystem.",
                "icl_query_3": "What unintended consequences might arise from the buoy project's success, considering the document's themes?",
                "icl_response_3": "If successful, the project might lead to over-reliance on technological solutions rather than addressing root causes of pollution. Restored beaches could attract more tourism, potentially creating new environmental pressures. The public data could also create tensions with corporations being named, and the high cost might make the solution inaccessible to other affected communities without similar university resources."
            }
        },
        {
            "name": "Regulatory Compliance (FINTRAC)",
            "description": "Real example from FINTRAC regulations on identity verification",
            "domain": "Financial Regulations",
            "template": {
                "icl_document": """## Overview

This guidance came into effect on June 1, 2021.

This guidance explains the methods that can be used by reporting entities (REs) to verify the identity of a person or an entity.

## 1. Meaning of verifying the identity of a person or an entity

It means to use the methods described in this guidance to ensure that the information in an identification document or from other informational sources matches the information that the person or entity provided.

Verifying identity is a foundational element of Canada's anti-money laundering and anti-terrorist financing regime and a key component of an RE's relationship with clients. It helps you to know your clients and to understand and assess any risk that may be associated to their transactions or activities.

## 2. How to verify the identity of a person

You can use any of the 5 methods described below to identify a person:

- 2.1 Government-issued photo identification method
- 2.2 Credit file method
- 2.3 Dual-process method
- 2.4 Affiliate or member method
- 2.5 Reliance method""",
                "icl_query_1": "In Canada, what are the methods for verifying someone's identity?",
                "icl_response_1": "In Canada, there are 5 methods for verifying a person's identity: (1) Government-issued photo identification method, (2) Credit file method, (3) Dual-process method, (4) Affiliate or member method, and (5) Reliance method. Reporting entities can use any of these methods to verify the identity of individuals.",
                "icl_query_2": "Why is it important to confirm a client's identity according to this guidance?",
                "icl_response_2": "Verifying identity is important because it is a foundational element of Canada's anti-money laundering and anti-terrorist financing regime. It helps reporting entities to know their clients and to understand and assess any risk that may be associated with their transactions or activities. It ensures that the information provided matches official identification documents or other informational sources.",
                "icl_query_3": "When did this FINTRAC guidance come into effect?",
                "icl_response_3": "This FINTRAC guidance came into effect on June 1, 2021. It provides reporting entities with the methods they can use to verify the identity of persons and entities in compliance with Canada's anti-money laundering and anti-terrorist financing regulations."
            }
        },
        {
            "name": "IBM Consulting Business (Financial)",
            "description": "Real example about IBM consulting signings and business performance",
            "domain": "Business Consulting",
            "template": {
                "icl_document": """## Consulting Signings and Book-to-Bill

| For the year ended December 31: | 2024 | 2023 | Yr.-to-Yr. Percent Change | Adjusted for Currency |
|-----------------------------------|-----------|-------------|-----------------------------|----------------------------------------------------|
| Total Consulting signings | $25,103M | $24,305M | 3.3% | 4.7% |

Consulting signings grew 3.3 percent as reported (4.7 percent adjusted for currency) in 2024, driven by our strong performance in signings in the fourth quarter of 2024. This performance reflects our ability to address client demands and the contribution of our generative AI solutions that help clients accelerate their digital transformations. Our book-to-bill ratio over the trailing twelve months was 1.21.

Book-to-bill represents the ratio of IBM Consulting signings to its revenue over the same period and is a useful indicator of the demand for our business over time. Signings are management's initial estimate of the value of a client's commitment under a services contract within IBM Consulting.""",
                "icl_query_1": "What was the total value of IBM's Consulting signings in 2024 and how did it compare to 2023?",
                "icl_response_1": "In 2024, IBM's total Consulting signings were valued at $25,103 million, representing a 3.3% increase from the 2023 value of $24,305 million. When adjusted for currency, the year-to-year increase was 4.7%.",
                "icl_query_2": "What factors contributed to the growth in IBM's Consulting signings in 2024?",
                "icl_response_2": "The growth in IBM's Consulting signings in 2024 was driven by strong performance in the fourth quarter and the impact of generative AI solutions, which supported clients in accelerating their digital transformations. IBM's ability to address client demands was a key factor in this growth.",
                "icl_query_3": "What does IBM's book-to-bill ratio indicate about its Consulting business in 2024?",
                "icl_response_3": "IBM's book-to-bill ratio for the trailing twelve months was 1.21, indicating that Consulting signings exceeded revenue. This suggests strong demand for IBM's Consulting services and is considered a useful indicator of future business performance, showing that new contract commitments are outpacing current service delivery."
            }
        }
    ]
    
    return {
        "templates": templates,
        "icl_fields": [
            "icl_document",
            "icl_query_1", "icl_response_1",
            "icl_query_2", "icl_response_2", 
            "icl_query_3", "icl_response_3"
        ]
    }


@router.post("/api/preprocessing/create-dataset/{job_id}")
async def create_dataset_from_preprocessing(job_id: str, request: PreprocessingDatasetRequest):
    """Create a final dataset from preprocessed and chunked documents."""
    job_id = os.path.basename(job_id)
    try:
        if job_id not in preprocessing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = preprocessing_jobs[job_id]
        
        if job["status"] != "chunked":
            current_status = job["status"]
            raise HTTPException(status_code=400, detail=f"Job must be chunked first. Current status: {current_status}")
        
        if not job.get("chunks"):
            raise HTTPException(status_code=400, detail="No chunks available")
        
        records = []
        for chunk in job["chunks"]:
            record = {
                request.content_column_name: chunk["document"],
            }
            
            if request.include_document_outline:
                record["document_outline"] = request.document_outline or f"Document chunk from {chunk['source_file']}"
            
            if request.include_domain:
                record["domain"] = request.domain or "General"
            
            if request.icl_template:
                record.update({
                    "icl_document": request.icl_template.icl_document,
                    "icl_query_1": request.icl_template.icl_query_1,
                    "icl_response_1": request.icl_template.icl_response_1,
                    "icl_query_2": request.icl_template.icl_query_2,
                    "icl_response_2": request.icl_template.icl_response_2,
                    "icl_query_3": request.icl_template.icl_query_3,
                    "icl_response_3": request.icl_template.icl_response_3,
                })
            
            record.update(request.additional_columns)
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if request.dataset_name:
            safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in request.dataset_name)
            output_filename = f"{safe_name}.jsonl"
        else:
            output_filename = f"preprocessed_{job_id}.jsonl"
        output_path = UPLOADS_DIR / output_filename
        
        df.to_json(output_path, orient="records", lines=True)
        
        job["status"] = "dataset_created"
        job["output_file"] = str(output_path)
        job["dataset_name"] = request.dataset_name or output_filename.replace('.jsonl', '')
        
        save_preprocessing_jobs()
        
        logger.info(f"📊 Created dataset with {len(records)} records from job {job_id}")
        
        return {
            "status": "success",
            "job_id": job_id,
            "file_path": f"uploads/{output_filename}",
            "num_records": len(records),
            "columns": list(df.columns),
            "preview": records[:2] if records else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset from preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/preprocessing/{job_id}")
async def cleanup_preprocessing_job(job_id: str):
    """Clean up a preprocessing job and its files."""
    job_id = os.path.basename(job_id)
    try:
        if job_id not in preprocessing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = preprocessing_jobs[job_id]
        
        if job.get("job_dir") and Path(job["job_dir"]).exists():
            job_dir_path = Path(job["job_dir"]).resolve()
            ensure_within_directory(PDF_UPLOADS_DIR, job_dir_path)
            shutil.rmtree(str(job_dir_path))
        
        if job.get("converted_dir") and Path(job["converted_dir"]).exists():
            converted_dir_path = Path(job["converted_dir"]).resolve()
            ensure_within_directory(PDF_CONVERTED_DIR, converted_dir_path)
            shutil.rmtree(str(converted_dir_path))
        
        del preprocessing_jobs[job_id]
        save_preprocessing_jobs()
        
        logger.info(f"🧹 Cleaned up preprocessing job {job_id}")
        
        return {"status": "success", "message": f"Job {job_id} cleaned up"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

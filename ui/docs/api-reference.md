# API Reference

Complete reference for the SDG Hub UI backend REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required. The API runs locally on your machine.

## Response Format

All responses are JSON:

```json
{
  "status": "success",
  "data": { ... }
}
```

Error responses:

```json
{
  "detail": "Error message here"
}
```

---

## Health Check

### GET /health

Check API server status.

**Response:**

```json
{
  "status": "healthy",
  "service": "sdg_hub_api"
}
```

---

## Flow Discovery

### GET /api/flows/list

List all available flows.

**Response:**

```json
[
  "Advanced Document Grounded QA Generation Flow",
  "Simple Summary Flow",
  "My Custom Flow (Custom)"
]
```

### GET /api/flows/list-with-details

List all flows with detailed information.

**Response:**

```json
[
  {
    "name": "Advanced QA Generation Flow",
    "id": "small-rock-799",
    "path": "/path/to/flow.yaml",
    "description": "Generates question-answer pairs from documents",
    "tags": ["question-generation", "qa-pairs"]
  }
]
```

### POST /api/flows/search

Search flows by tag or name.

**Request:**

```json
{
  "tag": "question-generation",
  "name_filter": "qa"
}
```

**Response:**

```json
[
  "QA Generation Flow",
  "Advanced QA Flow"
]
```

### GET /api/flows/{flow_name}/info

Get detailed flow information.

**Parameters:**

- `flow_name` (path) — Flow name (URL encoded, supports nested paths)

**Response:**

```json
{
  "name": "Advanced QA Generation Flow",
  "id": "small-rock-799",
  "path": "/path/to/flow.yaml",
  "description": "Generates question-answer pairs from documents",
  "version": "1.0.0",
  "author": "SDG Hub Team",
  "tags": ["question-generation", "qa-pairs"],
  "recommended_models": {
    "default": "meta-llama/Llama-3.3-70B-Instruct",
    "alternatives": ["gpt-4o"]
  },
  "dataset_requirements": {
    "required_columns": ["document", "domain"],
    "optional_columns": ["outline"],
    "min_samples": 1
  }
}
```

### POST /api/flows/{flow_name}/select

Select a flow for configuration.

**Response:**

```json
{
  "status": "success",
  "message": "Flow 'QA Generation' selected successfully",
  "flow_info": { ... }
}
```

### POST /api/flows/select-by-path

Select a flow by its file system path.

**Request:**

```json
{
  "flow_path": "/path/to/flow.yaml"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Flow selected successfully",
  "flow_info": { ... }
}
```

### GET /api/flows/{flow_name}/yaml

Get raw flow YAML content.

**Response:**

```json
{
  "metadata": {
    "name": "Flow Name",
    "version": "1.0.0"
  },
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": { ... }
    }
  ]
}
```

### POST /api/flows/create

Create a new flow from block definitions.

**Request:**

```json
{
  "metadata": {
    "name": "My New Flow",
    "description": "Custom pipeline"
  },
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": { ... }
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "flow_name": "My New Flow"
}
```

### POST /api/flows/save-custom

Save a custom flow with metadata and blocks.

**Request:**

```json
{
  "metadata": {
    "name": "My Custom Flow",
    "description": "Custom pipeline",
    "version": "1.0.0",
    "tags": ["custom"],
    "required_columns": ["document"]
  },
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": {
        "block_name": "generate",
        "input_cols": ["prompt"],
        "output_cols": ["response"]
      }
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "flow_name": "My Custom Flow",
  "flow_path": "/path/to/custom_flows/my_custom_flow/flow.yaml"
}
```

### GET /api/flows/templates

Get available flow templates.

**Response:**

```json
[
  {
    "name": "QA Generation Template",
    "description": "Template for generating Q&A pairs",
    "blocks": [ ... ]
  }
]
```

---

## Model Configuration

### GET /api/model/recommendations

Get model recommendations for selected flow.

**Response:**

```json
{
  "default": "meta-llama/Llama-3.3-70B-Instruct",
  "alternatives": ["gpt-4o", "claude-3-opus"],
  "notes": "Best results with 70B+ parameter models"
}
```

### POST /api/model/configure

Configure model settings.

**Request:**

```json
{
  "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
  "api_base": "http://localhost:8000/v1",
  "api_key": "env:OPENAI_API_KEY",
  "additional_params": {
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Model configuration applied"
}
```

### POST /api/model/test

Test model connection and configuration.

**Request:**

```json
{
  "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
  "api_base": "http://localhost:8000/v1",
  "api_key": "env:OPENAI_API_KEY",
  "test_prompt": "Hello, can you respond?"
}
```

**Response:**

```json
{
  "success": true,
  "response": "Hello! Yes, I can respond...",
  "latency_ms": 234,
  "error": null
}
```

---

## Dataset Management

### POST /api/dataset/upload

Upload a dataset file.

**Request:** `multipart/form-data`

- `file` — Dataset file (JSONL, JSON, CSV, Parquet)

**Response:**

```json
{
  "status": "success",
  "filename": "seed_data.jsonl",
  "path": "uploads/seed_data.jsonl",
  "rows": 100,
  "columns": ["document", "domain"]
}
```

### POST /api/dataset/load

Load dataset from file.

**Request:**

```json
{
  "data_files": "uploads/seed_data.jsonl",
  "file_format": "auto",
  "num_samples": 100,
  "shuffle": true,
  "seed": 42,
  "csv_delimiter": ",",
  "csv_encoding": "utf-8",
  "added_columns": {
    "extra_col": "default_value"
  }
}
```

**Response:**

```json
{
  "status": "success",
  "rows": 100,
  "columns": ["document", "domain"],
  "preview": [
    {"document": "...", "domain": "science"}
  ]
}
```

### GET /api/dataset/schema

Get required schema for selected flow.

**Response:**

```json
{
  "required_columns": ["document", "domain"],
  "optional_columns": ["outline"],
  "description": "Input data requirements"
}
```

### GET /api/dataset/preview

Get preview of loaded dataset.

**Response:**

```json
{
  "rows": 100,
  "columns": ["document", "domain"],
  "preview": [
    {"document": "...", "domain": "science"},
    {"document": "...", "domain": "history"}
  ]
}
```

### GET /api/dataset/check-duplicates

Check for duplicate rows in the loaded dataset.

**Response:**

```json
{
  "has_duplicates": true,
  "duplicate_count": 5,
  "total_rows": 100
}
```

### POST /api/dataset/remove-duplicates

Remove duplicate rows from the loaded dataset.

**Response:**

```json
{
  "status": "success",
  "rows_removed": 5,
  "remaining_rows": 95
}
```

---

## PDF Preprocessing

### POST /api/preprocessing/upload-pdf

Upload PDF or Markdown files for preprocessing.

**Request:** `multipart/form-data`

- `files` — One or more PDF/MD files

**Response:**

```json
{
  "job_id": "abc123",
  "files": ["document1.pdf", "document2.pdf"],
  "status": "uploaded"
}
```

### POST /api/preprocessing/convert/{job_id}

Convert uploaded PDFs to Markdown using Docling.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:**

```json
{
  "status": "success",
  "job_id": "abc123",
  "converted_files": ["document1.md", "document2.md"]
}
```

### GET /api/preprocessing/markdown-content/{job_id}/{filename}

Get the Markdown content of a converted file.

**Parameters:**

- `job_id` (path) — Preprocessing job ID
- `filename` (path) — Converted filename

**Response:**

```json
{
  "content": "# Document Title\n\nConverted markdown content..."
}
```

### GET /api/preprocessing/pdf/{job_id}/{filename}

Serve an uploaded PDF file for viewing.

**Parameters:**

- `job_id` (path) — Preprocessing job ID
- `filename` (path) — PDF filename

**Response:** PDF file (binary)

### GET /api/preprocessing/download/{job_id}/{filename}

Download a converted file.

**Parameters:**

- `job_id` (path) — Preprocessing job ID
- `filename` (path) — Filename to download

**Response:** File download

### POST /api/preprocessing/chunk/{job_id}

Chunk converted Markdown files into segments.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Request:**

```json
{
  "chunk_size": 1000,
  "overlap": 200,
  "method": "recursive",
  "selected_files": ["document1.md"],
  "file_configs": {
    "document1.md": {
      "chunk_size": 500,
      "overlap": 100
    }
  }
}
```

**Response:**

```json
{
  "status": "success",
  "total_chunks": 42,
  "chunks_per_file": {
    "document1.md": 42
  }
}
```

### GET /api/preprocessing/chunks/{job_id}

Get chunk information for a preprocessing job.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:**

```json
{
  "total_chunks": 42,
  "chunks": [
    {"file": "document1.md", "index": 0, "content": "..."}
  ]
}
```

### POST /api/preprocessing/create-dataset/{job_id}

Create a JSONL dataset from preprocessed and chunked data.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Request:**

```json
{
  "job_id": "abc123",
  "chunk_config": {
    "chunk_size": 1000,
    "overlap": 200
  },
  "additional_columns": {},
  "icl_template": {
    "icl_document": "Example document...",
    "icl_query_1": "Example question 1?",
    "icl_response_1": "Example answer 1.",
    "icl_query_2": "Example question 2?",
    "icl_response_2": "Example answer 2.",
    "icl_query_3": "Example question 3?",
    "icl_response_3": "Example answer 3."
  },
  "domain": "science",
  "document_outline": "Overview of the document structure",
  "dataset_name": "my_dataset",
  "content_column_name": "document",
  "include_domain": true,
  "include_document_outline": true
}
```

**Response:**

```json
{
  "status": "success",
  "dataset_path": "uploads/my_dataset.jsonl",
  "rows": 42,
  "columns": ["document", "domain", "document_outline", "icl_document", ...]
}
```

### GET /api/preprocessing/jobs

List all preprocessing jobs.

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "abc123",
      "status": "completed",
      "files": ["document1.pdf"],
      "created_at": "2024-11-27T10:00:00"
    }
  ]
}
```

### GET /api/preprocessing/status/{job_id}

Get status of a preprocessing job.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:**

```json
{
  "job_id": "abc123",
  "status": "completed",
  "files": ["document1.pdf"],
  "converted_files": ["document1.md"]
}
```

### GET /api/preprocessing/datasets

List all datasets created from preprocessing.

**Response:**

```json
[
  {
    "job_id": "abc123",
    "dataset_name": "my_dataset",
    "path": "uploads/my_dataset.jsonl",
    "rows": 42
  }
]
```

### GET /api/preprocessing/datasets/{job_id}/download

Download a preprocessed dataset.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:** File download (JSONL)

### DELETE /api/preprocessing/datasets/{job_id}

Delete a preprocessed dataset.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:**

```json
{
  "status": "success",
  "message": "Dataset deleted"
}
```

### DELETE /api/preprocessing/{job_id}

Delete a preprocessing job and all associated files.

**Parameters:**

- `job_id` (path) — Preprocessing job ID

**Response:**

```json
{
  "status": "success",
  "message": "Preprocessing job deleted"
}
```

### GET /api/preprocessing/icl-templates

Get available ICL (In-Context Learning) templates.

**Response:**

```json
{
  "templates": [
    {
      "name": "QA Template",
      "icl_document": "...",
      "icl_query_1": "...",
      "icl_response_1": "..."
    }
  ]
}
```

---

## Flow Execution

### GET /api/flow/dry-run-stream

Execute dry run with streaming output (SSE).

**Query Parameters:**

- `sample_size` — Number of samples (default: 2)
- `enable_time_estimation` — Estimate full run time (default: true)
- `max_concurrency` — Parallel requests (default: 10)

**Response:** Server-Sent Events (SSE)

```
data: {"type": "log", "message": "Starting dry run..."}
data: {"type": "log", "message": "Processing sample 1/2..."}
data: {"type": "complete", "execution_time_seconds": 5.2, "samples_processed": 2}
```

### POST /api/flow/dry-run

Execute dry run (non-streaming).

**Request:**

```json
{
  "sample_size": 2,
  "enable_time_estimation": true,
  "max_concurrency": 10
}
```

**Response:**

```json
{
  "status": "success",
  "execution_time_seconds": 12.5,
  "samples_processed": 2,
  "output_columns": ["question", "answer", "score"],
  "estimated_full_time_seconds": 625
}
```

### POST /api/flow/cancel-dry-run

Cancel a running dry run.

**Response:**

```json
{
  "status": "success",
  "message": "Dry run cancelled"
}
```

### GET /api/flow/dry-run-status

Check the status of a running dry run.

**Response:**

```json
{
  "running": true,
  "start_time": "2024-11-27T14:30:00",
  "samples_processed": 1
}
```

### POST /api/flow/test-step-by-step

Execute a step-by-step test of a flow's blocks (SSE).

**Request:**

```json
{
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": { ... }
    }
  ],
  "model_config_data": {
    "model": "hosted_vllm/...",
    "api_base": "http://localhost:8000/v1",
    "api_key": "env:API_KEY"
  },
  "workspace_id": "workspace_abc123"
}
```

**Response:** Server-Sent Events (SSE)

```
data: {"type": "block_start", "block_index": 0, "block_name": "generate"}
data: {"type": "block_output", "block_index": 0, "output": {...}}
data: {"type": "block_complete", "block_index": 0}
data: {"type": "complete"}
```

### GET /api/flow/generate-stream

Start generation with streaming output.

**Query Parameters:**

- `config_id` — Configuration ID
- `max_concurrency` — Parallel requests (default: 10)
- `resume` — Resume from checkpoint (true/false)

**Response:** Server-Sent Events (SSE)

```
data: {"type": "log", "message": "Starting generation..."}
data: {"type": "log", "message": "Executing block 1/4..."}
data: {"type": "complete", "num_samples": 100, "num_columns": 42, "output_file": "output.jsonl"}
```

### GET /api/flow/generation-status

Check generation status.

**Query Parameters:**

- `config_id` — Configuration ID (optional)

**Response:**

```json
{
  "running_generations": [
    {
      "config_id": "abc-123",
      "start_time": "2024-11-27T14:30:00",
      "samples_processed": 50
    }
  ]
}
```

### GET /api/flow/reconnect-stream

Reconnect to a running generation stream.

**Query Parameters:**

- `config_id` — Configuration ID

**Response:** Server-Sent Events (SSE)

### POST /api/flow/cancel-generation

Cancel running generation.

**Query Parameters:**

- `config_id` — Configuration to cancel (optional, cancels all if not specified)

**Response:**

```json
{
  "status": "success",
  "message": "Generation cancelled"
}
```

### GET /api/flow/download-generated

Download generated output file.

**Query Parameters:**

- `config_id` — Configuration ID

**Response:** File download (JSONL)

---

## Checkpoint Management

### GET /api/flow/checkpoints/{config_id}

Get checkpoint information.

**Response:**

```json
{
  "has_checkpoints": true,
  "checkpoint_count": 3,
  "samples_completed": 75,
  "last_checkpoint_time": "2024-11-27T14:35:00",
  "checkpoint_dir": "/path/to/checkpoints/abc-123"
}
```

### DELETE /api/flow/checkpoints/{config_id}

Clear checkpoints for configuration.

**Response:**

```json
{
  "status": "success",
  "message": "Checkpoints cleared"
}
```

---

## Workspace (Live Flow Editing)

### POST /api/workspace/create

Create a new workspace for editing a flow.

**Request:**

```json
{
  "source_flow_name": "Advanced QA Generation Flow"
}
```

**Response:**

```json
{
  "workspace_id": "workspace_abc123",
  "status": "created"
}
```

### POST /api/workspace/{workspace_id}/update-flow

Update the flow definition in a workspace.

**Parameters:**

- `workspace_id` (path) — Workspace ID

**Request:**

```json
{
  "metadata": {
    "name": "Updated Flow",
    "description": "Modified pipeline"
  },
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": { ... }
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Flow updated"
}
```

### POST /api/workspace/{workspace_id}/update-prompt

Update a prompt template in the workspace.

**Parameters:**

- `workspace_id` (path) — Workspace ID

**Request:**

```json
{
  "prompt_filename": "qa_prompt.yaml",
  "prompt_config": {
    "system": "You are a helpful assistant...",
    "user": "Given the document:\n{{ document }}\n\nGenerate questions."
  }
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Prompt updated"
}
```

### GET /api/workspace/{workspace_id}/blocks

Get the blocks in a workspace flow.

**Parameters:**

- `workspace_id` (path) — Workspace ID

**Response:**

```json
{
  "blocks": [
    {
      "block_type": "ChatCompletionBlock",
      "block_config": { ... }
    }
  ]
}
```

### POST /api/workspace/{workspace_id}/finalize

Finalize a workspace and save as a permanent custom flow.

**Parameters:**

- `workspace_id` (path) — Workspace ID

**Request:**

```json
{
  "flow_name": "My Finalized Flow"
}
```

**Response:**

```json
{
  "status": "success",
  "flow_name": "My Finalized Flow",
  "flow_path": "/path/to/custom_flows/my_finalized_flow/flow.yaml"
}
```

### DELETE /api/workspace/{workspace_id}

Delete a workspace.

**Parameters:**

- `workspace_id` (path) — Workspace ID

**Response:**

```json
{
  "status": "success",
  "message": "Workspace deleted"
}
```

---

## Configuration Management

### GET /api/config/current

Get the current active configuration state.

**Response:**

```json
{
  "flow_name": "QA Generation",
  "flow_path": "/path/to/flow.yaml",
  "model_config": { ... },
  "dataset_config": { ... }
}
```

### POST /api/config/reset

Reset the current configuration state.

**Response:**

```json
{
  "status": "success",
  "message": "Configuration reset"
}
```

### POST /api/config/import

Import a configuration from a file.

**Request:** `multipart/form-data`

- `file` — Configuration file (JSON)

**Response:**

```json
{
  "status": "success",
  "configuration": { ... }
}
```

---

## Saved Configurations

### GET /api/configurations/list

List all saved configurations.

**Response:**

```json
{
  "configurations": [
    {
      "id": "abc-123",
      "flow_name": "QA Generation",
      "flow_id": "small-rock-799",
      "flow_path": "/path/to/flow.yaml",
      "model_configuration": {
        "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
        "api_base": "http://localhost:8000/v1"
      },
      "dataset_configuration": {
        "data_files": "uploads/seed_data.jsonl",
        "num_samples": 100
      },
      "dry_run_configuration": {
        "sample_size": 2,
        "enable_time_estimation": true
      },
      "status": "configured",
      "tags": ["qa", "knowledge"],
      "created_at": "2024-11-27T10:00:00",
      "updated_at": "2024-11-27T10:30:00"
    }
  ]
}
```

### GET /api/configurations/{config_id}

Get specific configuration.

**Response:**

```json
{
  "id": "abc-123",
  "flow_name": "QA Generation",
  ...
}
```

### POST /api/configurations/save

Save a configuration.

**Request:**

```json
{
  "flow_name": "QA Generation",
  "flow_id": "small-rock-799",
  "flow_path": "/path/to/flow.yaml",
  "model_configuration": {
    "model": "hosted_vllm/...",
    "api_base": "http://localhost:8000/v1",
    "api_key": "env:API_KEY"
  },
  "dataset_configuration": {
    "data_files": "uploads/data.jsonl",
    "num_samples": 100,
    "shuffle": true
  },
  "dry_run_configuration": {
    "sample_size": 2,
    "enable_time_estimation": true
  },
  "tags": ["knowledge"],
  "status": "configured"
}
```

**Response:**

```json
{
  "status": "success",
  "configuration": { ... },
  "warning": "API key was not saved"
}
```

### DELETE /api/configurations/{config_id}

Delete a configuration.

**Response:**

```json
{
  "status": "success",
  "message": "Configuration deleted"
}
```

### POST /api/configurations/{config_id}/load

Load configuration into current context.

**Response:**

```json
{
  "status": "success",
  "message": "Configuration loaded"
}
```

---

## Run History

### GET /api/runs/list

List all run records.

**Response:**

```json
{
  "runs": [
    {
      "run_id": "run_abc123_1732789012",
      "config_id": "abc-123",
      "flow_name": "QA Generation",
      "flow_type": "existing",
      "model_name": "llama-70B",
      "status": "completed",
      "start_time": "2024-11-27T14:30:00",
      "end_time": "2024-11-27T14:35:23",
      "duration_seconds": 323,
      "input_samples": 100,
      "output_samples": 100,
      "output_columns": 42,
      "dataset_file": "uploads/data.jsonl",
      "output_file": "outputs/qa_generation_20241127.jsonl"
    }
  ]
}
```

### GET /api/runs/{run_id}

Get specific run details.

### POST /api/runs/create

Create run record.

**Request:**

```json
{
  "run_id": "run_abc123_1732789012",
  "config_id": "abc-123",
  "flow_name": "QA Generation",
  "flow_type": "existing",
  "model_name": "llama-70B",
  "status": "running",
  "start_time": "2024-11-27T14:30:00",
  "input_samples": 100,
  "dataset_file": "uploads/data.jsonl"
}
```

### PUT /api/runs/{run_id}/update

Update run record.

**Request:**

```json
{
  "status": "completed",
  "end_time": "2024-11-27T14:35:23",
  "duration_seconds": 323,
  "output_samples": 100,
  "output_columns": 42,
  "output_file": "outputs/output.jsonl"
}
```

### DELETE /api/runs/{run_id}

Delete run record and output file.

### GET /api/runs/config/{config_id}

Get all runs associated with a specific configuration.

**Parameters:**

- `config_id` (path) — Configuration ID

**Response:**

```json
{
  "runs": [
    {
      "run_id": "run_abc123_1732789012",
      "status": "completed",
      ...
    }
  ]
}
```

### POST /api/runs/{run_id}/analyze-logs

Analyze generation logs for a run.

**Request:**

```json
{
  "raw_logs": "Starting flow 'QA Generation'...\nBlock 1/4: generate_summary...\nError: Connection refused..."
}
```

**Response:**

```json
{
  "analysis": "The run failed due to a connection error...",
  "suggestions": ["Check model server is running", "Verify API base URL"]
}
```

### GET /api/runs/{run_id}/preview

Preview the output data of a run.

**Parameters:**

- `run_id` (path) — Run ID

**Response:**

```json
{
  "preview": [
    {"document": "...", "question": "...", "answer": "..."},
    {"document": "...", "question": "...", "answer": "..."}
  ],
  "total_rows": 100,
  "columns": ["document", "question", "answer", "score"]
}
```

### GET /api/runs/{run_id}/download

Download run output file.

**Response:** File download (JSONL)

---

## Custom Flows

### GET /api/custom-flows

List all custom flows.

**Response:**

```json
[
  {
    "flow_name": "my_custom_flow",
    "files": ["flow.yaml", "qa_prompt.yaml"],
    "created_at": "2024-11-27T10:00:00"
  }
]
```

### GET /api/custom-flows/{flow_name}/download/{filename}

Download a specific file from a custom flow.

**Parameters:**

- `flow_name` (path) — Custom flow name
- `filename` (path) — File to download

**Response:** File download (YAML)

### GET /api/custom-flows/{flow_name}/download-all

Download all files for a custom flow as a ZIP archive.

**Parameters:**

- `flow_name` (path) — Custom flow name

**Response:** ZIP file download

### DELETE /api/custom-flows/{flow_name}

Delete a specific custom flow.

**Parameters:**

- `flow_name` (path) — Custom flow name

**Response:**

```json
{
  "status": "success",
  "message": "Custom flow deleted"
}
```

### DELETE /api/custom-flows

Delete all custom flows.

**Response:**

```json
{
  "status": "success",
  "message": "All custom flows deleted"
}
```

---

## Block Registry

### GET /api/blocks/list

List all available blocks.

**Response:**

```json
{
  "blocks": [
    {
      "name": "ChatCompletionBlock",
      "category": "llm",
      "description": "LLM chat completion",
      "input_cols": ["prompt"],
      "output_cols": ["response"]
    },
    {
      "name": "ColumnMapperBlock",
      "category": "transform",
      "description": "Map/rename columns"
    }
  ]
}
```

### GET /api/blocks/templates

Get block templates with pre-configured settings.

**Response:**

```json
[
  {
    "name": "ChatCompletionBlock",
    "template_config": {
      "input_cols": ["prompt"],
      "output_cols": ["response"]
    }
  }
]
```

---

## Prompts

### POST /api/prompts/save

Save a prompt template.

**Request:**

```json
{
  "flow_name": "my_flow",
  "prompt_name": "qa_prompt",
  "content": "You are a helpful assistant...\n\n{{ document }}"
}
```

**Response:**

```json
{
  "status": "success",
  "path": "custom_flows/my_flow/qa_prompt.yaml"
}
```

### GET /api/prompts/load

Load a prompt template.

**Query Parameters:**

- `prompt_path` — Path to prompt file

**Response:**

```json
{
  "content": "You are a helpful assistant...",
  "variables": ["document", "query"]
}
```

---

## Static Files

### GET /uploads/{filename}

Serve an uploaded file.

**Parameters:**

- `filename` (path) — Name of the uploaded file

**Response:** File content

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Server error |

## Rate Limits

No rate limits on the API server. Rate limits may apply from upstream LLM providers.

## WebSocket / SSE Notes

The following endpoints use Server-Sent Events (SSE) for real-time updates:

- `/api/flow/dry-run-stream` — Dry run progress
- `/api/flow/generate-stream` — Generation progress
- `/api/flow/reconnect-stream` — Reconnect to running generation
- `/api/flow/test-step-by-step` — Step-by-step flow testing

Ensure your client properly handles:

- Connection keepalive
- Automatic reconnection
- Event parsing

Example client code:

```javascript
const eventSource = new EventSource('/api/flow/generate-stream?config_id=abc');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.message);
};

eventSource.onerror = (error) => {
  console.error('Connection error:', error);
  eventSource.close();
};
```

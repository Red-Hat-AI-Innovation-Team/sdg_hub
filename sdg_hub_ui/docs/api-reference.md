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

- `flow_name` (path) — Flow name (URL encoded)

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

### POST /api/flows/save-custom

Save a custom flow.

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
  "csv_encoding": "utf-8"
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

---

## Flow Execution

### POST /api/flow/dry-run

Execute dry run with small sample.

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

### GET /api/flow/generate-stream

Start generation with streaming output.

**Query Parameters:**

- `config_id` — Configuration ID
- `max_concurrency` — Parallel requests (default: 10)
- `resume` — Resume from checkpoint (true/false)

**Response:** Server-Sent Events (SSE)

```
event: message
data: {"type": "log", "message": "Starting generation..."}

event: message
data: {"type": "log", "message": "Executing block 1/4..."}

event: message
data: {"type": "complete", "num_samples": 100, "num_columns": 42, "output_file": "output.jsonl"}
```

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

Reconnect to running generation stream.

**Query Parameters:**

- `config_id` — Configuration ID

**Response:** Server-Sent Events (SSE)

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

## Configuration Management

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
      "status": "configured",
      "created_at": "2024-11-27T10:00:00",
      "tags": ["qa", "knowledge"]
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

### GET /api/runs/{run_id}/download

Download run output file.

**Response:** File download (JSONL)

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

The `/api/flow/generate-stream` and `/api/flow/reconnect-stream` endpoints use Server-Sent Events (SSE) for real-time updates. Ensure your client properly handles:

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


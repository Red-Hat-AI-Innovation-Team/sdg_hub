# Architecture

Technical architecture and design documentation for SDG Hub UI.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Browser                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Port 3000)                  │ │
│  │  ┌────────────┬────────────┬────────────┬───────────────────┐ │ │
│  │  │   Home     │  Dashboard │  Wizard    │  Flow Builder     │ │ │
│  │  │  Dashboard │  (Stats)   │  (Config)  │  (Visual Editor)  │ │ │
│  │  ├────────────┼────────────┼────────────┼───────────────────┤ │ │
│  │  │  Flows     │  Run       │  Live      │  Multi-Flow       │ │ │
│  │  │  Page      │  History   │  Monitor   │  Monitor          │ │ │
│  │  └────────────┴────────────┴────────────┴───────────────────┘ │ │
│  │                           │                                    │ │
│  │                    API Service Layer                           │ │
│  │                      (api.js)                                  │ │
│  └──────────────────────────│─────────────────────────────────────┘ │
└──────────────────────────────│───────────────────────────────────────┘
                               │ HTTP/SSE
┌──────────────────────────────│───────────────────────────────────────┐
│                FastAPI Backend (Port 8000)                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    api_server.py                               │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┬─────────────┐ │ │
│  │  │ Flow API │Config API│Execute   │ Runs API │Preprocess   │ │ │
│  │  │          │          │API       │          │API          │ │ │
│  │  ├──────────┼──────────┼──────────┼──────────┼─────────────┤ │ │
│  │  │Model API │Dataset   │Workspace │Custom    │Blocks/      │ │ │
│  │  │          │API       │API       │Flows API │Prompts API  │ │ │
│  │  └──────────┴──────────┴──────────┴──────────┴─────────────┘ │ │
│  └────────────────────────────│───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────│───────────────────────────────────┐ │
│  │                    SDG Hub Core                                │ │
│  │  ┌──────────────┬──────────────┬──────────────┐               │ │
│  │  │ FlowRegistry │ BlockRegistry│ Flow Engine  │               │ │
│  │  └──────────────┴──────────────┴──────────────┘               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Docling (PDF Processing)                    │ │
│  │  ┌──────────────┬──────────────┬──────────────┐               │ │
│  │  │ PDF Parser   │ MD Converter │ Chunking     │               │ │
│  │  └──────────────┴──────────────┴──────────────┘               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Structure

The frontend has been refactored with useReducer patterns, custom hooks, context providers, and component extraction:

```
frontend/src/
├── App.js                          # Main app, routing, global state
├── index.js                        # Entry point
├── index.css                       # Global styles, PatternFly imports
│
├── components/
│   ├── AppHeader.js                # Navigation header with sidebar toggle
│   ├── HomeDashboard.js            # Welcome page, quick start, prebuilt flows
│   ├── Dashboard.js                # Stats, flows overview, preprocessed datasets
│   ├── DataGenerationFlowsPage.js  # Flow configurations list and management
│   ├── FlowRunsHistoryPage.js      # Run history view with filters
│   ├── UnifiedFlowWizard.js        # Multi-step wizard (useReducer for wizard state)
│   ├── LiveMonitoring.js           # Real-time progress tracking
│   ├── MultiFlowMonitoringModal.js # Side-by-side multi-flow monitoring
│   ├── SimpleFlowCanvas.js         # Extracted flow canvas (React.memo optimized)
│   ├── MissingColumnsModal.js      # Extracted modal for missing column resolution
│   ├── DuplicatesModal.js          # Extracted modal for duplicate handling
│   │
│   ├── configurations/             # Configuration management components
│   │   ├── ConfigurationList.js    # Config list with actions and batch ops
│   │   ├── ConfigurationTable.js   # Data table with status and progress
│   │   └── ConfigurationDetailView.js # Expanded config view with tabs
│   │
│   ├── steps/                      # Wizard steps
│   │   ├── FlowSelectionStep.js    # Select flow from library
│   │   ├── ModelConfigurationStep.js # Model API settings
│   │   ├── DatasetConfigurationStep.js # Dataset config (useReducer for form state)
│   │   ├── PDFPreprocessingStep.js # PDF processing (split into 6 sub-components)
│   │   ├── ICLConfigurationStep.js # In-Context Learning template config
│   │   ├── DryRunSettingsStep.js   # Dry run configuration
│   │   ├── DryRunStep.js           # Dry run execution and results
│   │   ├── ReviewStep.js           # Final review before save
│   │   └── OverviewStep.js         # Summary view
│   │
│   └── flowCreator/                # Flow builder components
│       ├── FlowBuilderPage.js      # Main builder layout
│       ├── BlockLibrary.js         # Block picker sidebar
│       ├── BundlesCard.js          # Pre-configured block bundles
│       ├── BlockConfigModal.js     # Block configuration modal
│       ├── MetadataFormModal.js    # Flow metadata form
│       ├── PromptEditorModal.js    # Prompt template editor
│       ├── bundleDefinitions.js    # Bundle configurations
│       │
│       ├── TestRunner/             # Step-by-step flow testing
│       │   ├── index.js            # Test runner entry
│       │   ├── TestConfigModal.js  # Test configuration
│       │   ├── TestSampleRunner.js # Sample execution
│       │   └── NodeIOModal.js      # Node input/output preview
│       │
│       └── VisualFlowEditor/       # Node-based visual editor
│           ├── index.js            # Editor entry
│           ├── VisualFlowEditor.js # Main canvas (useReducer for editor state)
│           ├── ConnectionValidator.js # Connection rules
│           ├── constants.js        # Editor constants
│           ├── FlowSerializer.js   # Serialize/deserialize flows
│           ├── GuidedTour.js       # Onboarding tour
│           ├── NodeConfigDrawer.js # Node configuration drawer
│           ├── NodeSidebar.js      # Node palette sidebar
│           └── nodes/              # Node type components
│               ├── index.js        # Node type registry
│               ├── LLMNode.js      # LLM/Chat completion node
│               ├── ParserNode.js   # Text parser node
│               ├── TransformNode.js # Data transformation node
│               └── EvalNode.js     # Evaluation node
│
├── contexts/
│   ├── NotificationContext.js      # Global toast notifications
│   └── ExecutionContext.js         # Execution state (eliminates prop drilling)
│
├── hooks/
│   └── usePDFProcessing.js        # PDF processing logic (useReducer-based)
│
└── services/
    └── api.js                      # API client (Axios-based)
```

#### Key Frontend Refactoring Patterns

- **`usePDFProcessing` custom hook** — Extracts all PDF preprocessing logic from PDFPreprocessingStep into a dedicated hook using useReducer for state management
- **PDFPreprocessingStep sub-components** — The monolithic step is split into 6 focused sub-components for upload, conversion, preview, chunking, ICL configuration, and dataset creation
- **SimpleFlowCanvas** — Extracted from inline rendering, wrapped with React.memo to prevent unnecessary re-renders
- **useReducer groups** — Complex state in UnifiedFlowWizard, DatasetConfigurationStep, and VisualFlowEditor consolidated from multiple useState calls into useReducer
- **ExecutionContext** — Provides execution state (active generations, status updates) to deeply nested components without prop drilling
- **MissingColumnsModal and DuplicatesModal** — Extracted from DatasetConfigurationStep into standalone modal components

### Backend Structure

The backend has been refactored from a monolithic `api_server.py` (~6,800 lines) into a modular architecture:

```
backend/
├── api_server.py                   # Slim entry point (~150 lines)
│                                   # App setup, CORS middleware, router registration
├── config.py                       # Path constants, settings, environment config
├── state.py                        # Shared mutable state (active generations,
│                                   # preprocessing jobs, execution queues)
│
├── models/                         # Pydantic request/response models (8 files, 20 models)
│   ├── config.py                   # Configuration models
│   ├── dataset.py                  # Dataset models
│   ├── execution.py                # Execution/generation models
│   ├── flow.py                     # Flow models
│   ├── model.py                    # Model configuration models
│   ├── preprocessing.py            # PDF preprocessing models
│   ├── workspace.py                # Workspace models
│   └── __init__.py
│
├── utils/                          # Utility modules (7 modules)
│   ├── security.py                 # Path traversal protection, filename sanitization
│   ├── file_handling.py            # File I/O, directory management
│   ├── dataset_utils.py            # Dataset loading, schema detection, preview
│   ├── api_key_utils.py            # API key resolution, env: syntax handling
│   ├── flow_utils.py               # Flow discovery, template loading
│   ├── config_utils.py             # Configuration persistence
│   └── __init__.py
│
├── workers/                        # Background workers
│   ├── dry_run_worker.py           # Dry run subprocess execution
│   ├── generation_worker.py        # Full generation subprocess execution
│   └── __init__.py
│
├── routers/                        # FastAPI APIRouter modules (11 routers, 80 endpoints)
│   ├── health.py                   # GET /health
│   ├── flows.py                    # /api/flows/* — Flow discovery and templates
│   ├── model.py                    # /api/model/* — Model configuration
│   ├── dataset.py                  # /api/dataset/* — Dataset management
│   ├── preprocessing.py            # /api/preprocessing/* — PDF preprocessing
│   ├── execution.py                # /api/flow/* — Dry run, generation, SSE streams
│   ├── workspace.py                # /api/workspace/* — Live flow editing
│   ├── configurations.py           # /api/configurations/* — Configuration CRUD
│   ├── runs.py                     # /api/runs/* — Run history
│   ├── custom_flows.py             # /api/custom-flows/* — Custom flow management
│   └── blocks.py                   # /api/blocks/*, /api/prompts/* — Blocks and prompts
│
├── requirements.txt                # Python dependencies
└── start_api_with_restart.sh       # Auto-restart script
```

### Data Storage

```
backend/  (or $SDG_HUB_DATA_DIR)
├── uploads/                        # Uploaded datasets
│   └── *.jsonl, *.csv, *.parquet
├── outputs/                        # Generated outputs
│   └── {flow}_{timestamp}.jsonl
├── custom_flows/                   # Custom flow definitions
│   └── {flow_name}/
│       ├── flow.yaml
│       └── {prompt}.yaml
├── checkpoints/                    # Generation checkpoints
│   └── {config_id}/
│       ├── checkpoint_0001.jsonl
│       └── flow_metadata.json
├── pdf_uploads/                    # Raw PDF/MD uploads
│   └── {job_id}/
│       └── *.pdf, *.md
├── pdf_converted/                  # Converted Markdown files
│   └── {job_id}/
│       └── *.md
├── saved_configurations.json       # Saved flow configurations
├── preprocessing_jobs.json         # PDF preprocessing job records
└── runs_history.json               # Run history records
```

## Data Flow

### Flow Configuration

```
User Input → Wizard Component → API Service → Backend → SDG Hub Core
     ↑                                              │
     └──────────────── Response ←───────────────────┘
```

### PDF Preprocessing Pipeline

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Upload  │────▶│ Convert  │────▶│  Chunk   │────▶│ Create   │
│  PDF/MD  │     │ to MD    │     │ Text     │     │ Dataset  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                  (Docling)        (langchain       (JSONL with
                                   splitters)        ICL templates)
```

### Flow Execution

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│  SDG Hub    │
│  (React)    │     │  (FastAPI)  │     │  (Python)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │  EventSource      │  Subprocess       │  flow.generate()
       │  Connection       │  + Queue          │
       │◀──────────────────│◀──────────────────│
       │   SSE Stream      │   Log Queue       │   Logs/Results
```

### Workspace (Live Flow Editing)

```
Create Workspace → Edit Flow/Prompts → Test Step-by-Step → Finalize
       │                  │                    │                │
       ▼                  ▼                    ▼                ▼
  Copy source flow   Update blocks     Execute blocks     Save as
  to temp workspace  and prompts       one at a time      custom flow
```

### State Management

```
┌─────────────────────────────────────────────────────────────┐
│                     App.js (Global State)                    │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ executionStates  │  │ Navigation State │                │
│  │ (per config)     │  │ (activeItem)     │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                           │
│  ┌────────▼─────────┐  ┌───────▼────────┐                  │
│  │ localStorage     │  │ sessionStorage │                  │
│  │ (persist state)  │  │ (persist nav)  │                  │
│  └──────────────────┘  └────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               UnifiedFlowWizard (Session State)              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Wizard Progress  │  │ Draft State      │                │
│  │ (sessionStorage) │  │ (localStorage)   │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## API Service Layer

The frontend uses a single API module (`src/services/api.js`) with specialized sub-modules:

| API Module | Purpose |
|------------|---------|
| `flowAPI` | List flows, flow info, search, select, save custom |
| `modelAPI` | Model recommendations, configure, test connection |
| `datasetAPI` | Upload, load, schema, preview, duplicates |
| `executionAPI` | Dry run, cancel, generation status, reconnect |
| `flowTestAPI` | Step-by-step flow testing with SSE |
| `workspaceAPI` | Create/update/delete workspaces, update prompts |
| `customFlowsAPI` | List, download, delete custom flows |
| `checkpointAPI` | Get/clear checkpoints |
| `configAPI` | Get/reset/import config |
| `savedConfigAPI` | CRUD saved configurations |
| `blockAPI` | List blocks and templates |
| `runsAPI` | Run CRUD, download, preview, log analysis |
| `promptAPI` | Save/load prompt templates |
| `preprocessingAPI` | PDF upload, convert, chunk, create dataset, jobs |

**Configuration:**

- Base URL resolves via `REACT_APP_API_URL` or dynamically maps frontend port (300X) to backend port (800X)
- `package.json` proxy: `http://localhost:8000`
- Axios instance with shared config and error interceptor

## Key Design Patterns

### Page Routing

Routing is handled via `activeItem` state in `App.js` (no React Router):

| Route ID | Component | Purpose |
|----------|-----------|---------|
| `home` | HomeDashboard | Welcome page, quick start |
| `dashboard` | Dashboard | Stats, flows overview |
| `flows` | DataGenerationFlowsPage | Flow configurations |
| `configure-flow` | UnifiedFlowWizard | Configuration wizard |
| `flow-runs` | FlowRunsHistoryPage | Run history |

### Wizard Pattern

The configuration wizard uses a step-based pattern with:

- **Linear progression** with back navigation
- **Step validation** before advancing
- **State persistence** across steps via sessionStorage
- **Dynamic step rendering** based on selections (e.g., PDF preprocessing, ICL steps)
- **Draft auto-save** to localStorage

### Configuration State

Configurations progress through defined states:

```
draft → not_configured → configured → running → completed/failed/cancelled
```

State transitions are tracked in both frontend (executionStates in localStorage) and backend (saved_configurations.json).

### Streaming with SSE

Long-running operations use Server-Sent Events:

- **Dry Run** (`/api/flow/dry-run-stream`) — streaming dry run progress
- **Generation** (`/api/flow/generate-stream`) — streaming generation logs
- **Step-by-Step Test** (`/api/flow/test-step-by-step`) — streaming block-by-block test results
- **Reconnect** (`/api/flow/reconnect-stream`) — reconnect to running generation after page refresh

```python
# Backend
async def generate_stream():
    for log in log_queue:
        yield f"data: {json.dumps(log)}\n\n"
```

```javascript
// Frontend
const eventSource = new EventSource(url);
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateState(data);
};
```

### Checkpoint System

Generation progress is checkpointed:

1. **Save frequency** — Every N samples or on completion
2. **Checkpoint format** — JSONL with metadata
3. **Resume logic** — Skip completed samples, continue from last
4. **Per-config isolation** — Checkpoints stored in `checkpoints/{config_id}/`

### Workspace System

The workspace pattern enables live flow editing:

1. **Create** — Copies source flow to a temporary workspace directory
2. **Edit** — Modify flow blocks, prompts, and metadata in isolation
3. **Test** — Execute step-by-step testing against the workspace flow
4. **Finalize** — Save the workspace as a permanent custom flow

## Performance Optimizations

### Frontend

- **Memoization** — React.useMemo for expensive computations
- **Debouncing** — Search input debounced
- **State persistence** — Avoid refetching on navigation via sessionStorage/localStorage
- **Reconnection** — Reconnect to running generations after page refresh

### Backend

- **Async endpoints** — FastAPI async handlers
- **Process isolation** — Generation runs in subprocess with multiprocessing
- **Efficient data loading** — Pandas for large datasets
- **In-memory caching** — Active generations, configurations, and preprocessing jobs cached in memory
- **File-based storage** — JSON files for persistence (no database dependency)

### Generation

- **Concurrent requests** — Parallel LLM calls via `max_concurrency`
- **Checkpointing** — Resume without reprocessing
- **Streaming** — Real-time feedback via SSE, no polling

## Security

- **Path traversal protection** — `ensure_within_directory()` and `detect_path_traversal()` checks
- **Filename sanitization** — `sanitize_filename()` for uploads
- **API key masking** — Keys are not stored in saved configurations
- **Environment variable resolution** — `env:VAR_NAME` syntax for API keys
- **CORS** — Restricted to localhost origins (ports 3000-3005)
- **Whitelisted directories** — Flow, prompt, and data paths are restricted to allowed directories

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 18 | UI framework |
| UI Components | PatternFly 5 | Design system (core, icons, table, drag-drop, topology) |
| Frontend Patterns | useReducer, React Context, custom hooks, React.memo | State management and optimization |
| HTTP Client | Axios | API requests |
| Backend | FastAPI (11 APIRouter modules) | Modular REST API |
| Validation | Pydantic (20 models) | Request/response models |
| Server | Uvicorn | ASGI server |
| Core | SDG Hub | Generation engine |
| PDF Processing | Docling | PDF to Markdown conversion |
| Text Splitting | langchain-text-splitters | Document chunking |
| Data | Pandas | Dataset processing |

## Extension Points

### Adding New Blocks

1. Create block in SDG Hub core
2. Register with BlockRegistry
3. Add to bundleDefinitions.js for bundles
4. Add a new node type in `VisualFlowEditor/nodes/` if needed
5. Update BlockConfigModal if needed

### Adding New Node Types

1. Create a new node component in `flowCreator/VisualFlowEditor/nodes/`
2. Register in the node `index.js` registry
3. Add to the `NodeSidebar.js` palette
4. Update `ConnectionValidator.js` with connection rules
5. Update `FlowSerializer.js` for serialization support

### Adding New Endpoints

1. Add endpoint in the appropriate router module under `routers/`
2. Create Pydantic models in `models/` for request/response
3. Add utility functions in `utils/` if needed
4. Add API method in the appropriate sub-module in api.js
5. Update components to use new endpoint

### Adding New Wizard Steps

1. Create step component in `components/steps/`
2. Register in `UnifiedFlowWizard.js` step configuration
3. Add step validation logic
4. Update the wizard navigation flow

### Custom UI Components

1. Create component in components/
2. Import PatternFly components
3. Add to relevant page/wizard
4. Connect to API service

## Testing Strategy

### Frontend Tests

```
tests/frontend/
├── components/
│   ├── App.test.js
│   ├── ConfigurationTable.test.js
│   └── NotificationContext.test.js
├── api.test.js
└── setupTests.js
```

### Backend Tests

```
tests/backend/
├── test_flow_endpoints.py
├── test_configuration_endpoints.py
├── test_dataset_endpoints.py
├── test_checkpoint_endpoints.py
├── test_run_history_endpoints.py
├── test_block_endpoints.py
├── test_model_endpoints.py
├── test_health_endpoint.py
└── test_security_utils.py
```

### Running Tests

```bash
# Frontend
cd tests/frontend
npm test

# Backend
cd tests/backend
pytest
```

## Configuration

### Environment Variables

```bash
# Backend
SDG_HUB_DATA_DIR=          # Isolated data directory (relative to backend/)
SDG_HUB_MAX_UPLOAD_MB=512  # Max file upload size in MB
SDG_HUB_ALLOWED_DATA_DIRS= # Additional data paths (OS path-separated)

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

### API Keys (.env)

Copy `.env.example` to `backend/.env` and configure:

```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
# ... additional provider keys
```

Reference in the UI using `env:VARIABLE_NAME` syntax.

> **Note:** SDG Hub UI is designed for local use only. All services run on localhost.

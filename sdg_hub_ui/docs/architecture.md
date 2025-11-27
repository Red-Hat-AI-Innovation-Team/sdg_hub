# Architecture

Technical architecture and design documentation for SDG Hub UI.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Browser                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Port 3000)                  │ │
│  │  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │ │
│  │  │   App.js     │  Wizard      │  Flow Builder │ Monitoring  │ │ │
│  │  │   (Router)   │  (Config)    │  (Builder)    │ (Live)      │ │ │
│  │  └──────────────┴──────────────┴──────────────┴─────────────┘ │ │
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
│  │  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │ │
│  │  │ Flow API     │ Config API   │ Execute API  │ Runs API    │ │ │
│  │  └──────────────┴──────────────┴──────────────┴─────────────┘ │ │
│  └────────────────────────────│───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────│───────────────────────────────────┐ │
│  │                    SDG Hub Core                                │ │
│  │  ┌──────────────┬──────────────┬──────────────┐               │ │
│  │  │ FlowRegistry │ BlockRegistry│ Flow Engine  │               │ │
│  │  └──────────────┴──────────────┴──────────────┘               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Structure

```
frontend/src/
├── App.js                          # Main app, routing, global state
├── index.js                        # Entry point
├── index.css                       # Global styles
│
├── components/
│   ├── AppHeader.js                # Navigation header
│   ├── DataGenerationFlowsPage.js  # Main dashboard
│   ├── FlowRunsHistoryPage.js      # Run history view
│   ├── UnifiedFlowWizard.js        # Configuration wizard
│   ├── LiveMonitoring.js           # Real-time progress
│   ├── MultiFlowMonitoringModal.js # Multi-flow monitoring
│   │
│   ├── configurations/
│   │   ├── ConfigurationList.js    # Config table with actions
│   │   ├── ConfigurationTable.js   # Data table component
│   │   └── ConfigurationDetailView.js # Expanded config view
│   │
│   ├── steps/                      # Wizard steps
│   │   ├── FlowSelectionStep.js    # Step 2a: Select flow
│   │   ├── ModelConfigurationStep.js # Step 3: Model config
│   │   ├── DatasetConfigurationStep.js # Step 4: Dataset
│   │   ├── DryRunSettingsStep.js   # Step 5: Dry run
│   │   ├── DryRunStep.js           # Dry run execution
│   │   ├── ReviewStep.js           # Step 6: Review
│   │   └── OverviewStep.js         # Summary view
│   │
│   └── flowCreator/                # Flow builder components
│       ├── FlowBuilderPage.js      # Main builder interface
│       ├── BlockLibrary.js         # Available blocks list
│       ├── BundlesCard.js          # Block bundles
│       ├── BlockConfigModal.js     # Block configuration
│       ├── MetadataFormModal.js    # Flow metadata form
│       ├── PromptEditorModal.js    # Prompt template editor
│       └── bundleDefinitions.js    # Bundle configurations
│
├── contexts/
│   └── NotificationContext.js      # Global notifications
│
└── services/
    └── api.js                      # API client
```

### Backend Structure

```
backend/
├── api_server.py                   # FastAPI application
│   ├── Flow Discovery Endpoints    # /api/flows/*
│   ├── Model Configuration         # /api/model/*
│   ├── Dataset Management          # /api/dataset/*
│   ├── Flow Execution              # /api/flow/*
│   ├── Configuration CRUD          # /api/configurations/*
│   ├── Run History                 # /api/runs/*
│   ├── Block Registry              # /api/blocks/*
│   └── Prompt Management           # /api/prompts/*
│
├── requirements.txt                # Python dependencies
└── start_api_with_restart.sh       # Auto-restart script
```

### Data Storage

```
backend/
├── uploads/                        # Uploaded datasets
│   └── *.jsonl
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
├── saved_configurations.json       # Saved configs
└── runs_history.json               # Run records
```

## Data Flow

### Flow Configuration

```
User Input → Wizard Component → API Service → Backend → SDG Hub Core
     ↑                                              │
     └──────────────── Response ←───────────────────┘
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

## Key Design Patterns

### Wizard Pattern

The configuration wizard uses a step-based pattern with:

- **Linear progression** with back navigation
- **Step validation** before advancing
- **State persistence** across steps
- **Dynamic step rendering** based on selections

```jsx
<Wizard>
  <WizardStep id="source-selection" name="Choose Source">
    {/* Step 1 content */}
  </WizardStep>
  <WizardStep id="select-existing" name="Select Flow">
    {/* Step 2 content - conditional */}
  </WizardStep>
  {/* ... more steps */}
</Wizard>
```

### Configuration State

Configurations progress through defined states:

```
draft → not_configured → configured → running → completed/failed/cancelled
```

State transitions are tracked in both frontend and backend.

### Streaming with SSE

Long-running operations use Server-Sent Events:

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

```python
# In generation worker
if checkpoint_dir:
    generate_kwargs['checkpoint_dir'] = checkpoint_dir
    generate_kwargs['save_freq'] = save_freq
```

## Performance Optimizations

### Frontend

- **Lazy loading** — Components loaded on demand
- **Memoization** — React.useMemo for expensive computations
- **Debouncing** — Search input debounced
- **State persistence** — Avoid refetching on navigation

### Backend

- **Async endpoints** — FastAPI async handlers
- **Process isolation** — Generation in subprocess
- **Efficient data loading** — Pandas for large datasets
- **Connection pooling** — Reused HTTP connections

### Generation

- **Concurrent requests** — Parallel LLM calls
- **Checkpointing** — Resume without reprocessing
- **Streaming** — Real-time feedback, no polling

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 18 | UI framework |
| UI Components | PatternFly 5 | Design system |
| HTTP Client | Axios | API requests |
| Backend | FastAPI | REST API |
| Validation | Pydantic | Request/response models |
| Server | Uvicorn | ASGI server |
| Core | SDG Hub | Generation engine |
| Data | Pandas | Dataset processing |

## Extension Points

### Adding New Blocks

1. Create block in SDG Hub core
2. Register with BlockRegistry
3. Add to bundleDefinitions.js for bundles
4. Update BlockConfigModal if needed

### Adding New Endpoints

1. Add endpoint in api_server.py
2. Create Pydantic models for request/response
3. Add API method in api.js
4. Update components to use new endpoint

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
SDG_HUB_DATA_DIR=          # Isolated data directory
SDG_HUB_MAX_UPLOAD_MB=512  # Max file upload
SDG_HUB_ALLOWED_DATA_DIRS= # Additional data paths

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

> **Note:** SDG Hub UI is designed for local use only. All services run on localhost.

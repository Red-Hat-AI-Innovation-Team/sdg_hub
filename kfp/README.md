# SDG Hub KFP Component

Kubeflow Pipelines component for running SDG Hub flows in ML pipelines.

## Local Development Setup

### Prerequisites

- Python 3.10+
- Podman
- `uv` package manager

### 1. Install Podman

```bash
# macOS
brew install podman

# Initialize and start (first time)
podman machine init
podman machine start

# Verify
podman machine list
```

### 2. Install Python Dependencies

```bash
uv pip install ".[kfp]" docker pip kfp-kubernetes
```

Note: `pip` is required because KFP's SubprocessRunner uses `python -m pip` internally.

### 3. Build the Component Image

```bash
podman build -t sdg-hub-kfp:dev -f kfp/Dockerfile .

# Verify
podman images | grep sdg-hub-kfp
```

### 4. Run Tests

```bash
# Unit tests (29 tests)
uv run python -m pytest tests/kfp/ -v

# E2E test with real transform flow
uv run python kfp/test_e2e.py

# Clean up outputs
rm -rf local_outputs/
```

### 5. Compile Pipeline

```bash
# Compile pipeline definition to YAML
uv run python kfp/pipeline.py

# Output: kfp/pipeline.yaml (uploadable to a KFP instance)
```

## Development Workflow

1. Edit code in `src/sdg_hub/kfp/`
2. Run unit tests: `uv run python -m pytest tests/kfp/ -v`
3. Run E2E test: `uv run python kfp/test_e2e.py`
4. Clean up: `rm -rf local_outputs/`

## File Structure

```
kfp/
├── ARCHITECTURE.md                    # Component design documentation
├── README.md                          # This file
├── PROGRESS.md                        # Development progress tracking
├── Dockerfile                         # Container image definition
├── pipeline.py                        # Sample pipeline definition
├── test_local_runner.py               # Hello world test
├── test_sdg_skeleton.py               # SubprocessRunner test
├── test_e2e.py                        # E2E test with real flow
└── testdata/
    ├── sample_input.jsonl             # Test input data (3 rows)
    └── transform_test_flow.yaml       # Transform-only test flow

src/sdg_hub/kfp/
├── __init__.py                        # Package exports
└── component.py                       # Main SDG component

tests/kfp/
├── __init__.py                        # Test package
└── test_component.py                  # Unit + E2E tests (29 tests)
```

## Architecture Note

Full Kubeflow Pipelines on Kubernetes (Kind) does not work on Apple Silicon due to x86/ARM64 architecture incompatibility. The KFP Local Runner approach runs pipelines directly via Podman without needing a Kubernetes cluster.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component design details.

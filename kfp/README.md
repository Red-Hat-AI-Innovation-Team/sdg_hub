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
uv pip install ".[kfp]" docker
```

### 3. Build the Component Image

```bash
podman build -t sdg-hub-kfp:dev -f kfp/Dockerfile .

# Verify
podman images | grep sdg-hub-kfp
```

### 4. Run Tests

```bash
# Test hello world component
python kfp/test_local_runner.py

# Test SDG skeleton component
python kfp/test_sdg_skeleton.py

# Clean up outputs
rm -rf local_outputs/
```

## Development Workflow

1. Edit code in `src/sdg_hub/kfp/`
2. Rebuild image: `podman build -t sdg-hub-kfp:dev -f kfp/Dockerfile .`
3. Run test: `python kfp/test_sdg_skeleton.py`

## File Structure

```
kfp/
├── ARCHITECTURE.md        # Component design documentation
├── README.md              # This file
├── Dockerfile             # Container image definition
├── test_local_runner.py   # Hello world test
└── test_sdg_skeleton.py   # SDG component test

src/sdg_hub/kfp/
├── __init__.py            # Package exports
└── component.py           # Main SDG component
```

## Architecture Note

Full Kubeflow Pipelines on Kubernetes (Kind) does not work on Apple Silicon due to x86/ARM64 architecture incompatibility. The KFP Local Runner approach runs pipelines directly via Podman without needing a Kubernetes cluster.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component design details.

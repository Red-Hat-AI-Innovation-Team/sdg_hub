# Installation

SDG Hub requires Python 3.9+ and can be installed via pip or from source for development.

## 📦 Production Installation

### Basic Installation

```bash
pip install sdg-hub
```

### With UV (Recommended)

```bash
# Install SDG Hub
uv pip install sdg-hub

# Or create a new project with SDG Hub
uv init my-sdg-project
cd my-sdg-project
uv add sdg-hub
```

## 🔧 Optional Dependencies

SDG Hub supports optional feature sets that can be installed as needed:

### vLLM Support
For high-performance local LLM inference:

```bash
# With pip
pip install sdg-hub[vllm]

# With uv
uv pip install sdg-hub[vllm]
```

### Examples Dependencies
For running example notebooks and workflows:

```bash
# With pip
pip install sdg-hub[examples]

# With uv  
uv pip install sdg-hub[examples]
```

### All Optional Dependencies
To install everything at once:

```bash
# With pip
pip install sdg-hub[vllm,examples]

# With uv
uv pip install sdg-hub[vllm,examples]
```

## 🛠️ Development Installation

For contributing to SDG Hub or customizing the framework:

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub

# Install in development mode with all dependencies
uv pip install .[dev]

# Alternative: use uv sync for lock file management
uv sync --extra dev
```

### Development Dependencies

The `[dev]` extra includes:
- Testing frameworks (pytest, tox)
- Linting tools (pylint, ruff, mypy)
- Documentation tools
- Pre-commit hooks

### Verify Installation

```bash
# Run tests to verify everything works
uv run pytest tests/

# Check code quality
make verify

# Run a quick lint check
tox -e fastlint
```

## 🔍 Verification

After installation, verify SDG Hub is working correctly:

```python
# Test basic imports
from sdg_hub.core.flow import FlowRegistry
from sdg_hub.core.blocks import BlockRegistry

# Discover available components
FlowRegistry.discover_flows()
BlockRegistry.discover_blocks()

print("✅ SDG Hub installed successfully!")
print(f"Available flows: {len(FlowRegistry.list_flows())}")
print(f"Available blocks: {len(BlockRegistry.list_blocks())}")
```

## 🚀 Next Steps

Now that SDG Hub is installed, check out the [Quick Start Guide](quick-start.md) to build your first synthetic data pipeline!
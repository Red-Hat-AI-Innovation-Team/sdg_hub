# Installation

## System Requirements

- Python 3.8 or higher
- pip package manager

## Installation Methods

### Stable Release (Recommended)

Install the latest stable version from PyPI:

```bash
pip install sdg-hub
```

### Development Version

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
```

### Development Installation

For contributors and developers who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub

# Install in development mode
pip install -e .[dev]
```

## Optional Dependencies

### Web Interface

To use the web interface for flow visualization and management:

```bash
pip install -e .[web_interface]
```

### Examples Dependencies

To run the provided examples:

```bash
pip install -e .[examples]
```

### All Dependencies

To install all optional dependencies:

```bash
pip install -e .[dev,web_interface,examples]
```

## Verification

Verify your installation by running:

```python
import sdg_hub
print(sdg_hub.__version__)
```

You should see the version number printed without any errors.

## Troubleshooting

### Common Issues

1. **Permission denied errors**: Use `pip install --user` to install packages in your user directory
2. **Version conflicts**: Create a virtual environment to isolate dependencies
3. **Missing dependencies**: Ensure you have the latest pip version: `pip install --upgrade pip`

### Virtual Environment Setup

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv sdg_hub_env

# Activate virtual environment
# On macOS/Linux:
source sdg_hub_env/bin/activate
# On Windows:
sdg_hub_env\Scripts\activate

# Install SDG Hub
pip install sdg-hub
```
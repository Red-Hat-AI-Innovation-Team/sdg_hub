# Web Interface

SDG Hub includes a web-based interface for managing and executing flows through a user-friendly GUI.

## Overview

The web interface provides:

- **Flow Visualization**: View and understand flow configurations
- **Interactive Execution**: Run flows with real-time progress monitoring
- **Parameter Configuration**: Easily adjust flow parameters
- **Results Management**: View and download generated data

## Installation

Install the web interface dependencies:

```bash
pip install -e .[web_interface]
```

## Running the Web Interface

### Quick Start

```bash
cd web_interface
python app.py
```

The interface will be available at `http://localhost:5000`

### Configuration

The web interface can be configured through environment variables:

```bash
# Set custom port
export FLASK_PORT=8080

# Enable debug mode
export FLASK_DEBUG=1

# Run the application
python app.py
```

## Features

### Flow Management

- **Flow Selection**: Choose from available built-in flows
- **Custom Flows**: Upload and use your own YAML flow configurations
- **Flow Validation**: Automatic validation of flow syntax and structure

### Parameter Configuration

Configure flow execution parameters through the web interface:

- **Data Sources**: Select input datasets
- **Output Settings**: Configure save paths and formats
- **Processing Options**: Set batch sizes, worker counts, and checkpointing
- **Model Configuration**: Configure LLM endpoints and parameters

### Execution Monitoring

- **Real-time Progress**: Live updates on flow execution progress  
- **Logs Viewer**: View detailed execution logs
- **Error Handling**: Clear error messages and troubleshooting guidance
- **Resource Monitoring**: Track CPU, memory, and API usage

### Results Management

- **Output Preview**: Preview generated data before download
- **Export Options**: Multiple export formats (JSON, CSV, JSONL)
- **Checkpoint Management**: Resume interrupted flows
- **History Tracking**: View previous flow executions

## File Structure

```
web_interface/
├── app.py              # Main Flask application
├── static/
│   ├── css/
│   │   └── style.css   # Web interface styles
│   └── js/
│       └── app.js      # Frontend JavaScript
├── templates/
│   └── index.html      # Main HTML template
└── test_block_types.py # Block type testing utilities
```

## API Endpoints

The web interface exposes several REST API endpoints:

### Flow Management

```http
GET /api/flows
```
List available flows

```http
POST /api/flows/validate
Content-Type: application/json

{
  "flow_config": "yaml_content_here"
}
```
Validate flow configuration

### Execution

```http
POST /api/execute
Content-Type: application/json

{
  "flow_path": "path/to/flow.yaml",
  "ds_path": "path/to/data.json",
  "save_path": "path/to/output.json",
  "endpoint": "https://api.openai.com/v1",
  "batch_size": 8,
  "num_workers": 16
}
```
Execute a flow

```http
GET /api/status/{execution_id}
```
Get execution status

### Results

```http
GET /api/results/{execution_id}
```
Download execution results

```http
GET /api/logs/{execution_id}
```
View execution logs

## Custom Integration

### Embedding in Your Application

You can embed the web interface in your own Flask application:

```python
from web_interface.app import app as sdg_app
from flask import Flask

# Your main application
main_app = Flask(__name__)

# Register SDG Hub blueprint
main_app.register_blueprint(sdg_app, url_prefix='/sdg')

if __name__ == '__main__':
    main_app.run()
```

### Custom Block Types

Test custom block types using the included utilities:

```python
from web_interface.test_block_types import test_custom_block

# Test your custom block
result = test_custom_block(
    block_class=YourCustomBlock,
    test_data=sample_dataset
)
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   
   # Kill the process or use a different port
   export FLASK_PORT=8080
   ```

2. **Missing Dependencies**
   ```bash
   pip install -e .[web_interface]
   ```

3. **File Upload Issues**
   - Ensure proper file permissions
   - Check file size limits
   - Verify file format compatibility

### Debug Mode

Enable debug mode for detailed error messages:

```bash
export FLASK_DEBUG=1
python app.py
```

## Security Considerations

- The web interface is designed for local development use
- For production deployment, consider:
  - Adding authentication
  - Using a production WSGI server (e.g., Gunicorn)
  - Implementing rate limiting
  - Securing file upload functionality

## Browser Compatibility

The web interface supports:
- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+
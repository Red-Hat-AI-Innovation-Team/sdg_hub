# SDG Flow Builder Web Interface

A web-based interface for building and managing SDG flows using a drag-and-drop interface.

## Features

- Drag-and-drop block creation
- Visual block reordering with up/down buttons
- Block configuration through a modal interface
- YAML generation and parsing
- Support for multiple block types (LLM, Filter, Iteration, etc.)

## Block Types

### LLM Block
- Used for language model operations
- Configurable model ID, output columns, and generation parameters
- Supports batch processing and duplicate handling

### Filter Block
- Filters data based on column values
- Supports various operations (equals, greater than, contains)
- Can convert data types and drop columns

### Iteration Block
- Handles iterative operations
- Configurable number of iterations
- Specifies input field for iteration

### Retrieval Model Block
- Manages retrieval operations
- Configurable model and query field
- Supports top-k retrieval

### Utility Block
- General-purpose operations
- Supports map, reduce, and filter operations
- Configurable input field

## Usage

1. Start the Flask server:
```bash
cd web_interface
python app.py
```

2. Access the interface at http://127.0.0.1:8080

3. Build your flow:
   - Drag blocks from the left panel to the canvas
   - Use up/down buttons to reorder blocks
   - Click the gear icon to configure block parameters
   - Use the "Generate YAML" button to create YAML output

4. Save and load flows:
   - Use "Save Flow" to download the current flow as JSON
   - Use "Load YAML" to import a YAML flow

## Development

The interface is built using:
- Flask for the backend
- Bootstrap for styling
- Vanilla JavaScript for interactivity

## File Structure

- `app.py`: Main Flask application
- `templates/index.html`: Main interface template
- `static/`: Static assets (CSS, JS)
- `requirements.txt`: Python dependencies 
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

## Contributing: Modifying Block Types

The web interface uses a block-based system where each block type is defined in `app.py` within the `BLOCK_TYPES` dictionary. This guide explains how to add, modify, or remove block types.

### Block Type Structure

Each block type in `BLOCK_TYPES` follows this structure:
```python
'block_id': {  # Unique identifier for the block type
    'name': 'Display Name',  # Human-readable name
    'config': {  # Configuration schema
        'parameter_name': {
            'type': 'string|number|array|object',  # Data type
            'required': True|False,  # Whether parameter is required
            'default': value,  # Optional default value
            'enum': ['value1', 'value2'],  # Optional enum of allowed values
            'properties': {  # For object type, define nested properties
                'nested_param': {'type': 'string', ...}
            }
        }
    }
}
```

### Adding a New Block Type

1. Add a new entry to the `BLOCK_TYPES` dictionary in `app.py`:
```python
'new_block': {
    'name': 'New Block Type',
    'config': {
        'block_name': {'type': 'string', 'required': True},
        # Add your custom parameters here
        'custom_param': {'type': 'string', 'required': True},
        'optional_param': {'type': 'number', 'required': False, 'default': 0}
    }
}
```

### Modifying an Existing Block Type

1. To add a new parameter:
```python
'config': {
    'existing_param': {...},
    'new_param': {'type': 'string', 'required': False, 'default': 'value'}
}
```

2. To modify an existing parameter:
```python
'config': {
    'existing_param': {
        'type': 'string',
        'required': True,  # Changed from False
        'default': 'new_default'  # Added default value
    }
}
```

### Removing a Block Type

1. Delete the block type entry from `BLOCK_TYPES`
2. Update any documentation referencing the removed block type

### Best Practices

1. **Backward Compatibility**:
   - When modifying existing blocks, maintain backward compatibility
   - Use optional parameters with defaults for new features
   - Consider versioning if making breaking changes

2. **Validation**:
   - Always include required/optional status for parameters
   - Use enums for parameters with fixed values
   - Provide meaningful default values

3. **Documentation**:
   - Update this README when adding new block types
   - Document all parameters and their purposes
   - Include examples of common configurations

4. **Testing**:
   - Test the block with various configurations
   - Verify YAML generation and parsing
   - Test with different parameter combinations

### Example: Adding a Data Transform Block

```python
'transform': {
    'name': 'Data Transform Block',
    'config': {
        'block_name': {'type': 'string', 'required': True},
        'transform_type': {
            'type': 'string',
            'required': True,
            'enum': ['normalize', 'standardize', 'encode']
        },
        'input_columns': {'type': 'array', 'required': True},
        'output_columns': {'type': 'array', 'required': True},
        'transform_params': {'type': 'object', 'required': False},
        'batch_kwargs': {'type': 'object', 'required': False}
    }
}
```

### After Making Changes

1. Restart the Flask server to apply changes
2. Test the new/modified block in the web interface
3. Verify YAML generation and parsing
4. Update any existing flows that might be affected 
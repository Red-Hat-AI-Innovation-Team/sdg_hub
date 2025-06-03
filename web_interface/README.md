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

## Contributing: Modifying Block Types

The web interface now discovers block types **dynamically** from the Python block registry (`BlockRegistry`) and infers their configuration schemas from the `__init__` parameters of each block class. You **do not** need to manually edit a `BLOCK_TYPES` dictionary.

### How Block Types Are Discovered
- All block classes registered with `BlockRegistry` are automatically available in the web interface.
- The configuration schema for each block is generated from the parameters of its `__init__` method (excluding `self` and `block_name`).
- The parameter type hints and default values are used to build the schema shown in the UI.
- The special parameter `gen_kwargs` is only included if it is present in the block's `__init__`.

### Adding a New Block Type
1. **Create a new block class** in Python, and decorate it with `@BlockRegistry.register()`:
```python
from sdg_hub.registry import BlockRegistry

@BlockRegistry.register()
class MyCustomBlock(Block):
    def __init__(self, block_name: str, custom_param: str, optional_param: int = 0, gen_kwargs: dict = None):
        super().__init__(block_name)
        self.custom_param = custom_param
        self.optional_param = optional_param
        self.gen_kwargs = gen_kwargs or {}
    # ... implement block logic ...
```
2. **Restart the Flask server**. Your new block will appear in the web interface, and its configuration form will be generated from the constructor signature.

### Modifying an Existing Block Type
- To add, remove, or change parameters, simply update the `__init__` method of the block class.
- The web interface will automatically reflect these changes after a server restart.
- To add `gen_kwargs` support, add it as a parameter to the constructor.

### Removing a Block Type
- Remove or comment out the block class, or unregister it from `BlockRegistry`.
- The block will no longer appear in the web interface after a server restart.

### Best Practices
- Use type hints for all constructor parameters to ensure correct schema generation.
- Provide default values for optional parameters.
- Document your block classes and parameters in code.
- Test your block in the web interface and with YAML generation/parsing.

### Example: Adding a Data Transform Block
```python
from sdg_hub.registry import BlockRegistry

@BlockRegistry.register()
class TransformBlock(Block):
    def __init__(self, block_name: str, transform_type: str, input_columns: list, output_columns: list, transform_params: dict = None):
        super().__init__(block_name)
        self.transform_type = transform_type
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.transform_params = transform_params or {}
    # ... implement block logic ...
```

### After Making Changes
1. Restart the Flask server to apply changes
2. Test the new/modified block in the web interface
3. Verify YAML generation and parsing
4. Update any existing flows that might be affected 
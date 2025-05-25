"""
SDG Flow Builder Web Interface
-----------------------------
A Flask-based web interface for building and managing SDG flows.
Provides a drag-and-drop interface for creating and configuring flow blocks.
"""

from flask import Flask, render_template, jsonify, request
import yaml
import os
from sdg_hub.registry import BlockRegistry
import inspect
from typing import get_type_hints, get_origin, get_args, Any
from sdg_hub.blocks import *  # Ensure all blocks are registered

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Generate a random secret key

def get_block_types_from_registry():
    """
    Dynamically populate BLOCK_TYPES dictionary from the block registry.
    Each block type will have a configuration schema based on its class's __init__ parameters.
    """
    block_types = {}
    registry = BlockRegistry.get_registry()
    
    for block_name, block_class in registry.items():
        if block_name.lower() == 'block':
            continue  # Skip the base 'Block' type
        # Get the __init__ method and its parameters
        init_method = block_class.__init__
        signature = inspect.signature(init_method)
        type_hints = get_type_hints(init_method)
        param_names = list(signature.parameters.keys())
        # Start with base schema
        config_schema = {
            'block_name': {'type': 'string', 'required': True}
        }
        # Add parameters from __init__
        for param_name, param in signature.parameters.items():
            # Skip self, block_name, and client (already added or handled internally)
            if param_name in ['self', 'block_name', 'client']:
                continue
            # Get parameter type and default
            param_type = type_hints.get(param_name, Any)
            param_default = param.default if param.default is not inspect.Parameter.empty else None
            # Determine if parameter is required
            is_required = param.default is inspect.Parameter.empty and param.kind not in [
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ]
            # Map Python types to schema types
            type_mapping = {
                str: 'string',
                int: 'integer',
                float: 'number',
                bool: 'boolean',
                list: 'array',
                dict: 'object',
                Any: 'string'  # Default to string for unknown types
            }
            # Handle special cases
            if get_origin(param_type) is list:
                schema_type = 'array'
                items_type = get_args(param_type)[0] if get_args(param_type) else Any
                schema = {
                    'type': schema_type,
                    'required': is_required,
                    'items': {'type': type_mapping.get(items_type, 'string')}
                }
            elif get_origin(param_type) is dict:
                schema_type = 'object'
                schema = {
                    'type': schema_type,
                    'required': is_required,
                    'properties': {}  # Could be expanded with specific dict properties
                }
            else:
                schema_type = type_mapping.get(param_type, 'string')
                schema = {
                    'type': schema_type,
                    'required': is_required
                }
            # Add default value if present
            if param_default is not None:
                schema['default'] = param_default
            config_schema[param_name] = schema
        # Only add gen_kwargs if it is present in __init__
        if 'gen_kwargs' in param_names:
            config_schema['gen_kwargs'] = {
                'type': 'object',
                'required': False,
                'properties': {
                    'temperature': {'type': 'number', 'required': False, 'default': 0.7},
                    'max_tokens': {'type': 'integer', 'required': False, 'default': 2048},
                    'top_p': {'type': 'number', 'required': False, 'default': 1.0}
                }
            }
        # Store the original block name as the key
        block_types[block_name] = {
            'name': block_name,
            'config': config_schema
        }
    return block_types

# Initialize BLOCK_TYPES from registry
BLOCK_TYPES = get_block_types_from_registry()

@app.route('/')
def index():
    """Render the main interface page with block types."""
    try:
        return render_template('index.html', block_types=BLOCK_TYPES)
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/api/blocks', methods=['GET'])
def get_blocks():
    return jsonify(BLOCK_TYPES)

@app.route('/api/generate_yaml', methods=['POST'])
def generate_yaml():
    """
    Generate YAML from the flow configuration.
    
    Converts the JSON flow data into the correct YAML format for SDG.
    Each block is converted to the appropriate YAML structure with block_type and block_config.
    """
    try:
        flow_data = request.json
        if not flow_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert the flow data to the correct YAML format
        yaml_blocks = []
        
        # Process blocks in the order they appear in the flow
        for block in flow_data['blocks']:
            block_type = block['type']
            block_config = block['config']
            
            # Create the block entry in the correct format
            yaml_block = {
                'block_type': block_type,
                'block_config': {
                    'block_name': block_config.get('block_name', f"{block_type}_{len(yaml_blocks)}"),
                    **{k: v for k, v in block_config.items() if k != 'block_name'}
                }
            }
            
            # Add drop_duplicates if specified
            if 'drop_duplicates' in block_config:
                yaml_block['drop_duplicates'] = block_config['drop_duplicates']
            
            yaml_blocks.append(yaml_block)
        
        # Convert to YAML format
        yaml_config = yaml.dump(yaml_blocks, default_flow_style=False)
        return jsonify({'yaml': yaml_config})
    except Exception as e:
        print(f"Error generating YAML: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parse_yaml', methods=['POST'])
def parse_yaml():
    """
    Parse YAML into flow configuration.
    
    Converts YAML flow data into the format expected by the web interface.
    Validates block types and creates appropriate block configurations.
    """
    try:
        yaml_content = request.json.get('yaml', '')
        if not yaml_content:
            return jsonify({'error': 'No YAML content provided'}), 400
            
        yaml_blocks = yaml.safe_load(yaml_content)
        
        if not isinstance(yaml_blocks, list):
            return jsonify({'error': 'Invalid YAML format. Expected a list of blocks.'}), 400
        
        # Convert YAML blocks to canvas format
        blocks = []
        
        # First pass: create blocks
        for i, yaml_block in enumerate(yaml_blocks):
            block_type = yaml_block.get('block_type', '')
            # Find the matching block type in BLOCK_TYPES (case-insensitive)
            matching_block_type = next(
                (bt for bt in BLOCK_TYPES.keys() if bt.lower() == block_type.lower()),
                None
            )
            if not matching_block_type:
                return jsonify({'error': f'Unknown block type: {block_type}'}), 400
            
            block_config = yaml_block.get('block_config', {})
            
            # Create block with position
            block = {
                'id': i + 1,  # Use index as ID
                'type': matching_block_type,  # Use the original case from BLOCK_TYPES
                'name': block_config.get('block_name', f"{matching_block_type}_{i}"),
                'config': block_config,
                'position': {
                    'x': i * 250,  # Position blocks horizontally
                    'y': 100
                }
            }
            blocks.append(block)
        
        return jsonify({
            'blocks': blocks,
            'connections': [],
            'firstBlockId': blocks[0]['id'] if blocks else None
        })
    
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {str(e)}")
        return jsonify({'error': f'Invalid YAML: {str(e)}'}), 400
    except Exception as e:
        print(f"Error parsing YAML: {str(e)}")
        return jsonify({'error': f'Error parsing YAML: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080) 
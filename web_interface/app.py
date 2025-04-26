"""
SDG Flow Builder Web Interface
-----------------------------
A Flask-based web interface for building and managing SDG flows.
Provides a drag-and-drop interface for creating and configuring flow blocks.
"""

from flask import Flask, render_template, jsonify, request
import yaml
import os

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Generate a random secret key

# Define available block types and their configuration schemas
# Each block type has a name, configuration options, and connection rules
BLOCK_TYPES = {
    'llm': {
        'name': 'LLM Block',
        'config': {
            'block_name': {'type': 'string', 'required': True},
            'config_path': {'type': 'string', 'required': True},
            'model_id': {'type': 'string', 'required': True},
            'output_cols': {'type': 'array', 'required': True},
            'gen_kwargs': {
                'type': 'object',
                'required': False,
                'properties': {
                    'temperature': {'type': 'number', 'default': 0.7},
                    'max_tokens': {'type': 'number', 'default': 2048}
                }
            },
            'drop_duplicates': {'type': 'array', 'required': False},
            'batch_kwargs': {'type': 'object', 'required': False}
        },
        'connections': {
            'input': ['llm', 'filter', 'iter', 'rm', 'util'],
            'output': ['llm', 'filter', 'iter', 'util']
        }
    },
    'filter': {
        'name': 'Filter Block',
        'config': {
            'block_name': {'type': 'string', 'required': True},
            'filter_column': {'type': 'string', 'required': True},
            'filter_value': {'type': 'string', 'required': True},
            'operation': {'type': 'string', 'required': True},
            'convert_dtype': {'type': 'string', 'required': False},
            'batch_kwargs': {'type': 'object', 'required': False},
            'drop_columns': {'type': 'array', 'required': False}
        },
        'connections': {
            'input': ['llm', 'filter', 'iter', 'rm', 'util'],
            'output': ['llm', 'filter', 'iter', 'util']
        }
    },
    'iter': {
        'name': 'Iteration Block',
        'config': {
            'block_name': {'type': 'string', 'required': True},
            'iterations': {'type': 'number', 'required': True},
            'input_field': {'type': 'string', 'required': True}
        },
        'connections': {
            'input': ['llm', 'filter', 'iter', 'rm', 'util'],
            'output': ['llm', 'filter', 'iter', 'util']
        }
    },
    'rm': {
        'name': 'Retrieval Model Block',
        'config': {
            'block_name': {'type': 'string', 'required': True},
            'model': {'type': 'string', 'required': True},
            'query_field': {'type': 'string', 'required': True},
            'top_k': {'type': 'number', 'required': False, 'default': 5}
        },
        'connections': {
            'input': ['llm', 'filter', 'iter', 'rm', 'util'],
            'output': ['llm', 'filter', 'iter', 'util']
        }
    },
    'util': {
        'name': 'Utility Block',
        'config': {
            'block_name': {'type': 'string', 'required': True},
            'operation': {'type': 'string', 'required': True, 'enum': ['map', 'reduce', 'filter']},
            'input_field': {'type': 'string', 'required': True}
        },
        'connections': {
            'input': ['llm', 'filter', 'iter', 'rm', 'util'],
            'output': ['llm', 'filter', 'iter', 'util']
        }
    }
}

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
                'block_type': f"{block_type.upper()}Block",
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
            block_type = yaml_block.get('block_type', '').lower().replace('block', '')
            if block_type not in BLOCK_TYPES:
                return jsonify({'error': f'Unknown block type: {block_type}'}), 400
            
            block_config = yaml_block.get('block_config', {})
            
            # Create block with position
            block = {
                'id': i + 1,  # Use index as ID
                'type': block_type,
                'name': block_config.get('block_name', f"{block_type}_{i}"),
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
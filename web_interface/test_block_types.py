"""
Test script for get_block_types_from_registry function.
This script will print out the block types and their configurations
to verify that the schema is correctly populated from the block classes.
It will also check that the config keys match the __init__ parameters of each block class.
"""

from app import get_block_types_from_registry
from sdg_hub.blocks import BlockRegistry
import inspect
import json

def test_block_types():
    # Print the registry contents directly
    print("\nDirect Registry Contents (from sdg_hub.blocks):")
    print("=" * 80)
    registry = BlockRegistry.get_registry()
    print(f"Number of blocks in registry: {len(registry)}")
    print("\nRegistered blocks:")
    for name, block_class in registry.items():
        print(f"- {name}: {block_class.__name__}")
    
    # Get block types from registry function
    print("\nBlock Types from Registry Function:")
    print("=" * 80)
    block_types = get_block_types_from_registry()
    print(f"\nNumber of block types: {len(block_types)}")
    for block_name, block_info in block_types.items():
        print(f"\nBlock Type: {block_name}")
        print("-" * 40)
        print(f"Name: {block_info['name']}")
        print("\nConfiguration Schema:")
        print(json.dumps(block_info['config'], indent=2))
        print("=" * 80)

    # Print the entire BLOCK_TYPES dict as pretty JSON
    print("\nFull BLOCK_TYPES dictionary:")
    print(json.dumps(block_types, indent=2))

    # Get block types from registry function
    block_types = get_block_types_from_registry()
    registry = BlockRegistry.get_registry()

    print("\n=== Block Properties Consistency Check ===\n")
    all_ok = True
    for block_type, block_info in block_types.items():
        class_name = block_info['name']
        block_class = registry[class_name]
        # Get __init__ parameters (excluding self and block_name)
        sig = inspect.signature(block_class.__init__)
        param_names = [p for p in sig.parameters if p not in ('self', 'block_name')]
        # Get config keys (excluding block_name, always present)
        config_keys = [k for k in block_info['config'].keys() if k != 'block_name']
        # Check for missing or extra keys
        missing_in_config = [p for p in param_names if p not in config_keys]
        extra_in_config = [k for k in config_keys if k not in param_names]
        print(f"Block: {class_name}")
        print(f"  __init__ params: {param_names}")
        print(f"  config keys:    {config_keys}")
        if not missing_in_config and not extra_in_config:
            print("  ✅ Properties match!\n")
        else:
            all_ok = False
            if missing_in_config:
                print(f"  ❌ Missing in config: {missing_in_config}")
            if extra_in_config:
                print(f"  ❌ Extra in config:   {extra_in_config}")
            print()
    if all_ok:
        print("All block config schemas match their class __init__ parameters!\n")
    else:
        print("Some blocks have mismatches between config and __init__ parameters.\n")

if __name__ == "__main__":
    test_block_types() 
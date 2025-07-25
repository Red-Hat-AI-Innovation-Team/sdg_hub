#!/usr/bin/env python3
"""Test script for backward compatibility migration."""

# Standard
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Local
from sdg_hub.flow.base import Flow
from sdg_hub.flow.migration import FlowMigration

def test_old_format_detection():
    """Test detection of old vs new format flows."""
    print("Testing format detection...")
    
    # Test old format (list)
    old_config = [
        {"block_type": "LLMBlock", "block_config": {"block_name": "test"}}
    ]
    assert FlowMigration.is_old_format(old_config) == True
    print("✓ Old format (list) detected correctly")
    
    # Test new format (dict with metadata/blocks)
    new_config = {
        "metadata": {"name": "test"},
        "blocks": [{"block_type": "LLMBlock"}]
    }
    assert FlowMigration.is_old_format(new_config) == False
    print("✓ New format detected correctly")
    
    print("Format detection tests passed!\n")

def test_old_flow_loading():
    """Test loading an actual old flow file."""
    print("Testing old flow loading...")
    
    old_flow_path = "src/sdg_hub/flows/generation/skills/synth_grounded_skills.yaml"
    
    try:
        # This should work with backward compatibility
        flow = Flow.from_yaml(old_flow_path)
        print(f"✓ Successfully loaded old flow: {flow.metadata.name}")
        print(f"  - Version: {flow.metadata.version}")
        print(f"  - Blocks: {len(flow.blocks)}")
        print(f"  - Block types: {[block.__class__.__name__ for block in flow.blocks]}")
        
        # Check runtime params extraction
        if hasattr(flow, '_migrated_runtime_params'):
            print(f"  - Migrated runtime params: {len(flow._migrated_runtime_params)} blocks")
            for block_name, params in flow._migrated_runtime_params.items():
                print(f"    - {block_name}: {params}")
        
        # Test flow info
        info = flow.get_info()
        print(f"  - Total blocks: {info['total_blocks']}")
        
    except Exception as e:
        print(f"✗ Failed to load old flow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Old flow loading test passed!\n")
    return True

def test_new_flow_loading():
    """Test loading a new format flow file."""
    print("Testing new flow loading...")
    
    new_flow_path = "snyth_knowledge1.5_0.2.0.yaml"
    
    try:
        # This should also work
        flow = Flow.from_yaml(new_flow_path)
        print(f"✓ Successfully loaded new flow: {flow.metadata.name}")
        print(f"  - Version: {flow.metadata.version}")
        print(f"  - Blocks: {len(flow.blocks)}")
        print(f"  - Block types: {[block.__class__.__name__ for block in flow.blocks]}")
        
    except Exception as e:
        print(f"✗ Failed to load new flow: {e}")
        return False
    
    print("New flow loading test passed!\n")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("BACKWARD COMPATIBILITY MIGRATION TEST")
    print("=" * 50)
    
    try:
        test_old_format_detection()
        old_success = test_old_flow_loading()
        new_success = test_new_flow_loading()
        
        if old_success and new_success:
            print("🎉 All tests passed! Backward compatibility is working.")
        else:
            print("❌ Some tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
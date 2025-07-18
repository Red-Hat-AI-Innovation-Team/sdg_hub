"""Test suite for SamplePopulatorBlock functionality.

This module contains tests for the new SamplePopulatorBlock class, which populates datasets
with data from configuration files using the BaseBlock architecture.
"""

# Standard
import os
import tempfile

# Third Party
from datasets import Dataset
import pytest
import yaml

# First Party
from sdg_hub.blocks.loading import SamplePopulatorBlock
from sdg_hub.utils.error_handling import MissingColumnError, EmptyDatasetError


@pytest.fixture
def temp_config_files():
    """Create temporary config files for testing."""
    configs = {
        "coding": {
            "examples": ["def hello():\n    print('Hello')", "class Test:\n    pass"],
            "type": "programming",
            "difficulty": "beginner",
            "language": "python"
        },
        "writing": {
            "examples": ["Write a story", "Compose a poem"],
            "type": "composition",
            "difficulty": "intermediate", 
            "language": "english"
        },
        "math": {
            "examples": ["2 + 2 = ?", "Solve: x + 5 = 10"],
            "type": "calculation",
            "difficulty": "easy",
            "language": "mathematics"
        },
    }

    temp_dir = tempfile.mkdtemp()
    config_paths = []

    for name, content in configs.items():
        file_path = os.path.join(temp_dir, f"{name}.yaml")
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        config_paths.append(file_path)

    yield config_paths

    # Cleanup
    for path in config_paths:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "route": ["coding", "writing", "math"],
            "user_id": [1, 2, 3],
            "prompt": ["Help with code", "Write something", "Solve equation"]
        }
    )


def test_sample_populator_basic(temp_config_files, sample_dataset):
    """Test basic functionality of SamplePopulatorBlock."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )

    result = block.generate(sample_dataset)

    # Check basic structure
    assert len(result) == 3
    assert "route" in result.column_names
    assert "user_id" in result.column_names
    assert "prompt" in result.column_names
    assert "type" in result.column_names
    assert "examples" in result.column_names

    # Check data merging
    row0 = result[0]  # coding
    assert row0["route"] == "coding"
    assert row0["type"] == "programming"
    assert row0["language"] == "python"
    assert row0["user_id"] == 1  # Original data preserved

    row1 = result[1]  # writing
    assert row1["route"] == "writing"
    assert row1["type"] == "composition"
    assert row1["language"] == "english"

    row2 = result[2]  # math
    assert row2["route"] == "math"
    assert row2["type"] == "calculation"
    assert row2["language"] == "mathematics"


def test_sample_populator_with_postfix(temp_config_files):
    """Test SamplePopulatorBlock with postfix parameter."""
    # Create postfixed versions of config files
    temp_dir = os.path.dirname(temp_config_files[0])
    postfixed_paths = []
    
    for path in temp_config_files:
        base, ext = os.path.splitext(path)
        new_path = f"{base}_v2{ext}"
        with open(path, "r") as src, open(new_path, "w") as dst:
            dst.write(src.read())
        postfixed_paths.append(new_path)

    try:
        block = SamplePopulatorBlock(
            block_name="test_populator",
            input_cols="route",
            config_paths=temp_config_files,
            post_fix="v2",
        )
        
        dataset = Dataset.from_dict({"route": ["coding"], "user_id": [1]})
        result = block.generate(dataset)
        
        # Should have loaded from postfixed files
        assert result[0]["type"] == "programming"
        
    finally:
        # Cleanup postfixed files
        for path in postfixed_paths:
            if os.path.exists(path):
                os.remove(path)


def test_sample_populator_list_input_cols(temp_config_files):
    """Test with list input_cols."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols=["route"],  # List instead of string
        config_paths=temp_config_files,
    )
    
    assert block.input_cols == ["route"]
    assert block.column_name == "route"


def test_sample_populator_empty_dataset(temp_config_files):
    """Test with empty dataset."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    empty_dataset = Dataset.from_dict({"route": [], "user_id": []})
    
    # Should raise EmptyDatasetError via BaseBlock validation
    with pytest.raises(EmptyDatasetError):
        block(empty_dataset)  # Use __call__ to trigger validation


def test_sample_populator_duplicate_routes(temp_config_files):
    """Test with duplicate route values."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    dataset = Dataset.from_dict({
        "route": ["coding", "coding", "writing"],
        "user_id": [1, 2, 3]
    })
    
    result = block.generate(dataset)
    assert len(result) == 3
    
    # Both coding rows should have same config data
    assert result[0]["type"] == "programming"
    assert result[1]["type"] == "programming"
    assert result[2]["type"] == "composition"


def test_sample_populator_preserves_original_data(temp_config_files, sample_dataset):
    """Test that original dataset columns are preserved."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )

    result = block.generate(sample_dataset)
    
    # All original columns should be preserved
    for i, original_row in enumerate(sample_dataset):
        result_row = result[i]
        for key, value in original_row.items():
            assert result_row[key] == value


def test_sample_populator_schema_validation_warning(temp_config_files, caplog):
    """Test that schema inconsistencies trigger warnings."""
    # Create configs with different schemas
    temp_dir = os.path.dirname(temp_config_files[0])
    
    config1 = {"type": "programming", "language": "python"}
    config2 = {"type": "composition", "genre": "creative"}  # Different field
    
    config1_path = os.path.join(temp_dir, "test1.yaml")
    config2_path = os.path.join(temp_dir, "test2.yaml")
    
    with open(config1_path, "w") as f:
        yaml.dump(config1, f)
    with open(config2_path, "w") as f:
        yaml.dump(config2, f)
    
    try:
        SamplePopulatorBlock(
            block_name="test_validation",
            input_cols="route",
            config_paths=[config1_path, config2_path]
        )
        
        # Check that warning was logged
        assert "Schema inconsistencies detected" in caplog.text
        
    finally:
        os.remove(config1_path)
        os.remove(config2_path)


def test_sample_populator_missing_column(temp_config_files):
    """Test with missing input column in dataset."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    dataset = Dataset.from_dict({"wrong_col": ["coding"], "user_id": [1]})
    
    with pytest.raises(MissingColumnError):
        block(dataset)  # Use __call__ to trigger BaseBlock validation


def test_sample_populator_missing_configs(temp_config_files):
    """Test with dataset keys that don't have corresponding configs."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    dataset = Dataset.from_dict({
        "route": ["coding", "missing_key"],  # missing_key has no config
        "user_id": [1, 2]
    })
    
    with pytest.raises(ValueError, match="Missing configurations for lookup keys"):
        block(dataset)


def test_sample_populator_invalid_yaml(temp_config_files):
    """Test handling of invalid YAML files."""
    temp_dir = os.path.dirname(temp_config_files[0])
    invalid_path = os.path.join(temp_dir, "invalid.yaml")
    
    with open(invalid_path, "w") as f:
        f.write("invalid: yaml: content: [")  # Invalid YAML syntax

    try:
        paths_with_invalid = temp_config_files + [invalid_path]
        block = SamplePopulatorBlock(
            block_name="test_populator",
            input_cols="route",
            config_paths=paths_with_invalid,
        )
        
        # Should handle invalid YAML gracefully
        assert block.configs["invalid"] is None
        
    finally:
        if os.path.exists(invalid_path):
            os.remove(invalid_path)


def test_sample_populator_missing_file(temp_config_files):
    """Test handling of missing config files."""
    missing_path = "/nonexistent/missing.yaml"
    paths_with_missing = temp_config_files + [missing_path]
    
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=paths_with_missing,
    )
    
    # Should handle missing files gracefully
    assert block.configs["missing"] is None


def test_sample_populator_runtime_key_error(temp_config_files):
    """Test runtime error when lookup key not found."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    dataset = Dataset.from_dict({"route": ["nonexistent"], "user_id": [1]})
    
    with pytest.raises(KeyError, match="Lookup key 'nonexistent' not found"):
        block.generate(dataset)


def test_sample_populator_none_config_error(temp_config_files):
    """Test runtime error when config data is None."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    # Manually set config to None to simulate load failure
    block.configs["coding"] = None
    
    dataset = Dataset.from_dict({"route": ["coding"], "user_id": [1]})
    
    with pytest.raises(TypeError, match="Configuration data for 'coding' is None"):
        block.generate(dataset)


def test_sample_populator_validation_errors(temp_config_files):
    """Test Pydantic validation errors."""
    
    # Test multiple input columns not allowed
    with pytest.raises(ValueError, match="exactly one input column"):
        SamplePopulatorBlock(
            block_name="test_populator",
            input_cols=["route", "other"],
            config_paths=temp_config_files,
        )
    
    # Test empty input columns not allowed
    with pytest.raises(ValueError, match="exactly one input column"):
        SamplePopulatorBlock(
            block_name="test_populator",
            input_cols=[],
            config_paths=temp_config_files,
        )


def test_sample_populator_baseblock_integration(temp_config_files):
    """Test BaseBlock interface compliance."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    # Test get_config method
    config = block.get_config()
    assert config["block_name"] == "test_populator"
    assert config["input_cols"] == ["route"]
    assert config["config_paths"] == temp_config_files
    
    # Test get_info method
    info = block.get_info()
    assert info["block_type"] == "SamplePopulatorBlock"
    assert info["block_name"] == "test_populator"
    
    # Test string representation
    repr_str = repr(block)
    assert "SamplePopulatorBlock" in repr_str
    assert "test_populator" in repr_str


def test_sample_populator_full_pipeline(temp_config_files):
    """Test complete pipeline using __call__ method."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        input_cols="route",
        config_paths=temp_config_files,
    )
    
    dataset = Dataset.from_dict({
        "route": ["coding", "writing"],
        "task": ["implement function", "write essay"]
    })
    
    # Use __call__ which triggers full validation and logging
    result = block(dataset)
    
    assert len(result) == 2
    assert result[0]["type"] == "programming"
    assert result[1]["type"] == "composition"
    assert result[0]["task"] == "implement function"  # Original data preserved
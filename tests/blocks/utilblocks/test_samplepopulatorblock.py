"""Test suite for SamplePopulatorBlock functionality.

This module contains tests for the SamplePopulatorBlock class, which populates datasets
with data from configuration files.
"""

# Standard
import os
import tempfile

# Third Party
from datasets import Dataset
import pytest
import yaml

# First Party
from sdg_hub.blocks.utilblocks import SamplePopulatorBlock


@pytest.fixture
def temp_config_files():
    """Create temporary config files for testing."""
    configs = {
        "coding": {"examples": ["code1", "code2"], "type": "programming"},
        "writing": {"examples": ["write1", "write2"], "type": "composition"},
        "math": {"examples": ["math1", "math2"], "type": "calculation"},
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
        os.remove(path)
    os.rmdir(temp_dir)

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "route": ["coding", "writing", "math"],
            "other_col": ["value1", "value2", "value3"],
        }
    )


def test_sample_populator_basic(temp_config_files, sample_dataset):
    """Test basic functionality of SamplePopulatorBlock."""
    block = SamplePopulatorBlock(
        block_name="test_populator", config_paths=temp_config_files, column_name="route"
    )

    result = block.generate(sample_dataset)

    # Check that the config data was properly merged
    assert "examples" in result[0]
    assert "type" in result[0]
    assert result[0]["examples"] == ["code1", "code2"]
    assert result[0]["type"] == "programming"

    # Check that original data is preserved
    assert result[0]["other_col"] == "value1"
    assert result[0]["route"] == "coding"


def test_sample_populator_with_postfix(temp_config_files, sample_dataset):
    """Test SamplePopulatorBlock with postfix in config filenames."""
    # Create postfixed versions of config files
    postfixed_paths = []
    for path in temp_config_files:
        base, ext = os.path.splitext(path)
        new_path = f"{base}_test{ext}"
        with open(path, "r") as src, open(new_path, "w") as dst:
            dst.write(src.read())
        postfixed_paths.append(new_path)

    block = SamplePopulatorBlock(
        block_name="test_populator",
        config_paths=temp_config_files,
        column_name="route",
        post_fix="test",
    )

    result = block.generate(sample_dataset)

    # Verify the data was loaded from postfixed files
    assert "examples" in result[0]
    assert result[0]["examples"] == ["code1", "code2"]

    # Cleanup postfixed files
    for path in postfixed_paths:
        os.remove(path)


def test_sample_populator_invalid_route(temp_config_files):
    """Test SamplePopulatorBlock with invalid route values."""
    dataset = Dataset.from_dict({"route": ["invalid_route"], "other_col": ["value1"]})

    block = SamplePopulatorBlock(
        block_name="test_populator", config_paths=temp_config_files, column_name="route"
    )

    with pytest.raises(KeyError):
        block.generate(dataset)


def test_sample_populator_empty_dataset(temp_config_files):
    """Test SamplePopulatorBlock with empty dataset."""
    dataset = Dataset.from_dict({"route": [], "other_col": []})

    block = SamplePopulatorBlock(
        block_name="test_populator", config_paths=temp_config_files, column_name="route"
    )

    result = block.generate(dataset)
    assert len(result) == 0


def test_sample_populator_custom_num_procs(temp_config_files, sample_dataset):
    """Test SamplePopulatorBlock with custom number of processes."""
    block = SamplePopulatorBlock(
        block_name="test_populator",
        config_paths=temp_config_files,
        column_name="route",
        num_procs=2,
    )

    result = block.generate(sample_dataset)
    assert len(result) == 3
    assert "examples" in result[0]
    assert "type" in result[0]


def test_sample_populator_missing_config_keys(temp_config_files):
    """Test SamplePopulatorBlock with missing keys in config files."""
    # Create a dataset with a route that exists but config has missing keys
    dataset = Dataset.from_dict({"route": ["coding"], "other_col": ["value1"]})

    # Modify the config file to have missing keys
    config_path = temp_config_files[0]  # coding.yaml
    with open(config_path, "w") as f:
        yaml.dump({"type": "programming"}, f)  # Removed 'examples' key

    block = SamplePopulatorBlock(
        block_name="test_populator", config_paths=temp_config_files, column_name="route"
    )

    result = block.generate(dataset)

    # Verify that existing keys are merged and missing keys don't cause errors
    assert "type" in result[0]
    assert result[0]["type"] == "programming"
    assert "other_col" in result[0]
    assert result[0]["other_col"] == "value1"
    assert "route" in result[0]
    assert result[0]["route"] == "coding"


def test_sample_populator_invalid_yaml(temp_config_files):
    """Test SamplePopulatorBlock with invalid YAML content.

    Verifies that the block properly handles invalid YAML configuration files
    by returning None for the config and raising TypeError when trying to merge it.
    """
    # Create an invalid YAML file
    invalid_path = os.path.join(os.path.dirname(temp_config_files[0]), "invalid.yaml")
    with open(invalid_path, "w") as f:
        f.write("invalid: yaml: content: [")  # Invalid YAML syntax

    try:
        block = SamplePopulatorBlock(
            block_name="test_populator",
            config_paths=[invalid_path],
            column_name="route"
        )

        # Create a dataset with a route that matches the invalid config
        dataset = Dataset.from_dict({"route": ["invalid"], "other_col": ["value1"]})

        # The error should be raised when trying to merge with None config
        with pytest.raises(TypeError):
            block.generate(dataset)
    finally:
        # Clean up the invalid file
        if os.path.exists(invalid_path):
            os.remove(invalid_path)
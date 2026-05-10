# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for flow tests."""

# Standard
import tempfile

# Third Party
import pandas as pd
import pytest

# First Party
from sdg_hub import BaseBlock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup
    # Standard
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return pd.DataFrame(
        {
            "input": ["test input 1", "test input 2", "test input 3"],
            "label": ["label1", "label2", "label3"],
        }
    )


class MockBlock(BaseBlock):
    """Mock block for testing that inherits from BaseBlock."""

    def __init__(
        self, block_name="test_block", input_cols=None, output_cols=None, **kwargs
    ):
        super().__init__(
            block_name=block_name,
            input_cols=input_cols or ["input"],
            output_cols=output_cols or ["output"],
            **kwargs,
        )

    def __call__(self, dataset, **kwargs):
        """Mock block execution."""
        # Make a copy to avoid modifying the original
        result = dataset.copy()

        # Add output columns
        if isinstance(self.output_cols, list):
            for col in self.output_cols:
                result[col] = [
                    f"{self.block_name}_{col}_{i}" for i in range(len(dataset))
                ]
        else:
            result[self.output_cols] = [
                f"{self.block_name}_{self.output_cols}_{i}" for i in range(len(dataset))
            ]
        return result

    def generate(self, dataset, **kwargs):
        """Generate method for BaseBlock compatibility."""
        return self(dataset, **kwargs)


@pytest.fixture
def mock_block():
    """Create a mock block for testing."""

    def _create_mock_block(name="test_block", input_cols=None, output_cols=None):
        return MockBlock(
            block_name=name,
            input_cols=input_cols or ["input"],
            output_cols=output_cols or ["output"],
        )

    return _create_mock_block

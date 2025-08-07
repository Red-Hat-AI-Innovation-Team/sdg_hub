"""Test suite for SamplePopulatorBlock deprecation functionality."""

# Standard
import warnings

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.deprecated_blocks import SamplePopulatorBlock
import pytest


def test_sample_populator_deprecation():
    """Test that SamplePopulatorBlock shows deprecation warning and raises NotImplementedError."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        block = SamplePopulatorBlock(
            block_name="test_populator",
            config_paths=["config1.yaml"],
            column_name="route",
        )

        # Check deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated and will be replaced with a router block" in str(
            w[0].message
        )

    # Check that generate raises NotImplementedError
    test_data = Dataset.from_dict({"route": ["test"]})
    with pytest.raises(
        NotImplementedError, match="deprecated and will be replaced with a router block"
    ):
        block.generate(test_data)

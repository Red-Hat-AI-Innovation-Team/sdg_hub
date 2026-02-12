"""Tests for the SamplerBlock functionality."""

# Third Party
# First Party
from sdg_hub.core.blocks.transform import SamplerBlock
from sdg_hub.core.utils.error_handling import MissingColumnError
import pandas as pd
import pytest


def test_sampler_basic():
    """Test basic sampling functionality."""
    data = {
        "id": [1, 2, 3],
        "items": [
            ["a", "b", "c", "d", "e"],
            ["x", "y", "z", "w", "v"],
            ["1", "2", "3", "4", "5"],
        ],
    }
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled_items"],
        num_samples=3,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result) == 3
    assert "sampled_items" in result.columns.tolist()
    assert "items" in result.columns.tolist()

    for sampled in result["sampled_items"]:
        assert len(sampled) == 3
        assert len(sampled) == len(set(sampled))  # No duplicates


def test_sampler_reproducibility():
    """Test that random_seed provides reproducible results."""
    data = {"items": [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]]}
    dataset = pd.DataFrame(data)

    block1 = SamplerBlock(
        block_name="test_sampler_1",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=5,
        random_seed=42,
    )

    block2 = SamplerBlock(
        block_name="test_sampler_2",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=5,
        random_seed=42,
    )

    result1 = block1.generate(dataset)
    result2 = block2.generate(dataset)

    assert result1["sampled"].iloc[0] == result2["sampled"].iloc[0]


def test_sampler_edge_cases():
    """Test edge cases: empty list, None, and list smaller than num_samples."""
    data = {
        "items": [
            [],  # Empty list
            None,  # None value
            ["a", "b"],  # Smaller than num_samples
            ["x", "y", "z", "w", "v"],  # Normal case
        ],
    }
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=3,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert result["sampled"].iloc[0] == []  # Empty list returns empty
    assert result["sampled"].iloc[1] == []  # None returns empty
    assert len(result["sampled"].iloc[2]) == 2  # Returns all available
    assert len(result["sampled"].iloc[3]) == 3  # Normal sampling


def test_sampler_with_sets():
    """Test sampling from sets."""
    data = {"items": [{"a", "b", "c", "d", "e"}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=3,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result["sampled"].iloc[0]) == 3
    assert isinstance(result["sampled"].iloc[0], list)


def test_sampler_validation_input_cols():
    """Test validation errors for input_cols."""
    with pytest.raises(
        ValueError, match="SamplerBlock requires exactly one input column"
    ):
        SamplerBlock(
            block_name="test", input_cols=[], output_cols=["sampled"], num_samples=3
        )

    with pytest.raises(
        ValueError, match="SamplerBlock requires exactly one input column"
    ):
        SamplerBlock(
            block_name="test",
            input_cols=["a", "b"],
            output_cols=["sampled"],
            num_samples=3,
        )


def test_sampler_validation_output_cols():
    """Test validation errors for output_cols."""
    with pytest.raises(
        ValueError, match="SamplerBlock requires exactly one output column"
    ):
        SamplerBlock(
            block_name="test", input_cols=["items"], output_cols=[], num_samples=3
        )

    with pytest.raises(
        ValueError, match="SamplerBlock requires exactly one output column"
    ):
        SamplerBlock(
            block_name="test",
            input_cols=["items"],
            output_cols=["a", "b"],
            num_samples=3,
        )


def test_sampler_missing_input_column():
    """Test error when input column is missing from DataFrame."""
    data = {"other_col": [["a", "b"], ["c", "d"]]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=3,
    )

    with pytest.raises(MissingColumnError):
        block(dataset)


def test_sampler_weighted_dict():
    """Test weighted sampling from dictionary."""
    data = {"items": [{"a": 100, "b": 1, "c": 1, "d": 1, "e": 1}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=2,
        random_seed=42,
    )

    result = block.generate(dataset)
    assert len(result["sampled"].iloc[0]) == 2


def test_sampler_num_samples_validation():
    """Test that num_samples must be at least 1."""
    with pytest.raises(ValueError, match="num_samples must be at least 1"):
        SamplerBlock(
            block_name="test",
            input_cols=["items"],
            output_cols=["sampled"],
            num_samples=0,
        )


def test_sampler_negative_weights():
    """Test that negative weights raise an error."""
    data = {"items": [{"a": 1, "b": -1}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
    )

    with pytest.raises(ValueError, match="Weights must be finite and non-negative"):
        block.generate(dataset)


def test_sampler_empty_dict():
    """Test that empty dict returns empty list."""
    data = {"items": [{}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
    )

    result = block.generate(dataset)
    assert result["sampled"].iloc[0] == []


def test_sampler_all_zero_weights():
    """Test that all zero weights returns empty list."""
    data = {"items": [{"a": 0, "b": 0}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
    )

    result = block.generate(dataset)
    assert result["sampled"].iloc[0] == []


def test_sampler_unsortable_set():
    """Test sampling from set with unsortable mixed types."""
    data = {"items": [{1, "a", 2, "b"}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=2,
        random_seed=42,
    )

    result = block.generate(dataset)
    assert len(result["sampled"].iloc[0]) == 2


def test_sampler_non_iterable():
    """Test that non-iterable values return empty list."""
    data = {"items": [42]}  # Integer is not iterable
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
    )

    result = block.generate(dataset)
    assert result["sampled"].iloc[0] == []


def test_sampler_return_scalar():
    """Test return_scalar=True with num_samples=1 returns scalar values."""
    data = {
        "items": [
            ["a", "b", "c", "d", "e"],
            ["x", "y", "z"],
        ],
    }
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler_scalar",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
        return_scalar=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Should return scalar values, not lists
    assert not isinstance(result["sampled"].iloc[0], list)
    assert not isinstance(result["sampled"].iloc[1], list)
    assert result["sampled"].iloc[0] in ["a", "b", "c", "d", "e"]
    assert result["sampled"].iloc[1] in ["x", "y", "z"]


def test_sampler_return_scalar_with_empty_list():
    """Test return_scalar=True with empty list returns None."""
    data = {
        "items": [
            [],
            None,
        ],
    }
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler_scalar_empty",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
        return_scalar=True,
    )

    result = block.generate(dataset)

    # Empty list and None should return None
    assert result["sampled"].iloc[0] is None
    assert result["sampled"].iloc[1] is None


def test_sampler_return_scalar_default_false():
    """Test that return_scalar defaults to False."""
    data = {"items": [["a", "b", "c"]]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler_default",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Should return a list by default
    assert isinstance(result["sampled"].iloc[0], list)
    assert len(result["sampled"].iloc[0]) == 1


def test_sampler_return_scalar_ignored_when_num_samples_greater_than_1():
    """Test that return_scalar is ignored when num_samples > 1."""
    data = {"items": [["a", "b", "c", "d", "e"]]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler_multi",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=3,
        return_scalar=True,  # Should be ignored since num_samples > 1
        random_seed=42,
    )

    result = block.generate(dataset)

    # Should still return a list since num_samples > 1
    assert isinstance(result["sampled"].iloc[0], list)
    assert len(result["sampled"].iloc[0]) == 3


def test_sampler_return_scalar_with_dict():
    """Test return_scalar=True works with dictionary (weighted) input."""
    data = {"items": [{"a": 10, "b": 1, "c": 1}]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_sampler_scalar_dict",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=1,
        return_scalar=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Should return scalar value
    assert not isinstance(result["sampled"].iloc[0], list)
    assert result["sampled"].iloc[0] in ["a", "b", "c"]

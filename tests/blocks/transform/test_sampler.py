"""Tests for the SamplerBlock functionality."""

# Third Party
# First Party
import pandas as pd
import pytest

from sdg_hub.core.blocks.transform import SamplerBlock
from sdg_hub.core.utils.error_handling import MissingColumnError


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
        ValueError, match="SamplerBlock requires exactly one output column in cell mode"
    ):
        SamplerBlock(
            block_name="test", input_cols=["items"], output_cols=[], num_samples=3
        )

    with pytest.raises(
        ValueError, match="SamplerBlock requires exactly one output column in cell mode"
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


# --- Column mode tests ---


def test_column_mode_basic():
    """Column mode samples K values from the column into K output columns."""
    dataset = pd.DataFrame({"question": [f"q{i}" for i in range(10)]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1", "fs2", "fs3"],
        num_samples=3,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result) == 10
    assert list(result.columns) == ["question", "fs1", "fs2", "fs3"]

    for idx in range(len(result)):
        shots = [
            result["fs1"].iloc[idx],
            result["fs2"].iloc[idx],
            result["fs3"].iloc[idx],
        ]
        for s in shots:
            assert s in dataset["question"].tolist()


def test_column_mode_exclude_self():
    """When exclude_self=True, a row's own value never appears in its samples."""
    dataset = pd.DataFrame({"question": [f"q{i}" for i in range(20)]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1", "fs2"],
        num_samples=2,
        exclude_self=True,
        random_seed=0,
    )

    result = block.generate(dataset)

    for idx in range(len(result)):
        own_value = result["question"].iloc[idx]
        shots = [result["fs1"].iloc[idx], result["fs2"].iloc[idx]]
        assert own_value not in shots, (
            f"Row {idx} value '{own_value}' should not appear in its own shots"
        )


def test_column_mode_exclude_self_false():
    """When exclude_self=False, self-sampling is allowed."""
    dataset = pd.DataFrame({"question": ["a", "b", "c"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1", "fs2"],
        num_samples=2,
        exclude_self=False,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result) == 3
    for idx in range(len(result)):
        shots = [result["fs1"].iloc[idx], result["fs2"].iloc[idx]]
        for s in shots:
            assert s in ["a", "b", "c"]


def test_column_mode_reproducibility():
    """Same random_seed produces identical results in column mode."""
    dataset = pd.DataFrame({"question": [f"q{i}" for i in range(20)]})

    kwargs = dict(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1", "fs2", "fs3"],
        num_samples=3,
        random_seed=123,
    )

    result1 = SamplerBlock(**kwargs).generate(dataset)
    result2 = SamplerBlock(**kwargs).generate(dataset)

    pd.testing.assert_frame_equal(result1, result2)


def test_column_mode_preserves_existing_columns():
    """Output DataFrame retains all original columns in column mode."""
    dataset = pd.DataFrame({"question": ["a", "b", "c", "d"], "extra": [1, 2, 3, 4]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1"],
        num_samples=1,
        exclude_self=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert "extra" in result.columns
    assert list(result["extra"]) == [1, 2, 3, 4]


def test_column_mode_sample_range():
    """sample_range restricts the pool to the specified row range."""
    dataset = pd.DataFrame({"val": [f"v{i}" for i in range(10)]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["val"],
        output_cols=["s1", "s2"],
        num_samples=2,
        exclude_self=False,
        sample_range=[0, 3],
        random_seed=42,
    )

    result = block.generate(dataset)

    pool = {"v0", "v1", "v2"}
    for idx in range(len(result)):
        shots = [result["s1"].iloc[idx], result["s2"].iloc[idx]]
        for s in shots:
            assert s in pool, f"Row {idx} sampled '{s}' outside range [0, 3)"


def test_column_mode_sample_range_validation():
    """Rejects invalid sample_range values."""
    with pytest.raises(ValueError, match="2-element list"):
        SamplerBlock(
            block_name="test",
            source="column",
            input_cols=["q"],
            output_cols=["s1"],
            num_samples=1,
            sample_range=[0],
        )

    with pytest.raises(ValueError, match="start must be less than end"):
        SamplerBlock(
            block_name="test",
            source="column",
            input_cols=["q"],
            output_cols=["s1"],
            num_samples=1,
            sample_range=[5, 3],
        )

    with pytest.raises(ValueError, match="non-negative"):
        SamplerBlock(
            block_name="test",
            source="column",
            input_cols=["q"],
            output_cols=["s1"],
            num_samples=1,
            sample_range=[-1, 3],
        )


def test_column_mode_sample_range_exceeds_dataset():
    """Rejects sample_range that exceeds dataset length."""
    dataset = pd.DataFrame({"val": ["a", "b", "c"]})

    block = SamplerBlock(
        block_name="test",
        source="column",
        input_cols=["val"],
        output_cols=["s1"],
        num_samples=1,
        sample_range=[0, 10],
    )

    with pytest.raises(ValueError, match="exceeds dataset length"):
        block(dataset)


def test_column_mode_with_replacement():
    """Column mode with replace=True allows duplicate samples."""
    dataset = pd.DataFrame({"val": ["a", "b"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["val"],
        output_cols=["s1", "s2", "s3", "s4", "s5"],
        num_samples=5,
        exclude_self=False,
        replace=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result) == 2
    for idx in range(len(result)):
        shots = [result[f"s{j + 1}"].iloc[idx] for j in range(5)]
        for s in shots:
            assert s in ["a", "b"]


def test_cell_mode_with_replacement():
    """Cell mode with replace=True allows duplicate samples."""
    data = {"items": [["a", "b"]]}
    dataset = pd.DataFrame(data)

    block = SamplerBlock(
        block_name="test_replace",
        input_cols=["items"],
        output_cols=["sampled"],
        num_samples=5,
        replace=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    assert len(result["sampled"].iloc[0]) == 5
    for s in result["sampled"].iloc[0]:
        assert s in ["a", "b"]


def test_column_mode_output_cols_mismatch():
    """Rejects output_cols length != num_samples in column mode."""
    with pytest.raises(
        ValueError, match="output_cols length .* must match num_samples"
    ):
        SamplerBlock(
            block_name="test",
            source="column",
            input_cols=["q"],
            output_cols=["fs1", "fs2"],
            num_samples=3,
        )


def test_column_mode_too_few_rows():
    """Rejects dataset too small to sample from without replacement."""
    dataset = pd.DataFrame({"val": ["a", "b"]})

    block = SamplerBlock(
        block_name="test",
        source="column",
        input_cols=["val"],
        output_cols=["s1", "s2"],
        num_samples=2,
        exclude_self=True,
    )

    with pytest.raises(ValueError, match="needs at least 3"):
        block(dataset)


def test_column_mode_called_via_dunder_call():
    """Column mode works correctly when invoked via __call__."""
    dataset = pd.DataFrame({"question": [f"q{i}" for i in range(10)]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["question"],
        output_cols=["fs1", "fs2"],
        num_samples=2,
        random_seed=42,
    )

    result = block(dataset)

    assert len(result) == 10
    assert "fs1" in result.columns
    assert "fs2" in result.columns


# --- exclude_by_value tests ---


def test_column_mode_exclude_by_value():
    """exclude_by_value excludes all pool entries matching the current row's value."""
    dataset = pd.DataFrame({"q": ["a", "a", "a", "b", "c"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["q"],
        output_cols=["s1"],
        num_samples=1,
        exclude_self=True,
        exclude_by_value=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Rows 0, 1, 2 all have value "a" — none of their samples should be "a"
    for idx in range(3):
        assert result["s1"].iloc[idx] != "a", (
            f"Row {idx} sampled 'a' despite exclude_by_value=True"
        )


def test_column_mode_exclude_by_value_after_multiply():
    """Simulates RowMultiplierBlock then column-mode sampling with exclude_by_value."""
    # Simulate 3 original rows multiplied 2x
    original = pd.DataFrame({"q": ["alpha", "beta", "gamma"]})
    multiplied = pd.concat([original] * 2, ignore_index=True)
    # multiplied: alpha, beta, gamma, alpha, beta, gamma

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["q"],
        output_cols=["fs1"],
        num_samples=1,
        exclude_self=True,
        exclude_by_value=True,
        random_seed=42,
    )

    result = block.generate(multiplied)

    for idx in range(len(result)):
        own_value = result["q"].iloc[idx]
        assert result["fs1"].iloc[idx] != own_value, (
            f"Row {idx} (value='{own_value}') sampled itself"
        )


def test_column_mode_exclude_by_value_false_allows_duplicates():
    """Without exclude_by_value, index-only exclusion lets duplicate values through."""
    # 4 copies of "a" and 1 "b" — index exclusion removes only one "a"
    dataset = pd.DataFrame({"q": ["a", "a", "a", "a", "b"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["q"],
        output_cols=["s1", "s2", "s3"],
        num_samples=3,
        exclude_self=True,
        exclude_by_value=False,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Row 0 has value "a", but with index-only exclusion the other "a" copies are eligible
    row0_shots = [result["s1"].iloc[0], result["s2"].iloc[0], result["s3"].iloc[0]]
    assert "a" in row0_shots, "Index-only exclusion should still allow duplicate values"


def test_column_mode_exclude_by_value_pool_too_small():
    """Raises when exclude_by_value leaves too few pool entries for without-replacement."""
    # All values are "a" except one "b" — after excluding "a", only 1 left, need 2
    dataset = pd.DataFrame({"q": ["a", "a", "a", "b"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["q"],
        output_cols=["s1", "s2"],
        num_samples=2,
        exclude_self=True,
        exclude_by_value=True,
    )

    with pytest.raises(ValueError, match="eligible pool entries"):
        block.generate(dataset)


def test_column_mode_exclude_by_value_with_replacement():
    """exclude_by_value works with replace=True."""
    dataset = pd.DataFrame({"q": ["a", "a", "b"]})

    block = SamplerBlock(
        block_name="test_col",
        source="column",
        input_cols=["q"],
        output_cols=["s1", "s2", "s3"],
        num_samples=3,
        exclude_self=True,
        exclude_by_value=True,
        replace=True,
        random_seed=42,
    )

    result = block.generate(dataset)

    # Rows 0 and 1 (value "a") can only sample "b" — all 3 shots must be "b"
    for idx in range(2):
        shots = [result[f"s{j + 1}"].iloc[idx] for j in range(3)]
        assert all(s == "b" for s in shots), f"Row {idx} should only sample 'b'"

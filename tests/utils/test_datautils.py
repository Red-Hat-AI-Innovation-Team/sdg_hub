# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.utils.datautils import validate_no_duplicates
from sdg_hub.core.utils.error_handling import FlowValidationError
import numpy as np
import pytest


def test_validate_no_duplicates_with_unique_data():
    """Test that no exception is raised for datasets with unique rows."""
    dataset = Dataset.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Should not raise any exception and return None
    result = validate_no_duplicates(dataset)
    assert result is None


def test_validate_no_duplicates_with_duplicate_data():
    """Test that FlowValidationError is raised for datasets with duplicate rows."""
    dataset = Dataset.from_dict({"col1": [1, 2, 2, 3], "col2": ["a", "b", "b", "c"]})

    with pytest.raises(FlowValidationError, match="contains 1 duplicate rows"):
        validate_no_duplicates(dataset)


def test_validate_no_duplicates_with_multiple_duplicates():
    """Test correct duplicate count with multiple duplicate rows."""
    dataset = Dataset.from_dict(
        {"col1": [1, 1, 2, 2, 3], "col2": ["a", "a", "b", "b", "c"]}
    )

    with pytest.raises(FlowValidationError, match="contains 2 duplicate rows"):
        validate_no_duplicates(dataset)


def test_validate_no_duplicates_empty_dataset():
    """Test that empty datasets pass validation."""
    dataset = Dataset.from_dict({"col1": [], "col2": []})

    # Should not raise any exception and return None
    result = validate_no_duplicates(dataset)
    assert result is None


def test_validate_no_duplicates_with_numpy_arrays():
    """Test duplicate detection with numpy arrays and scalars."""
    # Create dataset with numpy arrays and scalars that should be considered duplicates
    # when converted to lists/items
    dataset = Dataset.from_dict(
        {
            "numpy_array": [np.array([1, 2, 3]), [1, 2, 3], np.array([1, 2, 3])],
            "numpy_scalar": [np.int64(42), 42, np.int64(42)],
            "mixed": ["text", "text", "text"],
        }
    )

    with pytest.raises(FlowValidationError, match="contains 2 duplicate rows"):
        validate_no_duplicates(dataset)


def test_validate_no_duplicates_with_sets():
    """Test duplicate detection with sets (should be deterministic)."""
    # Sets get converted to lists when stored in HuggingFace datasets,
    # so we test with actual duplicate rows containing the same elements
    dataset = Dataset.from_dict(
        {
            "set_col": [[1, 2, 3], [1, 2, 3], [4, 5, 6]],  # Two identical lists
            "other": ["a", "a", "b"],  # Two identical values
        }
    )

    with pytest.raises(FlowValidationError, match="contains 1 duplicate rows"):
        validate_no_duplicates(dataset)


def test_validate_no_duplicates_with_string_lists():
    """Test duplicate detection with columns containing lists of strings.

    This test specifically emulates the original issue where HuggingFace datasets
    convert list-of-strings columns to numpy arrays, causing TypeError in pandas.duplicated().
    """
    # Create dataset with columns containing lists of strings (the original problematic case)
    dataset = Dataset.from_dict(
        {
            "text_chunks": [
                [
                    "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution.",
                    "Technologists at the local university have developed an AI-powered buoy system to combat this.",
                ],
                [
                    "Different first paragraph about something else entirely.",
                    "And a completely different second paragraph as well.",
                ],
                [
                    "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution.",
                    "Technologists at the local university have developed an AI-powered buoy system to combat this.",
                ],  # Duplicate of first row
            ],
            "metadata": [
                "doc1",
                "doc2",
                "doc1",
            ],  # Make metadata also duplicate so rows are truly identical
        }
    )

    # This should detect the duplicate (first and third rows are identical)
    # Before the fix, this would throw: TypeError: unhashable type: 'numpy.ndarray'
    with pytest.raises(FlowValidationError, match="contains 1 duplicate rows"):
        validate_no_duplicates(dataset)


def test_validate_no_duplicates_with_dictionaries():
    """Test that dictionaries with different values are not considered duplicates."""
    # Test the edge case where dict keys are same but values differ - should NOT be duplicates, should not RAISE
    dataset = Dataset.from_dict(
        {
            "config": [
                {"model": "gpt-4", "temp": 0.7},
                {
                    "model": "gpt-4",
                    "temp": 0.9,
                },  # Same keys, different values - should NOT be duplicate
            ]
        }
    )

    # Should pass validation (no duplicates)
    result = validate_no_duplicates(dataset)
    assert result is None  # Should not raise any exception

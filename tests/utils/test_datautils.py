# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.utils.datautils import validate_no_duplicates
from sdg_hub.core.utils.error_handling import FlowValidationError
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

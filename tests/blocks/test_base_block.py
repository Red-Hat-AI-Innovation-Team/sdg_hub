"""Tests for the enhanced BaseBlock class."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from datasets import Dataset
import pytest

# First Party
from sdg_hub.blocks.base import BaseBlock
from sdg_hub.utils.error_handling import (
    EmptyDatasetError,
    MissingColumnError,
    OutputColumnCollisionError,
)


class DummyBlock(BaseBlock):
    """Concrete implementation of BaseBlock for testing."""

    def __init__(self, block_name: str, **kwargs):
        super().__init__(block_name=block_name, **kwargs)
        self.generate_called = False
        self.generate_args = None
        self.generate_kwargs = None

    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        """Simple test implementation that adds a 'test_output' column."""
        self.generate_called = True
        self.generate_args = samples
        self.generate_kwargs = kwargs

        # Add a simple test column
        def add_test_column(sample):
            sample["test_output"] = f"processed_{sample.get('input', 'unknown')}"
            return sample

        return samples.map(add_test_column)


class TestBaseBlockInitialization:
    """Test BaseBlock initialization and properties."""

    def test_basic_initialization(self):
        """Test basic block initialization."""
        block = DummyBlock("test_block")

        assert block.block_name == "test_block"
        assert block.input_cols == []
        assert block.output_cols == []
        assert hasattr(block, "kwargs")

    def test_initialization_with_single_columns(self):
        """Test initialization with single column specifications."""
        block = DummyBlock(
            "test_block", input_cols="input_col", output_cols="output_col"
        )

        assert block.input_cols == ["input_col"]
        assert block.output_cols == ["output_col"]

    def test_initialization_with_list_columns(self):
        """Test initialization with list column specifications."""
        block = DummyBlock(
            "test_block", input_cols=["col1", "col2"], output_cols=["out1", "out2"]
        )

        assert block.input_cols == ["col1", "col2"]
        assert block.output_cols == ["out1", "out2"]

    def test_initialization_with_kwargs(self):
        """Test initialization with additional kwargs."""
        block = DummyBlock(
            "test_block", input_cols="input", custom_param="value", another_param=42
        )

        assert block.kwargs["custom_param"] == "value"
        assert block.kwargs["another_param"] == 42

    def test_properties_are_immutable(self):
        """Test that properties return copies and are immutable."""
        block = DummyBlock("test_block", input_cols=["col1", "col2"])

        # Get the lists
        input_cols = block.input_cols
        output_cols = block.output_cols

        # Modify the returned lists
        input_cols.append("new_col")
        output_cols.append("new_out")

        # Original properties should be unchanged
        assert block.input_cols == ["col1", "col2"]
        assert block.output_cols == []

    def test_repr(self):
        """Test string representation."""
        block = DummyBlock("test_block", input_cols=["input"], output_cols=["output"])

        expected = "DummyBlock(name='test_block', input_cols=['input'], output_cols=['output'])"
        assert repr(block) == expected


class TestColumnNormalization:
    """Test column normalization functionality."""

    def test_normalize_none(self):
        """Test normalizing None returns empty list."""
        block = DummyBlock("test_block")
        result = block._normalize_columns(None)
        assert result == []

    def test_normalize_string(self):
        """Test normalizing string returns single-item list."""
        block = DummyBlock("test_block")
        result = block._normalize_columns("column")
        assert result == ["column"]

    def test_normalize_list(self):
        """Test normalizing list returns the same list."""
        block = DummyBlock("test_block")
        input_list = ["col1", "col2"]
        result = block._normalize_columns(input_list)
        assert result == input_list
        assert result is input_list  # Should be the same object

    def test_normalize_invalid_type(self):
        """Test normalizing invalid type raises ValueError."""
        block = DummyBlock("test_block")

        with pytest.raises(
            ValueError,
            match="Invalid column specification.*Must be str, List\\[str\\], or None",
        ):
            block._normalize_columns(123)

        with pytest.raises(
            ValueError,
            match="Invalid column specification.*Must be str, List\\[str\\], or None",
        ):
            block._normalize_columns({"col": "value"})


class TestValidation:
    """Test validation methods."""

    def create_test_dataset(self, data=None):
        """Helper to create test datasets."""
        if data is None:
            data = [
                {"input": "test1", "category": "A"},
                {"input": "test2", "category": "B"},
            ]
        return Dataset.from_list(data)

    def test_validate_columns_success(self):
        """Test successful column validation."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", input_cols=["input", "category"])

        # Should not raise any exception
        block._validate_columns(dataset)

    def test_validate_columns_no_input_cols(self):
        """Test validation with no input columns required."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block")  # No input_cols specified

        # Should not raise any exception
        block._validate_columns(dataset)

    def test_validate_columns_missing(self):
        """Test validation with missing columns."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", input_cols=["input", "missing_col"])

        with pytest.raises(MissingColumnError) as exc_info:
            block._validate_columns(dataset)

        assert exc_info.value.block_name == "test_block"
        assert exc_info.value.missing_columns == ["missing_col"]
        assert set(exc_info.value.available_columns) == {"input", "category"}

    def test_validate_dataset_not_empty_success(self):
        """Test successful empty dataset validation."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block")

        # Should not raise any exception
        block._validate_dataset_not_empty(dataset)

    def test_validate_dataset_not_empty_failure(self):
        """Test empty dataset validation failure."""
        empty_dataset = Dataset.from_list([])
        block = DummyBlock("test_block")

        with pytest.raises(EmptyDatasetError) as exc_info:
            block._validate_dataset_not_empty(empty_dataset)

        assert exc_info.value.block_name == "test_block"

    def test_validate_output_columns_success(self):
        """Test successful output column validation."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", output_cols=["new_col"])

        # Should not raise any exception
        block._validate_output_columns(dataset)

    def test_validate_output_columns_no_output_cols(self):
        """Test validation with no output columns."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block")  # No output_cols specified

        # Should not raise any exception
        block._validate_output_columns(dataset)

    def test_validate_output_columns_collision(self):
        """Test output column collision detection."""
        dataset = self.create_test_dataset()
        block = DummyBlock(
            "test_block", output_cols=["input", "new_col"]
        )  # "input" already exists

        with pytest.raises(OutputColumnCollisionError) as exc_info:
            block._validate_output_columns(dataset)

        assert exc_info.value.block_name == "test_block"
        assert exc_info.value.collision_columns == ["input"]
        assert set(exc_info.value.existing_columns) == {"input", "category"}

    def test_validate_dataset_comprehensive(self):
        """Test comprehensive dataset validation."""
        dataset = self.create_test_dataset()
        block = DummyBlock(
            "test_block", input_cols=["input", "category"], output_cols=["new_col"]
        )

        # Should not raise any exception
        block._validate_dataset(dataset)

    def test_validate_dataset_fails_on_empty(self):
        """Test comprehensive validation fails on empty dataset."""
        empty_dataset = Dataset.from_list([])
        block = DummyBlock("test_block")

        with pytest.raises(EmptyDatasetError):
            block._validate_dataset(empty_dataset)

    def test_validate_dataset_fails_on_missing_columns(self):
        """Test comprehensive validation fails on missing columns."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", input_cols=["missing_col"])

        with pytest.raises(MissingColumnError):
            block._validate_dataset(dataset)

    def test_validate_dataset_fails_on_column_collision(self):
        """Test comprehensive validation fails on column collision."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", output_cols=["input"])

        with pytest.raises(OutputColumnCollisionError):
            block._validate_dataset(dataset)


class TestLogging:
    """Test Rich panel logging functionality."""

    def create_test_dataset(self, data=None):
        """Helper to create test datasets."""
        if data is None:
            data = [
                {"input": "test1", "category": "A"},
                {"input": "test2", "category": "B"},
            ]
        return Dataset.from_list(data)

    @patch("sdg_hub.blocks.base.console")
    def test_log_input_data(self, mock_console):
        """Test input data logging."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", input_cols=["input"], output_cols=["output"])

        block._log_input_data(dataset)

        # Verify console.print was called
        mock_console.print.assert_called_once()

        # Get the panel that was printed
        call_args = mock_console.print.call_args[0]
        panel = call_args[0]

        # Verify panel properties
        assert "test_block" in str(panel.title)
        assert panel.border_style == "blue"

    @patch("sdg_hub.blocks.base.console")
    def test_log_output_data(self, mock_console):
        """Test output data logging."""
        input_dataset = self.create_test_dataset()

        # Create output dataset with changes
        output_data = [
            {"input": "test1", "category": "A", "new_col": "value1"},
            {"input": "test2", "category": "B", "new_col": "value2"},
            {"input": "test3", "category": "C", "new_col": "value3"},  # Added row
        ]
        output_dataset = Dataset.from_list(output_data)

        block = DummyBlock("test_block")
        block._log_output_data(input_dataset, output_dataset)

        # Verify console.print was called
        mock_console.print.assert_called_once()

        # Get the panel that was printed
        call_args = mock_console.print.call_args[0]
        panel = call_args[0]

        # Verify panel properties
        assert "test_block - Complete" in str(panel.title)
        assert panel.border_style == "green"


class TestCallMethod:
    """Test the __call__ method integration."""

    def create_test_dataset(self, data=None):
        """Helper to create test datasets."""
        if data is None:
            data = [
                {"input": "test1", "category": "A"},
                {"input": "test2", "category": "B"},
            ]
        return Dataset.from_list(data)

    @patch("sdg_hub.blocks.base.console")
    def test_call_success(self, mock_console):
        """Test successful __call__ execution."""
        dataset = self.create_test_dataset()
        block = DummyBlock(
            "test_block", input_cols=["input", "category"], output_cols=["test_output"]
        )

        result = block(dataset, test_param="value")

        # Verify generate was called
        assert block.generate_called
        assert block.generate_kwargs["test_param"] == "value"

        # Verify result has new column
        assert "test_output" in result.column_names
        assert result[0]["test_output"] == "processed_test1"

        # Verify logging was called (input and output panels)
        assert mock_console.print.call_count == 2

    @patch("sdg_hub.blocks.base.console")
    def test_call_validation_failure(self, mock_console):
        """Test __call__ with validation failure."""
        dataset = self.create_test_dataset()
        block = DummyBlock("test_block", input_cols=["missing_col"])

        with pytest.raises(MissingColumnError):
            block(dataset)

        # Verify generate was NOT called
        assert not block.generate_called

        # Verify input logging was called but not output logging
        assert mock_console.print.call_count == 1

    @patch("sdg_hub.blocks.base.console")
    def test_call_empty_dataset(self, mock_console):
        """Test __call__ with empty dataset."""
        empty_dataset = Dataset.from_list([])
        block = DummyBlock("test_block")

        with pytest.raises(EmptyDatasetError):
            block(empty_dataset)

        # Verify generate was NOT called
        assert not block.generate_called

        # Verify input logging was called
        assert mock_console.print.call_count == 1

    @patch("sdg_hub.blocks.base.console")
    def test_call_column_collision(self, mock_console):
        """Test __call__ with output column collision."""
        dataset = self.create_test_dataset()
        block = DummyBlock(
            "test_block", output_cols=["input"]
        )  # Collision with existing column

        with pytest.raises(OutputColumnCollisionError):
            block(dataset)

        # Verify generate was NOT called
        assert not block.generate_called

        # Verify input logging was called
        assert mock_console.print.call_count == 1


class TestGetInfo:
    """Test the get_info method."""

    def test_get_info_basic(self):
        """Test basic get_info functionality."""
        block = DummyBlock("test_block", input_cols=["input"], output_cols=["output"])

        info = block.get_info()

        expected = {
            "block_name": "test_block",
            "block_type": "DummyBlock",
            "input_cols": ["input"],
            "output_cols": ["output"],
        }

        assert info == expected

    def test_get_info_no_columns(self):
        """Test get_info with no columns specified."""
        block = DummyBlock("test_block")

        info = block.get_info()

        assert info["input_cols"] == []
        assert info["output_cols"] == []
        assert info["block_name"] == "test_block"
        assert info["block_type"] == "DummyBlock"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_abstract_base_cannot_be_instantiated(self):
        """Test that BaseBlock cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBlock("test")  # Should fail because generate() is abstract

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create a larger dataset
        large_data = [{"input": f"test_{i}", "category": "A"} for i in range(1000)]
        dataset = Dataset.from_list(large_data)

        block = DummyBlock("test_block", input_cols=["input", "category"])

        # Should handle large datasets without issues
        block._validate_dataset(dataset)

    def test_unicode_column_names(self):
        """Test handling of unicode column names."""
        data = [{"测试": "value", "カテゴリ": "A"}]
        dataset = Dataset.from_list(data)

        block = DummyBlock("test_block", input_cols=["测试", "カテゴリ"])

        # Should handle unicode column names
        block._validate_columns(dataset)

    def test_empty_column_names(self):
        """Test handling of empty string column names."""
        block = DummyBlock("test_block")

        # Empty string should be treated as valid column name
        result = block._normalize_columns("")
        assert result == [""]

    @patch("sdg_hub.blocks.base.console")
    def test_logging_with_many_columns(self, mock_console):
        """Test logging with datasets that have many columns."""
        # Create dataset with many columns
        data = [{f"col_{i}": f"value_{i}" for i in range(50)}]
        dataset = Dataset.from_list(data)

        block = DummyBlock("test_block")

        # Should handle logging without issues
        block._log_input_data(dataset)

        # Verify logging was called
        mock_console.print.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

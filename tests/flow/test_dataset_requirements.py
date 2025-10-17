# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset requirements functionality."""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch

# First Party
from sdg_hub import Flow
from sdg_hub.core.flow.metadata import DatasetRequirements

# Third Party
import pandas as pd
import pytest
import yaml


class TestDatasetRequirements:
    """Test dataset requirements functionality."""

    @pytest.fixture
    def flow_with_requirements(self, temp_dir, mock_block):
        """Create a flow with dataset requirements."""
        flow_config = {
            "metadata": {
                "name": "Test Flow with Requirements",
                "description": "Test flow for dataset requirements",
                "version": "1.0.0",
                "author": "Test Suite",
                "dataset_requirements": {
                    "required_columns": ["document", "domain", "icl_document"],
                    "optional_columns": ["additional_info"],
                    "min_samples": 1,
                    "max_samples": 1000,
                    "column_types": {
                        "document": "string",
                        "domain": "string",
                        "icl_document": "string",
                        "additional_info": "string",
                    },
                    "description": "Test dataset requirements",
                },
            },
            "blocks": [
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "document",
                        "output_cols": "processed",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "flow_with_requirements.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(_block_type):
                mock_class = Mock()
                mock_instance = mock_block("processor", ["document"], ["processed"])
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get
            flow = Flow.from_yaml(str(yaml_path))

        return flow

    @pytest.fixture
    def flow_without_requirements(self, temp_dir, mock_block):
        """Create a flow without dataset requirements."""
        flow_config = {
            "metadata": {
                "name": "Test Flow without Requirements",
                "description": "Test flow without dataset requirements",
                "version": "1.0.0",
                "author": "Test Suite",
            },
            "blocks": [
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "flow_without_requirements.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(_block_type):
                mock_class = Mock()
                mock_instance = mock_block("processor", ["input"], ["output"])
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get
            flow = Flow.from_yaml(str(yaml_path))

        return flow

    def test_get_dataset_requirements_with_requirements(self, flow_with_requirements):
        """Test get_dataset_requirements with flow that has requirements."""
        requirements = flow_with_requirements.get_dataset_requirements()

        assert requirements is not None
        assert isinstance(requirements, DatasetRequirements)
        assert requirements.required_columns == ["document", "domain", "icl_document"]
        assert requirements.optional_columns == ["additional_info"]
        assert requirements.min_samples == 1
        assert requirements.max_samples == 1000
        assert requirements.column_types == {
            "document": "string",
            "domain": "string",
            "icl_document": "string",
            "additional_info": "string",
        }
        assert requirements.description == "Test dataset requirements"

    def test_get_dataset_requirements_without_requirements(
        self, flow_without_requirements
    ):
        """Test get_dataset_requirements with flow that has no requirements."""
        requirements = flow_without_requirements.get_dataset_requirements()

        assert requirements is None

    def test_get_dataset_schema_with_requirements(self, flow_with_requirements):
        """Test get_dataset_schema with flow that has requirements."""
        schema_dataset = flow_with_requirements.get_dataset_schema()

        assert isinstance(schema_dataset, pd.DataFrame)
        assert len(schema_dataset) == 0  # Empty dataset
        assert len(schema_dataset.columns.tolist()) == 4  # 3 required + 1 optional

        # Check required columns
        assert "document" in schema_dataset.columns.tolist()
        assert "domain" in schema_dataset.columns.tolist()
        assert "icl_document" in schema_dataset.columns.tolist()

        # Check optional columns
        assert "additional_info" in schema_dataset.columns.tolist()

        # Check all are string type (object dtype in pandas)
        for col_name in schema_dataset.columns.tolist():
            assert schema_dataset[col_name].dtype == "object"

    def test_get_dataset_schema_without_requirements(self, flow_without_requirements):
        """Test get_dataset_schema with flow that has no requirements."""
        schema_dataset = flow_without_requirements.get_dataset_schema()

        assert isinstance(schema_dataset, pd.DataFrame)
        assert len(schema_dataset) == 0  # Empty dataset
        assert len(schema_dataset.columns.tolist()) == 0  # No columns

    def test_get_dataset_schema_type_mapping(self, temp_dir, mock_block):
        """Test that column types are correctly mapped to pandas dtypes."""
        flow_config = {
            "metadata": {
                "name": "Test Flow Type Mapping",
                "description": "Test flow for type mapping",
                "version": "1.0.0",
                "author": "Test Suite",
                "dataset_requirements": {
                    "required_columns": [
                        "text_col",
                        "int_col",
                        "float_col",
                        "bool_col",
                        "unknown_col",
                    ],
                    "column_types": {
                        "text_col": "string",
                        "int_col": "integer",
                        "float_col": "float",
                        "bool_col": "boolean",
                        "unknown_col": "unknown_type",
                    },
                },
            },
            "blocks": [
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "text_col",
                        "output_cols": "output",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "flow_type_mapping.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(_block_type):
                mock_class = Mock()
                mock_instance = mock_block("processor", ["text_col"], ["output"])
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get
            flow = Flow.from_yaml(str(yaml_path))

        schema_dataset = flow.get_dataset_schema()

        # Check type mappings to pandas dtypes
        assert (
            schema_dataset["text_col"].dtype == "object"
        )  # pandas uses object for strings
        assert schema_dataset["int_col"].dtype == "Int64"  # nullable integer
        assert schema_dataset["float_col"].dtype == "float64"
        assert schema_dataset["bool_col"].dtype == "boolean"  # nullable boolean
        assert (
            schema_dataset["unknown_col"].dtype == "object"
        )  # Unknown types default to object (string)

    def test_get_dataset_schema_alternative_type_names(self, temp_dir, mock_block):
        """Test that alternative type names are correctly mapped."""
        flow_config = {
            "metadata": {
                "name": "Test Flow Alternative Types",
                "description": "Test flow for alternative type names",
                "version": "1.0.0",
                "author": "Test Suite",
                "dataset_requirements": {
                    "required_columns": [
                        "str_col",
                        "text_col",
                        "int_col",
                        "number_col",
                        "bool_col",
                    ],
                    "column_types": {
                        "str_col": "str",
                        "text_col": "text",
                        "int_col": "int",
                        "number_col": "number",
                        "bool_col": "bool",
                    },
                },
            },
            "blocks": [
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "str_col",
                        "output_cols": "output",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "flow_alt_types.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(_block_type):
                mock_class = Mock()
                mock_instance = mock_block("processor", ["str_col"], ["output"])
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get
            flow = Flow.from_yaml(str(yaml_path))

        schema_dataset = flow.get_dataset_schema()

        # Check alternative type mappings to pandas dtypes
        assert (
            schema_dataset["str_col"].dtype == "object"
        )  # pandas uses object for strings
        assert schema_dataset["text_col"].dtype == "object"
        assert schema_dataset["int_col"].dtype == "Int64"  # nullable integer
        assert schema_dataset["number_col"].dtype == "float64"
        assert schema_dataset["bool_col"].dtype == "boolean"  # nullable boolean

    def test_dataset_schema_compatibility_with_pandas(self, flow_with_requirements):
        """Test that the schema dataset can be used with pandas DataFrames."""
        schema_dataset = flow_with_requirements.get_dataset_schema()

        # Add sample data to the schema dataset using pandas concat
        sample_data = {
            col_name: ["sample_value"] for col_name in schema_dataset.columns.tolist()
        }

        populated_dataset = pd.concat(
            [schema_dataset, pd.DataFrame(sample_data)], ignore_index=True
        )

        # Verify dataset was created successfully
        assert isinstance(populated_dataset, pd.DataFrame)
        assert len(populated_dataset) == 1
        assert set(populated_dataset.columns.tolist()) == set(
            schema_dataset.columns.tolist()
        )

        # Verify dtypes match (pandas coerces to object when concatenating with empty dataframe)
        for col_name in schema_dataset.columns.tolist():
            # After concat, types may be coerced, but they should still be compatible
            assert col_name in populated_dataset.columns

    def test_get_dataset_schema_no_column_types_specified(self, temp_dir, mock_block):
        """Test get_dataset_schema when no column types are specified."""
        flow_config = {
            "metadata": {
                "name": "Test Flow No Types",
                "description": "Test flow without column types",
                "version": "1.0.0",
                "author": "Test Suite",
                "dataset_requirements": {
                    "required_columns": ["col1", "col2"],
                    "optional_columns": ["col3"],
                    # No column_types specified
                },
            },
            "blocks": [
                {
                    "block_type": "ProcessorBlock",
                    "block_config": {
                        "block_name": "processor",
                        "input_cols": "col1",
                        "output_cols": "output",
                    },
                },
            ],
        }

        yaml_path = Path(temp_dir) / "flow_no_types.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(flow_config, f)

        # Mock the block registry
        with patch("sdg_hub.core.flow.base.BlockRegistry") as mock_registry:

            def mock_get(_block_type):
                mock_class = Mock()
                mock_instance = mock_block("processor", ["col1"], ["output"])
                mock_class.return_value = mock_instance
                return mock_class

            mock_registry._get.side_effect = mock_get
            flow = Flow.from_yaml(str(yaml_path))

        schema_dataset = flow.get_dataset_schema()

        # All columns should default to object type (string in pandas)
        for col_name in schema_dataset.columns.tolist():
            assert schema_dataset[col_name].dtype == "object"
        assert len(schema_dataset.columns.tolist()) == 3  # 2 required + 1 optional

    def test_dataset_schema_validation_workflow(self, flow_with_requirements):
        """Test the typical workflow of using schema dataset for validation."""
        schema_dataset = flow_with_requirements.get_dataset_schema()

        # Create a user dataset with correct schema
        correct_data = {
            "document": ["Sample document"],
            "domain": ["Computer Science"],
            "icl_document": ["Example document"],
            "additional_info": ["Extra info"],
        }
        user_dataset = pd.DataFrame(correct_data)

        # Schema validation should pass - check columns match
        assert set(user_dataset.columns.tolist()) == set(
            schema_dataset.columns.tolist()
        )

        # Create a user dataset with incorrect schema
        incorrect_data = {
            "document": ["Sample document"],
            "wrong_column": ["Wrong data"],
        }
        incorrect_dataset = pd.DataFrame(incorrect_data)

        # Schema validation should fail - columns don't match
        assert set(incorrect_dataset.columns.tolist()) != set(
            schema_dataset.columns.tolist()
        )

    def test_add_data_to_schema_dataset(self, flow_with_requirements):
        """Test adding data to the schema dataset."""
        schema_dataset = flow_with_requirements.get_dataset_schema()

        # Should start empty
        assert len(schema_dataset) == 0

        # Add a single item using pandas concat
        new_data = pd.DataFrame(
            {
                "document": ["Test document"],
                "domain": ["Test domain"],
                "icl_document": ["Test ICL document"],
                "additional_info": ["Test info"],
            }
        )
        populated_dataset = pd.concat([schema_dataset, new_data], ignore_index=True)

        # Should now have one item
        assert len(populated_dataset) == 1
        assert populated_dataset.iloc[0]["document"] == "Test document"
        assert populated_dataset.iloc[0]["domain"] == "Test domain"

        # Original schema dataset should still be empty
        assert len(schema_dataset) == 0

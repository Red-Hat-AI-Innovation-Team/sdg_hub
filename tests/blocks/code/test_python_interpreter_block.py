# SPDX-License-Identifier: Apache-2.0
"""Tests for PythonInterpreterBlock."""

from unittest.mock import AsyncMock, MagicMock, patch

from sdg_hub.core.blocks.code import PythonInterpreterBlock
from sdg_hub.core.blocks.registry import BlockRegistry
from sdg_hub.core.connectors.code_interpreter.base import CodeExecutionResult
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
import pandas as pd
import pytest


class TestPythonInterpreterBlockRegistration:
    """Test PythonInterpreterBlock registration."""

    def test_registered_in_block_registry(self):
        """Test PythonInterpreterBlock is registered."""
        block_class = BlockRegistry._get("PythonInterpreterBlock")
        assert block_class == PythonInterpreterBlock

    def test_registered_in_code_category(self):
        """Test PythonInterpreterBlock is in code category."""
        code_blocks = BlockRegistry.list_blocks(category="code")
        assert "PythonInterpreterBlock" in code_blocks


class TestPythonInterpreterBlockConfiguration:
    """Test PythonInterpreterBlock configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        assert block.interpreter_framework == "monty"
        assert block.timeout == 30.0
        assert block.async_mode is False
        assert block.max_concurrency == 10

    def test_custom_configuration(self):
        """Test custom configuration values."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="monty",
            timeout=10.0,
            async_mode=True,
            max_concurrency=5,
            input_cols=["code"],
            output_cols=["result"],
        )

        assert block.interpreter_framework == "monty"
        assert block.timeout == 10.0
        assert block.async_mode is True
        assert block.max_concurrency == 5

    def test_required_block_name(self):
        """Test that block_name is required."""
        with pytest.raises(ValueError):
            PythonInterpreterBlock(
                input_cols=["code"],
                output_cols=["result"],
            )


class TestPythonInterpreterBlockHelperMethods:
    """Test PythonInterpreterBlock helper methods."""

    def test_get_code_col_from_list(self):
        """Test getting code column from list input_cols."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["generated_code"],
            output_cols=["result"],
        )

        assert block._get_code_col() == "generated_code"

    def test_get_code_col_from_dict(self):
        """Test getting code column from dict input_cols."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols={"code_column": "alias"},
            output_cols=["result"],
        )

        assert block._get_code_col() == "code_column"

    def test_get_code_col_empty_raises_error(self):
        """Test error when input_cols is empty."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=[],
            output_cols=["result"],
        )

        with pytest.raises(ConnectorError, match="input_cols must specify"):
            block._get_code_col()

    def test_get_output_col_from_list(self):
        """Test getting output column from list."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["execution_result"],
        )

        assert block._get_output_col() == "execution_result"

    def test_get_output_col_from_dict(self):
        """Test getting output column from dict."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols={"result": "alias"},
        )

        assert block._get_output_col() == "result"

    def test_get_output_col_default(self):
        """Test default output column name."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=[],
        )

        assert block._get_output_col() == "execution_result"


class TestPythonInterpreterBlockGenerate:
    """Test PythonInterpreterBlock generate method."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector."""
        connector = MagicMock()
        connector.execute_code.return_value = CodeExecutionResult(
            success=True,
            output="Hello",
            execution_time_ms=1.0,
        )
        return connector

    def test_generate_sync_mode(self, mock_connector):
        """Test generate in sync mode."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
            async_mode=False,
        )

        df = pd.DataFrame(
            {
                "code": ["print('Hello')", "print('World')"],
            }
        )

        mock_connector.execute_code.side_effect = [
            CodeExecutionResult(success=True, output="Hello", execution_time_ms=1.0),
            CodeExecutionResult(success=True, output="World", execution_time_ms=1.5),
        ]

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 2
        assert "result" in result.columns
        assert result["result"].iloc[0]["success"] is True
        assert result["result"].iloc[0]["output"] == "Hello"
        assert result["result"].iloc[1]["output"] == "World"

    def test_generate_handles_errors(self, mock_connector):
        """Test generate handles code execution errors."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame(
            {
                "code": ["1/0"],
            }
        )

        mock_connector.execute_code.return_value = CodeExecutionResult(
            success=False,
            error="ZeroDivisionError: division by zero",
            execution_time_ms=0.5,
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert result["result"].iloc[0]["success"] is False
        assert "ZeroDivisionError" in result["result"].iloc[0]["error"]

    def test_generate_handles_empty_code(self, mock_connector):
        """Test generate handles empty or invalid code."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame(
            {
                "code": ["", None, "   "],
            }
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        # Empty code should result in failure
        for idx in range(len(result)):
            assert result["result"].iloc[idx]["success"] is False
            assert "Empty or invalid" in result["result"].iloc[idx]["error"]

    def test_generate_async_mode(self):
        """Test generate in async mode."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
            async_mode=True,
            max_concurrency=2,
        )

        df = pd.DataFrame(
            {
                "code": ["print('A')", "print('B')"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            side_effect=[
                CodeExecutionResult(success=True, output="A", execution_time_ms=1.0),
                CodeExecutionResult(success=True, output="B", execution_time_ms=1.0),
            ]
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 2
        assert "result" in result.columns

    @pytest.mark.asyncio
    async def test_generate_async_from_async_context(self):
        """Test generate in async mode from async context."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
            async_mode=True,
        )

        df = pd.DataFrame(
            {
                "code": ["print('test')"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            return_value=CodeExecutionResult(
                success=True, output="test", execution_time_ms=1.0
            )
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 1


class TestPythonInterpreterBlockConnectorIntegration:
    """Test PythonInterpreterBlock connector integration."""

    def test_get_connector_creates_correct_type(self):
        """Test that _get_connector creates correct connector type."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="monty",
            timeout=10.0,
            input_cols=["code"],
            output_cols=["result"],
        )

        # Mock the registry to return a mock connector class
        mock_connector_class = MagicMock()
        mock_connector_instance = MagicMock()
        mock_connector_class.return_value = mock_connector_instance

        with patch.object(ConnectorRegistry, "get", return_value=mock_connector_class):
            connector = block._get_connector()

        assert connector is mock_connector_instance
        mock_connector_class.assert_called_once()

    def test_get_connector_caches_instance(self):
        """Test that _get_connector caches the connector."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        mock_connector_class = MagicMock()
        mock_connector_instance = MagicMock()
        mock_connector_class.return_value = mock_connector_instance

        with patch.object(ConnectorRegistry, "get", return_value=mock_connector_class):
            connector1 = block._get_connector()
            connector2 = block._get_connector()

        # Should only create once
        assert mock_connector_class.call_count == 1
        assert connector1 is connector2

    def test_get_connector_invalid_framework_raises_error(self):
        """Test that invalid framework raises ConnectorError."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="nonexistent",
            input_cols=["code"],
            output_cols=["result"],
        )

        with pytest.raises(ConnectorError, match="not found"):
            block._get_connector()

    def test_get_connector_invalidates_on_config_change(self):
        """Test connector cache invalidation on config change."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="monty",
            timeout=10.0,
            input_cols=["code"],
            output_cols=["result"],
        )

        mock_connector_class = MagicMock()

        with patch.object(ConnectorRegistry, "get", return_value=mock_connector_class):
            block._get_connector()

            # Change timeout
            block.timeout = 20.0
            block._get_connector()

        # Should create new connector
        assert mock_connector_class.call_count == 2


class TestPythonInterpreterBlockCallable:
    """Test PythonInterpreterBlock __call__ method (full pipeline)."""

    def test_callable_validates_columns(self):
        """Test that __call__ validates input columns."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame(
            {
                "wrong_column": ["print('hello')"],
            }
        )

        with pytest.raises(Exception):  # MissingColumnError
            block(df)

    def test_callable_validates_output_collision(self):
        """Test that __call__ detects output column collisions."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame(
            {
                "code": ["print('hello')"],
                "result": ["existing"],  # Collision
            }
        )

        with pytest.raises(Exception):  # OutputColumnCollisionError
            block(df)

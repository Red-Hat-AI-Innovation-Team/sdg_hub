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
        assert block.max_concurrency == 10

    def test_custom_configuration(self):
        """Test custom configuration values."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="monty",
            timeout=10.0,
            max_concurrency=5,
            input_cols=["code"],
            output_cols=["result"],
        )

        assert block.interpreter_framework == "monty"
        assert block.timeout == 10.0
        assert block.max_concurrency == 5

    def test_required_block_name(self):
        """Test that block_name is required."""
        with pytest.raises(ValueError):
            PythonInterpreterBlock(
                input_cols=["code"],
                output_cols=["result"],
            )

    def test_input_cols_must_have_exactly_one(self):
        """Test that input_cols must have exactly one element."""
        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=["code1", "code2"],
                output_cols=["result"],
            )

    def test_input_cols_cannot_be_empty(self):
        """Test that input_cols cannot be empty."""
        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=[],
                output_cols=["result"],
            )

    def test_output_cols_must_have_exactly_one(self):
        """Test that output_cols must have exactly one element."""
        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=["code"],
                output_cols=["result1", "result2"],
            )

    def test_output_cols_cannot_be_empty(self):
        """Test that output_cols cannot be empty."""
        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=["code"],
                output_cols=[],
            )


class TestPythonInterpreterBlockGenerate:
    """Test PythonInterpreterBlock generate method."""

    def test_generate_executes_code(self):
        """Test generate executes code and returns results."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame(
            {
                "code": ["print('Hello')", "print('World')"],
            }
        )

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            return_value=CodeExecutionResult(
                success=True, output="executed", execution_time_ms=1.0
            )
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert len(result) == 2
        assert "result" in result.columns
        assert result["result"].iloc[0]["success"] is True
        assert result["result"].iloc[1]["success"] is True
        assert mock_connector.aexecute_code.call_count == 2

    def test_generate_handles_errors(self):
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

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            return_value=CodeExecutionResult(
                success=False,
                error="ZeroDivisionError: division by zero",
                execution_time_ms=0.5,
            )
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert result["result"].iloc[0]["success"] is False
        assert "ZeroDivisionError" in result["result"].iloc[0]["error"]

    def test_generate_handles_empty_code(self):
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

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock()

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        # Empty code should result in failure without calling connector
        for idx in range(len(result)):
            assert result["result"].iloc[idx]["success"] is False
            assert "Empty or invalid" in result["result"].iloc[idx]["error"]

        # Connector should not be called for empty code
        mock_connector.aexecute_code.assert_not_called()

    def test_generate_respects_max_concurrency(self):
        """Test that max_concurrency limits concurrent executions."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
            max_concurrency=2,
        )

        df = pd.DataFrame(
            {
                "code": ["print('A')", "print('B')", "print('C')"],
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

        assert len(result) == 3
        assert mock_connector.aexecute_code.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_from_async_context(self):
        """Test generate works when called from async context."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
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
        assert result["result"].iloc[0]["success"] is True


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
            block.timeout = 20.0
            block._get_connector()

        assert mock_connector_class.call_count == 2


class TestPythonInterpreterBlockCallable:
    """Test PythonInterpreterBlock __call__ method."""

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
                "result": ["existing"],
            }
        )

        with pytest.raises(Exception):  # OutputColumnCollisionError
            block(df)

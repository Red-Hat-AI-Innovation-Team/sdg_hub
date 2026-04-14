# SPDX-License-Identifier: Apache-2.0
"""Tests for PythonInterpreterBlock."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from sdg_hub.core.blocks.code import PythonInterpreterBlock
from sdg_hub.core.blocks.registry import BlockRegistry
from sdg_hub.core.connectors.code_interpreter.base import CodeExecutionResult
from sdg_hub.core.connectors.exceptions import ConnectorError


class TestPythonInterpreterBlock:
    """Test PythonInterpreterBlock."""

    def test_registered_in_code_category(self):
        """Test block is registered."""
        assert "PythonInterpreterBlock" in BlockRegistry.list_blocks(category="code")

    def test_requires_exactly_one_input_and_output_col(self):
        """Test validation of input/output cols."""
        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=["a", "b"],
                output_cols=["result"],
            )

        with pytest.raises(ValueError, match="exactly one"):
            PythonInterpreterBlock(
                block_name="test",
                input_cols=["code"],
                output_cols=[],
            )

    def test_generate_executes_code(self):
        """Test generate executes code and returns results."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame({"code": ["print('Hello')", "print('World')"]})

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
        assert "result_success" in result.columns
        assert all(r["success"] for r in result["result"])
        assert all(r["output"] == "executed" for r in result["result"])
        assert result["result_success"].all()
        assert mock_connector.aexecute_code.call_count == 2

    def test_generate_validates_code_column_exists(self):
        """Test generate raises error when code column is missing."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame({"wrong_col": ["print('Hello')"]})

        mock_connector = MagicMock()
        with patch.object(block, "_get_connector", return_value=mock_connector):
            with pytest.raises(ValueError, match="Code column 'code' not found"):
                block.generate(df)

    def test_generate_handles_connector_exception(self):
        """Test generate captures connector-level exceptions per row."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame({"code": ["print('x')"]})

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            side_effect=RuntimeError("connector crash")
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert result["result"].iloc[0]["success"] is False
        assert not result["result_success"].iloc[0]
        assert "connector crash" in result["result"].iloc[0]["error"]

    def test_generate_handles_errors(self):
        """Test generate handles execution errors."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame({"code": ["1/0"]})

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock(
            return_value=CodeExecutionResult(
                success=False,
                error="ZeroDivisionError",
                execution_time_ms=0.5,
            )
        )

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert result["result"].iloc[0]["success"] is False
        assert not result["result_success"].iloc[0]
        assert "ZeroDivisionError" in result["result"].iloc[0]["error"]

    def test_generate_skips_empty_code(self):
        """Test empty code returns error without calling connector."""
        block = PythonInterpreterBlock(
            block_name="test",
            input_cols=["code"],
            output_cols=["result"],
        )

        df = pd.DataFrame({"code": ["", None]})

        mock_connector = MagicMock()
        mock_connector.aexecute_code = AsyncMock()

        with patch.object(block, "_get_connector", return_value=mock_connector):
            result = block.generate(df)

        assert all(not r["success"] for r in result["result"])
        assert not result["result_success"].any()
        mock_connector.aexecute_code.assert_not_called()

    def test_invalid_framework_raises_error(self):
        """Test invalid interpreter framework raises error."""
        block = PythonInterpreterBlock(
            block_name="test",
            interpreter_framework="nonexistent",
            input_cols=["code"],
            output_cols=["result"],
        )

        with pytest.raises(ConnectorError, match="not found"):
            block._get_connector()

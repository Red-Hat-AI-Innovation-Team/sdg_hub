# SPDX-License-Identifier: Apache-2.0
"""Tests for MontyConnector."""

from unittest.mock import MagicMock, patch

from sdg_hub.core.connectors.code_interpreter.base import CodeExecutionResult
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
import pytest
import sdg_hub.core.connectors.code_interpreter.monty as monty_module


class TestMontyConnector:
    """Test MontyConnector."""

    def test_registered_in_registry(self):
        """Test connector is registered."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        assert ConnectorRegistry.get("monty") == MontyConnector

    def test_raises_error_when_monty_unavailable(self):
        """Test instantiation fails without pydantic-monty."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        with patch.object(monty_module, "MONTY_AVAILABLE", False):
            with pytest.raises(ConnectorError, match="pydantic-monty is not installed"):
                MontyConnector()

    @pytest.fixture
    def mock_monty(self):
        """Create mock pydantic_monty module."""
        mock_module = MagicMock()
        mock_module.MontyError = Exception

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module

    def test_execute_code_success(self, mock_monty):
        """Test successful code execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "42"

        connector = MontyConnector()
        result = connector.execute_code("print(21 * 2)")

        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_execute_code_with_inputs(self, mock_monty):
        """Test code execution with input variables."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "30"

        connector = MontyConnector()
        result = connector.execute_code("print(x + y)", inputs={"x": 10, "y": 20})

        assert result.success is True
        mock_monty.Monty.assert_called_once()
        assert set(mock_monty.Monty.call_args.kwargs["inputs"]) == {"x", "y"}

    def test_execute_code_handles_error(self, mock_monty):
        """Test error handling during execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.side_effect = mock_monty.MontyError("Division by zero")

        connector = MontyConnector()
        result = connector.execute_code("1/0")

        assert result.success is False
        assert "Division by zero" in result.error

    @pytest.fixture
    def mock_monty_async(self):
        """Create mock with async support."""
        mock_module = MagicMock()
        mock_module.MontyError = Exception

        async def mock_run_async(*args, **kwargs):
            return "async result"

        mock_module.run_monty_async = mock_run_async

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module

    @pytest.mark.asyncio
    async def test_aexecute_code(self, mock_monty_async):
        """Test async code execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        connector = MontyConnector()
        result = await connector.aexecute_code("print('hello')")

        assert result.success is True
        assert result.output == "async result"


class TestCodeExecutionResult:
    """Test CodeExecutionResult model."""

    def test_success_and_error_results(self):
        """Test creating success and error results."""
        success = CodeExecutionResult(success=True, output="Hello")
        assert success.success is True
        assert success.output == "Hello"

        error = CodeExecutionResult(success=False, error="Failed")
        assert error.success is False
        assert error.error == "Failed"

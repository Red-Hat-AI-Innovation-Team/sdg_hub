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

    def test_execute_code_passes_resource_limits(self, mock_monty):
        """Test that resource limits are passed to run_monty."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "result"
        mock_limits = MagicMock()
        mock_monty.ResourceLimits.return_value = mock_limits

        connector = MontyConnector()
        connector.execute_code("print(1)", timeout=30.0)

        # Verify ResourceLimits was created with correct timeout
        mock_monty.ResourceLimits.assert_called_once_with(max_duration_secs=30.0)

        # Verify limits were passed to run_monty
        call_kwargs = mock_monty.run_monty.call_args.kwargs
        assert call_kwargs["limits"] == mock_limits

    def test_execute_code_timeout_exceeded(self, mock_monty):
        """Test that timeout errors are handled correctly."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        # Simulate Monty raising an error when execution exceeds time limit
        mock_monty.run_monty.side_effect = mock_monty.MontyError(
            "execution exceeded time limit"
        )

        connector = MontyConnector()
        result = connector.execute_code("while True: pass", timeout=0.1)

        assert result.success is False
        assert "time limit" in result.error.lower()
        assert result.execution_time_ms is not None

    def test_execute_code_uses_default_timeout(self, mock_monty):
        """Test that default config timeout is used when not specified."""
        from sdg_hub.core.connectors.base import ConnectorConfig
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "result"

        connector = MontyConnector(config=ConnectorConfig(timeout=60.0))
        connector.execute_code("print(1)")

        # Verify ResourceLimits was created with config timeout
        mock_monty.ResourceLimits.assert_called_once_with(max_duration_secs=60.0)

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

    @pytest.mark.asyncio
    async def test_aexecute_code_passes_resource_limits(self):
        """Test that resource limits are passed to run_monty_async."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module = MagicMock()
        mock_module.MontyError = Exception
        mock_limits = MagicMock()
        mock_module.ResourceLimits.return_value = mock_limits

        captured_kwargs = {}

        async def mock_run_async(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return "async result"

        mock_module.run_monty_async = mock_run_async

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                connector = MontyConnector()
                await connector.aexecute_code("print(1)", timeout=15.0)

        # Verify ResourceLimits was created with correct timeout
        mock_module.ResourceLimits.assert_called_once_with(max_duration_secs=15.0)

        # Verify limits were passed to run_monty_async
        assert captured_kwargs["limits"] == mock_limits


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

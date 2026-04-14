# SPDX-License-Identifier: Apache-2.0
"""Tests for MontyConnector."""

from unittest.mock import MagicMock, patch

import pytest

from sdg_hub.core.connectors.code_interpreter.base import CodeExecutionResult
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
import sdg_hub.core.connectors.code_interpreter.monty as monty_module


class MockMontyError(Exception):
    """Custom exception for mocking pydantic_monty.MontyError."""


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
        """Create mock pydantic_monty module with instance-method API."""
        mock_module = MagicMock()
        mock_module.MontyError = MockMontyError

        # Monty() returns a mock instance with .run() method
        mock_instance = MagicMock()
        mock_module.Monty.return_value = mock_instance

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module, mock_instance

    def test_execute_code_success(self, mock_monty):
        """Test successful code execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module, mock_instance = mock_monty

        # Simulate print_callback capturing output and return value
        def fake_run(**kwargs):
            cb = kwargs.get("print_callback")
            if cb:
                cb("stdout", "42")
                cb("stdout", "\n")
            return None  # print() returns None

        mock_instance.run.side_effect = fake_run

        connector = MontyConnector()
        result = connector.execute_code("print(21 * 2)")

        assert result.success is True
        assert result.output == "42\n"
        assert result.return_value is None
        assert result.error is None

    def test_execute_code_with_inputs(self, mock_monty):
        """Test code execution with input variables."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module, mock_instance = mock_monty

        connector = MontyConnector()
        connector.execute_code("print(x + y)", inputs={"x": 10, "y": 20})

        mock_module.Monty.assert_called_once()
        assert set(mock_module.Monty.call_args.kwargs["inputs"]) == {"x", "y"}

    def test_execute_code_handles_monty_error(self, mock_monty):
        """Test MontyError handling during execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module, mock_instance = mock_monty
        mock_instance.run.side_effect = MockMontyError("Division by zero")

        connector = MontyConnector()
        result = connector.execute_code("1/0")

        assert result.success is False
        assert "Division by zero" in result.error

    def test_execute_code_handles_unexpected_error(self, mock_monty):
        """Test that non-MontyError exceptions are also handled."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        _, mock_instance = mock_monty
        mock_instance.run.side_effect = RuntimeError("unexpected failure")

        connector = MontyConnector()
        result = connector.execute_code("x")

        assert result.success is False
        assert "RuntimeError" in result.error
        assert "unexpected failure" in result.error

    def test_execute_code_passes_resource_limits(self, mock_monty):
        """Test that resource limits are passed to run."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module, mock_instance = mock_monty
        mock_limits = MagicMock()
        mock_module.ResourceLimits.return_value = mock_limits

        connector = MontyConnector()
        connector.execute_code("print(1)", timeout=30.0)

        mock_module.ResourceLimits.assert_called_once_with(max_duration_secs=30.0)

        call_kwargs = mock_instance.run.call_args.kwargs
        assert call_kwargs["limits"] == mock_limits

    def test_execute_code_timeout_exceeded(self, mock_monty):
        """Test that timeout errors are handled correctly."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        _, mock_instance = mock_monty
        mock_instance.run.side_effect = MockMontyError("execution exceeded time limit")

        connector = MontyConnector()
        result = connector.execute_code("while True: pass", timeout=0.1)

        assert result.success is False
        assert "time limit" in result.error.lower()
        assert result.execution_time_ms is not None

    def test_execute_code_uses_default_timeout(self, mock_monty):
        """Test that default config timeout is used when not specified."""
        from sdg_hub.core.connectors.base import ConnectorConfig
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module, mock_instance = mock_monty

        connector = MontyConnector(config=ConnectorConfig(timeout=60.0))
        connector.execute_code("print(1)")

        mock_module.ResourceLimits.assert_called_once_with(max_duration_secs=60.0)

    @pytest.fixture
    def mock_monty_async(self):
        """Create mock with async support."""
        mock_module = MagicMock()
        mock_module.MontyError = MockMontyError

        mock_instance = MagicMock()
        mock_module.Monty.return_value = mock_instance

        async def mock_run_async(**kwargs):
            cb = kwargs.get("print_callback")
            if cb:
                cb("stdout", "async result")
            return 42

        mock_instance.run_async = mock_run_async

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module, mock_instance

    @pytest.mark.asyncio
    async def test_aexecute_code(self, mock_monty_async):
        """Test async code execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        connector = MontyConnector()
        result = await connector.aexecute_code("print('hello')")

        assert result.success is True
        assert result.output == "async result"
        assert result.return_value == 42
        assert result.error is None
        assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_aexecute_code_handles_error(self):
        """Test async error handling."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module = MagicMock()
        mock_module.MontyError = MockMontyError

        mock_instance = MagicMock()
        mock_module.Monty.return_value = mock_instance

        async def mock_run_async(**kwargs):
            raise MockMontyError("async division error")

        mock_instance.run_async = mock_run_async

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                connector = MontyConnector()
                result = await connector.aexecute_code("1/0")

        assert result.success is False
        assert "async division error" in result.error

    @pytest.mark.asyncio
    async def test_aexecute_code_passes_resource_limits(self):
        """Test that resource limits are passed to run_async."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_module = MagicMock()
        mock_module.MontyError = MockMontyError
        mock_limits = MagicMock()
        mock_module.ResourceLimits.return_value = mock_limits

        mock_instance = MagicMock()
        mock_module.Monty.return_value = mock_instance

        captured_kwargs = {}

        async def mock_run_async(**kwargs):
            captured_kwargs.update(kwargs)

        mock_instance.run_async = mock_run_async

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                connector = MontyConnector()
                await connector.aexecute_code("print(1)", timeout=15.0)

        mock_module.ResourceLimits.assert_called_once_with(max_duration_secs=15.0)
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

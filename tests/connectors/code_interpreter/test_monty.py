# SPDX-License-Identifier: Apache-2.0
"""Tests for MontyConnector."""

from unittest.mock import MagicMock, patch

from sdg_hub.core.connectors.code_interpreter.base import CodeExecutionResult
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
import pytest

# Import monty module to check availability
import sdg_hub.core.connectors.code_interpreter.monty as monty_module


class TestMontyConnectorRegistration:
    """Test MontyConnector registration."""

    def test_registered_in_registry(self):
        """Test connector is registered under 'monty' name."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        assert ConnectorRegistry.get("monty") == MontyConnector


class TestMontyConnectorWithoutMonty:
    """Test MontyConnector behavior when pydantic-monty is not installed."""

    def test_raises_error_when_monty_unavailable(self):
        """Test that instantiation fails when pydantic-monty is not installed."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        # Patch at the instance level check
        with patch.object(monty_module, "MONTY_AVAILABLE", False):
            with pytest.raises(ConnectorError, match="pydantic-monty is not installed"):
                MontyConnector()

    def test_execute_code_returns_error_when_unavailable(self):
        """Test execute_code returns error result when monty unavailable."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        # Create connector with monty available
        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            connector = MontyConnector()

        # Now simulate monty becoming unavailable for execution
        with patch.object(monty_module, "MONTY_AVAILABLE", False):
            result = connector.execute_code("print('hello')")

        assert result.success is False
        assert "not installed" in result.error


class TestMontyConnectorConfiguration:
    """Test MontyConnector configuration."""

    @pytest.fixture
    def mock_monty_available(self):
        """Patch MONTY_AVAILABLE to True for tests."""
        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            yield

    def test_default_config(self, mock_monty_available):
        """Test default configuration values."""
        from sdg_hub.core.connectors.code_interpreter.monty import (
            MontyConnector,
        )

        # Use default config by not passing anything
        connector = MontyConnector()

        assert connector.config.timeout == 120.0
        assert connector.config.memory_limit == 100 * 1024 * 1024  # 100MB
        assert connector.config.time_limit == 30.0
        assert connector.config.stack_depth_limit == 100

    def test_custom_config(self, mock_monty_available):
        """Test custom configuration values."""
        from sdg_hub.core.connectors.code_interpreter.monty import (
            MontyConnector,
            MontyConnectorConfig,
        )

        config = MontyConnectorConfig(
            timeout=60.0,
            memory_limit=50 * 1024 * 1024,
            time_limit=10.0,
            stack_depth_limit=50,
        )
        connector = MontyConnector(config=config)

        assert connector.config.timeout == 60.0
        assert connector.config.memory_limit == 50 * 1024 * 1024
        assert connector.config.time_limit == 10.0
        assert connector.config.stack_depth_limit == 50


class TestMontyConnectorExecution:
    """Test MontyConnector code execution with mocked Monty."""

    @pytest.fixture
    def mock_monty(self):
        """Create mock pydantic_monty module."""
        mock_module = MagicMock()
        mock_module.MontyError = Exception

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module

    def test_execute_simple_code(self, mock_monty):
        """Test executing simple code."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "2"

        connector = MontyConnector()
        result = connector.execute_code("x = 1 + 1\nprint(x)")

        assert result.success is True
        assert result.output == "2"
        assert result.error is None
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    def test_execute_with_inputs(self, mock_monty):
        """Test executing code with input variables."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = "30"

        connector = MontyConnector()
        result = connector.execute_code(
            "result = x + y\nprint(result)", inputs={"x": 10, "y": 20}
        )

        assert result.success is True
        assert result.output == "30"

        # Verify Monty was called with correct inputs
        mock_monty.Monty.assert_called_once()
        call_args = mock_monty.Monty.call_args
        assert set(call_args.kwargs["inputs"]) == {"x", "y"}

    def test_execute_handles_monty_error(self, mock_monty):
        """Test handling MontyError during execution."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.side_effect = mock_monty.MontyError("Division by zero")

        connector = MontyConnector()
        result = connector.execute_code("1/0")

        assert result.success is False
        assert "Division by zero" in result.error
        assert result.execution_time_ms is not None

    def test_execute_handles_unexpected_error(self, mock_monty):
        """Test handling unexpected exceptions."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        # Use a non-Exception based error that won't be caught as MontyError
        mock_monty.run_monty.side_effect = RuntimeError("Unexpected error")

        connector = MontyConnector()
        result = connector.execute_code("print('hello')")

        assert result.success is False
        # RuntimeError extends Exception which is our mock MontyError,
        # so it's caught as a MontyError - we just check the message
        assert "Unexpected error" in result.error

    def test_execute_with_timeout(self, mock_monty):
        """Test that timeout parameter is respected."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = None

        connector = MontyConnector()
        result = connector.execute_code("print('hello')", timeout=5.0)

        assert result.success is True

    def test_execute_with_none_output(self, mock_monty):
        """Test handling None return value from Monty."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        mock_monty.run_monty.return_value = None

        connector = MontyConnector()
        result = connector.execute_code("x = 1")

        assert result.success is True
        assert result.output == ""
        assert result.return_value is None


class TestMontyConnectorAsyncExecution:
    """Test MontyConnector async execution."""

    @pytest.fixture
    def mock_monty_async(self):
        """Create mock pydantic_monty module with async support."""
        mock_module = MagicMock()
        mock_module.MontyError = Exception

        async def mock_run_monty_async(*args, **kwargs):
            return "async result"

        mock_module.run_monty_async = mock_run_monty_async

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
    async def test_aexecute_code_with_inputs(self, mock_monty_async):
        """Test async execution with inputs."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        connector = MontyConnector()
        result = await connector.aexecute_code("print(x)", inputs={"x": 42})

        assert result.success is True


class TestMontyConnectorBaseInterface:
    """Test BaseConnector interface compliance."""

    @pytest.fixture
    def mock_monty(self):
        """Create mock pydantic_monty module."""
        mock_module = MagicMock()
        mock_module.MontyError = Exception
        mock_module.run_monty.return_value = "result"

        with patch.object(monty_module, "MONTY_AVAILABLE", True):
            with patch.object(monty_module, "pydantic_monty", mock_module):
                yield mock_module

    def test_execute_interface(self, mock_monty):
        """Test execute() method from BaseConnector interface."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        connector = MontyConnector()
        result = connector.execute({"code": "print('hello')"})

        assert isinstance(result, dict)
        assert "success" in result
        assert "output" in result
        assert "error" in result

    def test_execute_with_inputs_and_timeout(self, mock_monty):
        """Test execute() with inputs and timeout in request."""
        from sdg_hub.core.connectors.code_interpreter.monty import MontyConnector

        connector = MontyConnector()
        result = connector.execute(
            {
                "code": "print(x)",
                "inputs": {"x": 10},
                "timeout": 5.0,
            }
        )

        assert isinstance(result, dict)
        assert result["success"] is True


class TestCodeExecutionResult:
    """Test CodeExecutionResult model."""

    def test_success_result(self):
        """Test creating a success result."""
        result = CodeExecutionResult(
            success=True,
            output="Hello World\n",
            execution_time_ms=1.5,
        )

        assert result.success is True
        assert result.output == "Hello World\n"
        assert result.error is None
        assert result.execution_time_ms == 1.5

    def test_error_result(self):
        """Test creating an error result."""
        result = CodeExecutionResult(
            success=False,
            error="ZeroDivisionError: division by zero",
            execution_time_ms=0.5,
        )

        assert result.success is False
        assert result.output is None
        assert "ZeroDivisionError" in result.error

    def test_result_with_return_value(self):
        """Test result with return value."""
        result = CodeExecutionResult(
            success=True,
            output="42",
            return_value=42,
        )

        assert result.return_value == 42

    def test_model_dump(self):
        """Test serialization to dict."""
        result = CodeExecutionResult(
            success=True,
            output="test",
            execution_time_ms=1.0,
        )

        data = result.model_dump()

        assert isinstance(data, dict)
        assert data["success"] is True
        assert data["output"] == "test"

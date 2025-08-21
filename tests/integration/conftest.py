# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and configuration for integration tests."""

from pathlib import Path
from typing import Any, Dict, Generator
import tempfile

import pytest


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_openai_server_config() -> Dict[str, Any]:
    """Mock OpenAI server configuration for testing."""
    return {
        "api_key": "test-key",
        "base_url": "http://localhost:8000/v1",
        "model": "test-model",
    }

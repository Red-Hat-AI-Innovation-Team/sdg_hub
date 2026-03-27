# SPDX-License-Identifier: Apache-2.0
"""Fixtures for InstructLab Q&A flow integration tests."""

from pathlib import Path
import os

import pytest


@pytest.fixture(scope="session")
def test_env_setup():
    """Set up environment variables for testing."""
    from dotenv import load_dotenv

    example_env = Path("examples/knowledge_tuning/instructlab_qna/.env")
    if example_env.exists():
        load_dotenv(example_env, override=False)

    test_defaults = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "SDG_MODEL": os.getenv("SDG_MODEL", "openai/gpt-5-mini"),
    }

    for key, value in test_defaults.items():
        if key not in os.environ and value:
            os.environ[key] = value

    return test_defaults


@pytest.fixture(scope="session")
def notebook_path():
    """Path to the InstructLab Q&A demo notebook."""
    return Path("examples/knowledge_tuning/instructlab_qna/demo.ipynb")

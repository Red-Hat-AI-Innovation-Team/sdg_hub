# SPDX-License-Identifier: Apache-2.0

"""Fixtures for enhanced summary knowledge tuning integration tests."""

from pathlib import Path
import os

import pytest


@pytest.fixture(scope="session")
def test_env_setup():
    """
    Set up environment variables for testing.

    Reads from GitHub secrets in CI (via environment variables),
    or from .env file locally if it exists.
    """
    # In CI, environment variables will be set from GitHub secrets
    # Locally, python-dotenv will load from .env if it exists
    from dotenv import load_dotenv

    # Load .env from the example directory (won't override existing env vars)
    example_env = Path(
        "examples/knowledge_tuning/enhanced_summary_knowledge_tuning/.env"
    )
    if example_env.exists():
        load_dotenv(example_env, override=False)

    # Set test-specific defaults if not already set
    test_defaults = {
        "MODEL_PROVIDER": os.getenv("MODEL_PROVIDER", "openai"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        "NUMBER_OF_SUMMARIES": os.getenv("NUMBER_OF_SUMMARIES", "1"),
        "RUN_ON_VALIDATION_SET": os.getenv("RUN_ON_VALIDATION_SET", "true"),
        "OUTPUT_DATA_FOLDER": os.getenv("OUTPUT_DATA_FOLDER", "test_output"),
    }

    # Only set if not already in environment
    for key, value in test_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    return test_defaults


@pytest.fixture(scope="session")
def notebook_path():
    """Path to the knowledge generation notebook."""
    return Path(
        "examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb"
    )


@pytest.fixture(scope="session")
def converted_script_dir(tmp_path_factory):
    """Directory for converted notebook scripts."""
    return tmp_path_factory.mktemp("converted_scripts")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")

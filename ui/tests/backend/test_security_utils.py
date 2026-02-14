# SPDX-License-Identifier: Apache-2.0
"""Tests for security utility functions."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestMaskApiKey:
    """Tests for mask_api_key function."""
    
    def test_mask_empty_key(self):
        """Test masking empty API key."""
        from utils.api_key_utils import mask_api_key
        assert mask_api_key("") == ""
        assert mask_api_key(None) == ""
    
    def test_mask_empty_value(self):
        """Test masking EMPTY value."""
        from utils.api_key_utils import mask_api_key
        assert mask_api_key("EMPTY") == "EMPTY"
    
    def test_mask_short_key(self):
        """Test masking short API key."""
        from utils.api_key_utils import mask_api_key
        result = mask_api_key("12345678")
        assert result == "********"
    
    def test_mask_long_key(self):
        """Test masking long API key."""
        from utils.api_key_utils import mask_api_key
        key = "test-key-1234567890abcdef"
        result = mask_api_key(key)
        # Should show first 4 and last 4 characters
        assert result.startswith("test")
        assert result.endswith("cdef")
        assert "*" in result


class TestSanitizeModelConfig:
    """Tests for sanitize_model_config function."""
    
    def test_sanitize_empty_config(self):
        """Test sanitizing empty config."""
        from utils.api_key_utils import sanitize_model_config
        assert sanitize_model_config({}) == {}
        assert sanitize_model_config(None) == {}
    
    def test_sanitize_with_env_reference(self):
        """Test sanitizing config with env reference."""
        from utils.api_key_utils import sanitize_model_config
        config = {"api_key": "env:OPENAI_API_KEY", "model": "test-model"}
        result = sanitize_model_config(config, mask_key=True)
        assert result["api_key"] == "env:OPENAI_API_KEY"
        assert result["model"] == "test-model"
    
    def test_sanitize_with_empty_value(self):
        """Test sanitizing config with EMPTY value."""
        from utils.api_key_utils import sanitize_model_config
        config = {"api_key": "EMPTY", "model": "test-model"}
        result = sanitize_model_config(config, mask_key=True)
        assert result["api_key"] == "EMPTY"
    
    def test_sanitize_with_real_key_masked(self):
        """Test sanitizing config with real key (masked)."""
        from utils.api_key_utils import sanitize_model_config
        config = {"api_key": "test-key-1234567890abcdef", "model": "test-model"}
        result = sanitize_model_config(config, mask_key=True)
        assert result["api_key"] != "test-key-1234567890abcdef"
        assert "*" in result["api_key"]
    
    def test_sanitize_with_real_key_removed(self):
        """Test sanitizing config with real key (removed)."""
        from utils.api_key_utils import sanitize_model_config
        config = {"api_key": "test-key-1234567890abcdef", "model": "test-model"}
        result = sanitize_model_config(config, mask_key=False)
        assert "api_key" not in result
        assert result["model"] == "test-model"


class TestResolveEnvVariable:
    """Tests for resolve_env_variable function."""
    
    def test_resolve_non_env_value(self):
        """Test resolving non-env value."""
        from utils.api_key_utils import resolve_env_variable
        assert resolve_env_variable("regular-value") == "regular-value"
        assert resolve_env_variable("") == ""
        assert resolve_env_variable(None) is None
    
    def test_resolve_env_value_exists(self):
        """Test resolving existing env variable."""
        from utils.api_key_utils import resolve_env_variable
        os.environ["TEST_API_KEY_12345"] = "resolved-key"
        try:
            result = resolve_env_variable("env:TEST_API_KEY_12345")
            assert result == "resolved-key"
        finally:
            del os.environ["TEST_API_KEY_12345"]
    
    def test_resolve_env_value_not_exists(self):
        """Test resolving non-existing env variable."""
        from utils.api_key_utils import resolve_env_variable
        result = resolve_env_variable("env:NONEXISTENT_VAR_12345")
        assert result is None


class TestValidateApiKeyFormat:
    """Tests for validate_api_key_format function."""
    
    def test_validate_empty_key(self):
        """Test validating empty API key."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("")
        assert is_valid is False
        assert error == "API key is required"
    
    def test_validate_special_values(self):
        """Test validating special values."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("EMPTY")
        assert is_valid is True
        assert error is None
        
        is_valid, error = validate_api_key_format("NONE")
        assert is_valid is True
        assert error is None
    
    def test_validate_env_reference(self):
        """Test validating env reference."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("env:OPENAI_API_KEY")
        assert is_valid is True
        assert error is None
    
    def test_validate_env_reference_empty_name(self):
        """Test validating env reference with empty name."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("env:")
        assert is_valid is False
        assert "cannot be empty" in error
    
    def test_validate_short_key(self):
        """Test validating too short key."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("short")
        assert is_valid is False
        assert "too short" in error
    
    def test_validate_long_key(self):
        """Test validating too long key."""
        from utils.api_key_utils import validate_api_key_format
        is_valid, error = validate_api_key_format("x" * 600)
        assert is_valid is False
        assert "too long" in error
    
    def test_validate_openai_key_format(self):
        """Test validating OpenAI key format."""
        from utils.api_key_utils import validate_api_key_format
        # Valid OpenAI key format - must start with expected prefix
        # Using obviously fake test value that passes format check
        test_key = "sk-" + "test" * 5  # sk-testtesttesttesttest
        is_valid, error = validate_api_key_format(test_key, "openai")
        assert is_valid is True
        
        # Invalid OpenAI key
        is_valid, error = validate_api_key_format("invalid-key-12345", "openai")
        assert is_valid is False
        assert "sk-" in error  # Error message mentions expected prefix
    
    def test_validate_placeholder_keys(self):
        """Test validating placeholder keys that are explicitly checked."""
        from utils.api_key_utils import validate_api_key_format
        # The function only explicitly checks these specific placeholder strings
        # Short placeholders fail the length check
        short_placeholders = ["test", "example"]
        for placeholder in short_placeholders:
            is_valid, error = validate_api_key_format(placeholder)
            assert is_valid is False
            assert "too short" in error
        
        # These exact strings are checked as placeholders
        exact_placeholders = ["your-api-key", "your-key-here"]
        for placeholder in exact_placeholders:
            is_valid, error = validate_api_key_format(placeholder)
            assert is_valid is False
            # Should fail either due to short length or placeholder check


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""
    
    def test_sanitize_empty(self):
        """Test sanitizing empty filename."""
        from utils.file_handling import sanitize_filename
        assert sanitize_filename("") == ""
        assert sanitize_filename(None) == ""
    
    def test_sanitize_simple_filename(self):
        """Test sanitizing simple filename."""
        from utils.file_handling import sanitize_filename
        assert sanitize_filename("test.jsonl") == "test.jsonl"
        assert sanitize_filename("data_file.csv") == "data_file.csv"
    
    def test_sanitize_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        from utils.file_handling import sanitize_filename
        result = sanitize_filename("test@file#name$.jsonl")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
    
    def test_sanitize_filename_with_path(self):
        """Test sanitizing filename with path."""
        from utils.file_handling import sanitize_filename
        result = sanitize_filename("/path/to/test.jsonl")
        assert "/" not in result
        assert result == "test.jsonl"


class TestSlugifyName:
    """Tests for slugify_name function."""
    
    def test_slugify_empty(self):
        """Test slugifying empty name."""
        from utils.file_handling import slugify_name
        result = slugify_name("")
        assert result.startswith("flow_")
    
    def test_slugify_simple_name(self):
        """Test slugifying simple name."""
        from utils.file_handling import slugify_name
        result = slugify_name("Test Flow")
        assert result == "test_flow"
    
    def test_slugify_name_with_special_chars(self):
        """Test slugifying name with special characters."""
        from utils.file_handling import slugify_name
        result = slugify_name("Test@Flow#Name")
        assert "@" not in result
        assert "#" not in result


class TestDetectPathTraversal:
    """Tests for detect_path_traversal function."""
    
    def test_detect_empty_path(self):
        """Test detecting empty path."""
        from utils.security import detect_path_traversal
        assert detect_path_traversal("") is False
        assert detect_path_traversal(None) is False
    
    def test_detect_normal_path(self):
        """Test normal paths without traversal."""
        from utils.security import detect_path_traversal
        assert detect_path_traversal("file.yaml") is False
        assert detect_path_traversal("subdir/file.yaml") is False
        assert detect_path_traversal("/absolute/path/file.yaml") is False
    
    def test_detect_path_traversal_patterns(self):
        """Test detecting path traversal patterns."""
        from utils.security import detect_path_traversal
        # Basic path traversal
        assert detect_path_traversal("../file.yaml") is True
        assert detect_path_traversal("../../etc/passwd") is True
        assert detect_path_traversal("subdir/../../../etc/passwd") is True
        # Windows-style path traversal
        assert detect_path_traversal("..\\file.yaml") is True
        # Hidden traversal
        assert detect_path_traversal("./subdir/../../../etc/passwd") is True


class TestEnsureWithinDirectory:
    """Tests for ensure_within_directory function."""
    
    def test_path_within_directory(self):
        """Test path within directory."""
        from utils.security import ensure_within_directory
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir).resolve()
            # Create the target path as a file that exists
            subdir = base / "subdir"
            subdir.mkdir(exist_ok=True)
            target_file = subdir / "file.txt"
            target_file.touch()  # Create the file so resolve() works
            result = ensure_within_directory(base, target_file)
            assert str(result).startswith(str(base))
    
    def test_path_outside_directory(self):
        """Test path outside directory raises exception."""
        from utils.security import ensure_within_directory
        from fastapi import HTTPException
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            target = Path("/etc/passwd")
            
            with pytest.raises(HTTPException) as exc_info:
                ensure_within_directory(base, target)
            
            assert exc_info.value.status_code == 400
            assert "outside allowed directory" in str(exc_info.value.detail)


class TestGetCheckpointInfo:
    """Tests for checkpoint utility functions."""
    
    def test_get_checkpoint_info_no_directory(self):
        """Test getting checkpoint info when directory doesn't exist."""
        from utils.checkpoint_utils import get_checkpoint_info
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
                result = get_checkpoint_info("nonexistent-config")
                assert result["has_checkpoints"] is False
                assert result["checkpoint_count"] == 0
    
    def test_clear_checkpoints_no_directory(self):
        """Test clearing checkpoints when directory doesn't exist."""
        from utils.checkpoint_utils import clear_checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utils.checkpoint_utils.CHECKPOINTS_DIR", Path(temp_dir)):
                result = clear_checkpoints("nonexistent-config")
                assert result is True


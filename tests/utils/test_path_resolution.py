"""
Unit tests for path resolution utilities.

This module tests the resolve_path function with various scenarios including
absolute paths, relative paths, single directory searches, and multiple directory searches.
"""

# Standard
import os
import tempfile

# First Party
from sdg_hub.core.utils.path_resolution import resolve_path

# Third Party
import pytest


class TestPathResolution:
    """Test cases for the resolve_path function."""

    def test_absolute_path_returns_unchanged(self):
        """Test that absolute paths are returned unchanged."""
        # Test with Unix-style absolute path
        result = resolve_path("/absolute/path/file.txt", "/some/base/dir")
        assert result == "/absolute/path/file.txt"

    def test_single_directory_search_found(self):
        """Test path resolution with single directory when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path("test.txt", temp_dir)
            assert result == test_file

    def test_single_directory_search_not_found(self):
        """Test path resolution with single directory when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = resolve_path("nonexistent.txt", temp_dir)
            assert result == "nonexistent.txt"  # Returns original filename

    def test_multiple_directory_search_found_in_first(self):
        """Test path resolution with multiple directories when file found in first."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            # Create test file in first directory
            test_file = os.path.join(temp_dir1, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path("test.txt", [temp_dir1, temp_dir2])
            assert result == test_file

    def test_multiple_directory_search_found_in_second(self):
        """Test path resolution with multiple directories when file found in second."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            # Create test file in second directory
            test_file = os.path.join(temp_dir2, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path("test.txt", [temp_dir1, temp_dir2])
            assert result == test_file

    def test_multiple_directory_search_not_found(self):
        """Test path resolution with multiple directories when file not found."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            result = resolve_path("nonexistent.txt", [temp_dir1, temp_dir2])
            assert result == "nonexistent.txt"  # Returns original filename

    def test_multiple_directory_search_prioritizes_first(self):
        """Test that first directory in list takes priority when file exists in both."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            # Create test files in both directories
            test_file1 = os.path.join(temp_dir1, "test.txt")
            test_file2 = os.path.join(temp_dir2, "test.txt")

            with open(test_file1, "w") as f:
                f.write("content from dir1")
            with open(test_file2, "w") as f:
                f.write("content from dir2")

            result = resolve_path("test.txt", [temp_dir1, temp_dir2])
            assert result == test_file1  # Should return first match

    def test_string_directory_converted_to_list(self):
        """Test that string directory is properly converted to list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path("test.txt", temp_dir)
            assert result == test_file

    def test_empty_directory_list(self):
        """Test path resolution with empty directory list."""
        result = resolve_path("test.txt", [])
        assert result == "test.txt"  # Returns original filename

    def test_nonexistent_directories(self):
        """Test path resolution with nonexistent directories."""
        result = resolve_path("test.txt", ["/nonexistent/dir1", "/nonexistent/dir2"])
        assert result == "test.txt"  # Returns original filename

    def test_subdirectory_paths(self):
        """Test path resolution with subdirectory paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory structure
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)

            test_file = os.path.join(subdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path("subdir/test.txt", temp_dir)
            assert result == test_file

    def test_current_directory_relative_path(self):
        """Test that relative paths are returned unchanged when not found in search dirs."""
        result = resolve_path("relative/path/file.txt", ["/some/base/dir"])
        assert result == "relative/path/file.txt"

    def test_dot_notation_paths(self):
        """Test path resolution with dot notation paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Test with "./" prefix
            result = resolve_path("./test.txt", temp_dir)
            # Normalize both paths for comparison
            result_normalized = os.path.normpath(result)
            expected_normalized = os.path.normpath(test_file)
            assert result_normalized == expected_normalized

            # Test with "../" prefix (should still work if parent exists)
            parent_dir = os.path.dirname(temp_dir)
            if os.path.exists(parent_dir):
                result = resolve_path("../test.txt", parent_dir)
                # This would only work if there's a test.txt in the parent's parent
                # For this test, we just verify it doesn't crash

    def test_special_characters_in_filename(self):
        """Test path resolution with special characters in filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with special characters
            special_filename = "test file with spaces and (parentheses).txt"
            test_file = os.path.join(temp_dir, special_filename)
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path(special_filename, temp_dir)
            assert result == test_file

    def test_unicode_characters_in_filename(self):
        """Test path resolution with unicode characters in filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with unicode characters
            unicode_filename = "test_файл_中文_日本語.txt"
            test_file = os.path.join(temp_dir, unicode_filename)
            with open(test_file, "w") as f:
                f.write("test content")

            result = resolve_path(unicode_filename, temp_dir)
            assert result == test_file

    def test_directory_with_trailing_slash(self):
        """Test path resolution with directory paths that have trailing slashes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Test with trailing slash
            dir_with_slash = temp_dir + os.sep
            result = resolve_path("test.txt", dir_with_slash)
            assert result == test_file

    def test_filename_with_leading_slash(self):
        """Test path resolution with filename that has leading slash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Test with leading slash in filename
            result = resolve_path("/test.txt", temp_dir)
            assert result == "/test.txt"  # Should be treated as absolute path

    def test_empty_filename(self):
        """Test path resolution with empty filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = resolve_path("", temp_dir)
            assert result == ""

    def test_none_filename(self):
        """Test path resolution with None filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(TypeError):
                resolve_path(None, temp_dir)

    def test_none_search_dirs(self):
        """Test path resolution with None search directories."""
        with pytest.raises(TypeError):
            resolve_path("test.txt", None)

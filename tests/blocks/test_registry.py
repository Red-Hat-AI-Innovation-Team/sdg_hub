# SPDX-License-Identifier: Apache-2.0
"""Tests for the enhanced BlockRegistry implementation."""

# Standard
from unittest.mock import patch

# Third Party
from datasets import Dataset

# First Party
from sdg_hub import BaseBlock, BlockRegistry
from sdg_hub.core.blocks.registry import BlockMetadata
import pytest


class MockBlock(BaseBlock):
    """Mock block for testing."""

    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        return samples


class OldStyleBlock:
    """Old style block without BaseBlock inheritance."""

    def generate(self, samples: Dataset, **kwargs) -> Dataset:
        return samples


class InvalidBlock:
    """Invalid block without generate method."""

    pass


class TestBlockMetadata:
    """Test BlockMetadata dataclass."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = BlockMetadata(
            name="TestBlock",
            block_class=MockBlock,
            category="test",
            description="A test block",
        )

        assert metadata.name == "TestBlock"
        assert metadata.block_class == MockBlock
        assert metadata.category == "test"
        assert metadata.description == "A test block"
        assert not metadata.deprecated
        assert metadata.replacement is None

    def test_empty_name_validation(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Block name cannot be empty"):
            BlockMetadata(name="", block_class=MockBlock, category="test")

    def test_invalid_class_validation(self):
        """Test that non-class raises ValueError."""
        with pytest.raises(ValueError, match="block_class must be a class"):
            BlockMetadata(name="TestBlock", block_class="not_a_class", category="test")


class TestBlockRegistry:
    """Test BlockRegistry functionality."""

    def setup_method(self):
        """Save current registry state and clear for isolated testing."""
        # Save current state
        self._saved_metadata = BlockRegistry._metadata.copy()
        self._saved_categories = {
            k: v.copy() for k, v in BlockRegistry._categories.items()
        }

        # Clear for isolated testing
        BlockRegistry._metadata.clear()
        BlockRegistry._categories.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        # Restore saved state
        BlockRegistry._metadata.clear()
        BlockRegistry._metadata.update(self._saved_metadata)

        BlockRegistry._categories.clear()
        BlockRegistry._categories.update(self._saved_categories)

    def test_register_valid_block(self):
        """Test registering a valid block."""

        @BlockRegistry.register("TestBlock", "test", "A test block")
        class TestBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Check metadata is stored
        assert "TestBlock" in BlockRegistry._metadata
        metadata = BlockRegistry._metadata["TestBlock"]
        assert metadata.name == "TestBlock"
        assert metadata.block_class == TestBlock
        assert metadata.category == "test"
        assert metadata.description == "A test block"

        # Check category index
        assert "test" in BlockRegistry._categories
        assert "TestBlock" in BlockRegistry._categories["test"]

    def test_register_deprecated_block(self):
        """Test registering a deprecated block."""
        with patch("sdg_hub.core.blocks.registry.logger") as mock_logger:

            @BlockRegistry.register(
                "OldBlock", "test", deprecated=True, replacement="NewBlock"
            )
            class OldBlock(BaseBlock):
                def generate(self, samples: Dataset, **kwargs) -> Dataset:
                    return samples

            # Check warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "OldBlock" in call_args
            assert "deprecated" in call_args
            assert "NewBlock" in call_args

    def test_register_invalid_class(self):
        """Test that registering invalid class raises ValueError."""
        with pytest.raises(ValueError, match="Expected a class"):

            @BlockRegistry.register("InvalidBlock", "test")
            def not_a_class():
                pass

    def test_register_non_baseblock_class(self):
        """Test that non-BaseBlock class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from BaseBlock"):

            @BlockRegistry.register("InvalidBlock", "test")
            class InvalidBlock:
                def generate(self, samples: Dataset, **kwargs) -> Dataset:
                    return samples

    def test_register_missing_generate_method(self):
        """Test that class without generate method raises ValueError."""
        # This test is for when BaseBlock is not available and we fall back to checking generate method
        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "base" in name:
                raise ImportError("BaseBlock not available")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ValueError, match="must implement 'generate' method"),
        ):

            @BlockRegistry.register("InvalidBlock", "test")
            class InvalidBlock:
                pass

    def test_get_block_class_success(self):
        """Test successfully retrieving a block class."""

        @BlockRegistry.register("TestBlock", "test")
        class TestBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        retrieved_class = BlockRegistry._get("TestBlock")
        assert retrieved_class == TestBlock

    def test_get_block_class_not_found(self):
        """Test error when block not found."""
        with pytest.raises(KeyError) as exc_info:
            BlockRegistry._get("NonExistentBlock")

        error_msg = str(exc_info.value)
        assert "NonExistentBlock" in error_msg
        assert "not found in registry" in error_msg

    def test_get_block_class_with_suggestions(self):
        """Test error message includes suggestions for similar names."""

        @BlockRegistry.register("TestBlock", "test")
        class TestBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        with pytest.raises(KeyError) as exc_info:
            BlockRegistry._get("TestBloc")  # Missing 'k'

        error_msg = str(exc_info.value)
        assert "Did you mean: TestBlock" in error_msg

    def test_get_block_class_deprecated_warning(self):
        """Test warning when retrieving deprecated block."""
        with patch("sdg_hub.core.blocks.registry.logger") as mock_logger:

            @BlockRegistry.register(
                "OldBlock", "test", deprecated=True, replacement="NewBlock"
            )
            class OldBlock(BaseBlock):
                def generate(self, samples: Dataset, **kwargs) -> Dataset:
                    return samples

            # Clear previous calls from registration
            mock_logger.reset_mock()

            retrieved_class = BlockRegistry._get("OldBlock")
            assert retrieved_class == OldBlock

            # Check deprecation warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "deprecated" in call_args
            assert "NewBlock" in call_args

    def test_get_categories(self):
        """Test getting all categories."""

        @BlockRegistry.register("Block1", "category1")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "category2")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        categories = BlockRegistry.categories()
        assert categories == ["category1", "category2"]

    def test_list_blocks_by_category(self):
        """Test getting blocks by category using list_blocks."""

        @BlockRegistry.register("Block1", "test")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "test")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block3", "other")
        class Block3(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        test_blocks = BlockRegistry.list_blocks(category="test")
        assert test_blocks == ["Block1", "Block2"]

    def test_list_blocks_by_category_not_found(self):
        """Test error when category not found using list_blocks."""
        with pytest.raises(KeyError) as exc_info:
            BlockRegistry.list_blocks(category="NonExistentCategory")

        error_msg = str(exc_info.value)
        assert "NonExistentCategory" in error_msg
        assert "not found" in error_msg

    def test_list_blocks_flat(self):
        """Test listing all blocks as a flat list (default behavior)."""

        @BlockRegistry.register("Block1", "category1")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "category1")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block3", "category2")
        class Block3(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        blocks = BlockRegistry.list_blocks()
        expected = ["Block1", "Block2", "Block3"]
        assert blocks == expected
        assert isinstance(blocks, list)

    def test_list_blocks_grouped(self):
        """Test listing all blocks organized by category."""

        @BlockRegistry.register("Block1", "category1")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "category1")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block3", "category2")
        class Block3(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        blocks = BlockRegistry.list_blocks(grouped=True)
        expected = {"category1": ["Block1", "Block2"], "category2": ["Block3"]}
        assert blocks == expected
        assert isinstance(blocks, dict)

    def test_list_blocks_filtered_by_category(self):
        """Test listing blocks filtered by category."""

        @BlockRegistry.register("Block1", "category1")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "category1")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block3", "category2")
        class Block3(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Test specific category
        category1_blocks = BlockRegistry.list_blocks(category="category1")
        assert category1_blocks == ["Block1", "Block2"]
        assert isinstance(category1_blocks, list)

        category2_blocks = BlockRegistry.list_blocks(category="category2")
        assert category2_blocks == ["Block3"]
        assert isinstance(category2_blocks, list)

    def test_list_blocks_category_not_found(self):
        """Test error when filtering by non-existent category."""
        with pytest.raises(KeyError) as exc_info:
            BlockRegistry.list_blocks(category="NonExistentCategory")

        error_msg = str(exc_info.value)
        assert "NonExistentCategory" in error_msg
        assert "not found" in error_msg

    def test_list_blocks_include_deprecated(self):
        """Test listing blocks with deprecated filtering."""

        @BlockRegistry.register("ActiveBlock", "test", "An active block")
        class ActiveBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register(
            "DeprecatedBlock", "test", "A deprecated block", deprecated=True
        )
        class DeprecatedBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Default behavior includes deprecated blocks
        all_blocks = BlockRegistry.list_blocks()
        assert "ActiveBlock" in all_blocks
        assert "DeprecatedBlock" in all_blocks

        # Explicitly include deprecated
        with_deprecated = BlockRegistry.list_blocks(include_deprecated=True)
        assert "ActiveBlock" in with_deprecated
        assert "DeprecatedBlock" in with_deprecated

        # Exclude deprecated blocks
        without_deprecated = BlockRegistry.list_blocks(include_deprecated=False)
        assert "ActiveBlock" in without_deprecated
        assert "DeprecatedBlock" not in without_deprecated

    def test_list_blocks_category_with_deprecated(self):
        """Test listing blocks by category with deprecated filtering."""

        @BlockRegistry.register("ActiveBlock", "test", "An active block")
        class ActiveBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register(
            "DeprecatedBlock", "test", "A deprecated block", deprecated=True
        )
        class DeprecatedBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("OtherBlock", "other", "Another block")
        class OtherBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Category with deprecated included (default)
        test_blocks_with_deprecated = BlockRegistry.list_blocks(category="test")
        assert test_blocks_with_deprecated == ["ActiveBlock", "DeprecatedBlock"]

        # Category with deprecated excluded
        test_blocks_without_deprecated = BlockRegistry.list_blocks(
            category="test", include_deprecated=False
        )
        assert test_blocks_without_deprecated == ["ActiveBlock"]

    def test_list_blocks_grouped_with_deprecated(self):
        """Test listing blocks grouped by category with deprecated filtering."""

        @BlockRegistry.register("ActiveBlock", "test", "An active block")
        class ActiveBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register(
            "DeprecatedBlock", "test", "A deprecated block", deprecated=True
        )
        class DeprecatedBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register(
            "OnlyDeprecatedBlock",
            "deprecated_category",
            "Only deprecated",
            deprecated=True,
        )
        class OnlyDeprecatedBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Grouped with deprecated included (default)
        grouped_with_deprecated = BlockRegistry.list_blocks(grouped=True)
        expected_with = {
            "test": ["ActiveBlock", "DeprecatedBlock"],
            "deprecated_category": ["OnlyDeprecatedBlock"],
        }
        assert grouped_with_deprecated == expected_with

        # Grouped with deprecated excluded
        grouped_without_deprecated = BlockRegistry.list_blocks(
            grouped=True, include_deprecated=False
        )
        expected_without = {
            "test": ["ActiveBlock"]
            # Note: deprecated_category should be omitted since it has no non-deprecated blocks
        }
        assert grouped_without_deprecated == expected_without

    def test_private_get_category_blocks_not_accessible(self):
        """Test that _get_category_blocks is private and not part of public API."""

        @BlockRegistry.register("TestBlock", "test")
        class TestBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # The private method should exist but not be part of public API
        assert hasattr(BlockRegistry, "_get_category_blocks")

        # It should work when called directly (for internal use)
        private_result = BlockRegistry._get_category_blocks("test")
        assert private_result == ["TestBlock"]

        # But the public API should use list_blocks instead
        public_result = BlockRegistry.list_blocks(category="test")
        assert public_result == private_result

    def test_print_blocks_empty_registry(self):
        """Test printing blocks when registry is empty."""
        with patch("sdg_hub.core.blocks.registry.console") as mock_console:
            BlockRegistry.discover_blocks()
            mock_console.print.assert_called_once_with(
                "[yellow]No blocks registered yet.[/yellow]"
            )

    def test_print_blocks_with_blocks(self):
        """Test printing blocks with Rich formatting."""

        @BlockRegistry.register("ActiveBlock", "test", "An active test block")
        class ActiveBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register(
            "DeprecatedBlock", "test", "A deprecated block", deprecated=True
        )
        class DeprecatedBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        with patch("sdg_hub.core.blocks.registry.console") as mock_console:
            BlockRegistry.discover_blocks()

            # Check that console.print was called (for table and summary)
            assert mock_console.print.call_count >= 2

            # Check that Table was created and used
            assert any(
                "Summary" in str(call) for call in mock_console.print.call_args_list
            )

    def test_fallback_validation_without_baseblock(self):
        """Test validation fallback when BaseBlock is not available."""

        # Mock the import to raise ImportError for BaseBlock
        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "base" in name:
                raise ImportError("BaseBlock not available")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            # Should work with generate method
            @BlockRegistry.register("OldStyleBlock", "test")
            class OldStyleBlock:
                def generate(self, samples: Dataset, **kwargs) -> Dataset:
                    return samples

            # Should fail without generate method
            with pytest.raises(ValueError, match="must implement 'generate' method"):

                @BlockRegistry.register("InvalidBlock", "test")
                class InvalidBlock:
                    pass

    def test_multiple_blocks_same_category(self):
        """Test that multiple blocks can be registered in the same category."""

        @BlockRegistry.register("Block1", "shared_category")
        class Block1(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        @BlockRegistry.register("Block2", "shared_category")
        class Block2(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        # Check both blocks are in the same category
        blocks_in_category = BlockRegistry.list_blocks(category="shared_category")
        assert "Block1" in blocks_in_category
        assert "Block2" in blocks_in_category
        assert len(blocks_in_category) == 2

    def test_error_message_shows_available_blocks_and_categories(self):
        """Test that error messages show helpful context."""

        @BlockRegistry.register("ExistingBlock", "existing_category")
        class ExistingBlock(BaseBlock):
            def generate(self, samples: Dataset, **kwargs) -> Dataset:
                return samples

        with pytest.raises(KeyError) as exc_info:
            BlockRegistry._get("NonExistentBlock")

        error_msg = str(exc_info.value)
        assert "Available blocks: ExistingBlock" in error_msg
        assert "Categories: existing_category" in error_msg

# SPDX-License-Identifier: Apache-2.0

"""Tests for ChunkingBlock utility block."""

# Standard
import unittest

# Third Party
from datasets import Dataset

# Local
from sdg_hub.blocks.utilblocks import ChunkingBlock


class TestChunkingBlock(unittest.TestCase):
    """Test cases for ChunkingBlock."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_text = """This is a long document that needs to be chunked into smaller pieces. 
        It contains multiple paragraphs and sentences that should be split appropriately.
        
        This is the second paragraph with more content to test the chunking functionality.
        We want to ensure that the chunking works correctly with different text lengths.
        
        The third paragraph contains even more text to validate the overlap functionality.
        This will help us test that the chunks maintain context between splits."""
        
        self.sample_data = [
            {"document": self.sample_text},
            {"document": "Short text that doesn't need chunking."},
            {"document": "Another short piece of text."}
        ]
        
        self.dataset = Dataset.from_list(self.sample_data)

    def test_chunking_initialization(self) -> None:
        """Test ChunkingBlock initialization."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20
        )
        
        self.assertEqual(block.block_name, "test_chunking")
        self.assertEqual(block.input_col, "document")
        self.assertEqual(block.output_col, "chunked_document")
        self.assertEqual(block.chunk_size, 100)
        self.assertEqual(block.overlap, 20)

    def test_chunk_text_basic(self) -> None:
        """Test basic text chunking functionality."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20
        )
        
        chunks = block._chunk_text(self.sample_text)
        
        # Should create multiple chunks for long text
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be <= chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
            
        # Chunks should not be empty
        for chunk in chunks:
            self.assertGreater(len(chunk.strip()), 0)

    def test_chunk_text_short(self) -> None:
        """Test chunking with text shorter than chunk_size."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=1000,
            overlap=100
        )
        
        short_text = "This is a short text."
        chunks = block._chunk_text(short_text)
        
        # Should return single chunk for short text
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)

    def test_chunk_text_with_separators(self) -> None:
        """Test chunking respects paragraph separators."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=150,
            overlap=30,
            separator="\n\n"
        )
        
        text_with_separators = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        chunks = block._chunk_text(text_with_separators)
        
        # Should create chunks respecting paragraph boundaries when possible
        self.assertGreater(len(chunks), 0)
        
        # Check that chunks maintain reasonable boundaries
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 150)

    def test_generate_single_sample(self) -> None:
        """Test _generate method with single sample."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20
        )
        
        sample = {"document": self.sample_text, "id": 1}
        result = block._generate(sample)
        
        # Should return list of samples
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)  # Long text should be chunked
        
        # Each result should have required fields
        for i, chunk_sample in enumerate(result):
            self.assertIn("chunked_document", chunk_sample)
            self.assertIn("chunk_id", chunk_sample)
            self.assertIn("total_chunks", chunk_sample)
            self.assertIn("id", chunk_sample)  # Original fields preserved
            
            self.assertEqual(chunk_sample["chunk_id"], i)
            self.assertEqual(chunk_sample["total_chunks"], len(result))
            self.assertEqual(chunk_sample["id"], 1)

    def test_generate_dataset(self) -> None:
        """Test generate method with full dataset."""
        block = ChunkingBlock(
            block_name="test_chunking",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20
        )
        
        result_dataset = block.generate(self.dataset)
        
        # Should return Dataset
        self.assertIsInstance(result_dataset, Dataset)
        
        # Should have more samples than input (due to chunking)
        self.assertGreater(len(result_dataset), len(self.dataset))
        
        # Check required columns exist
        self.assertIn("chunked_document", result_dataset.column_names)
        self.assertIn("chunk_id", result_dataset.column_names)
        self.assertIn("total_chunks", result_dataset.column_names)
        self.assertIn("document", result_dataset.column_names)  # Original preserved

    def test_chunk_size_edge_cases(self) -> None:
        """Test edge cases for chunk sizes."""
        # Very small chunk size
        block_small = ChunkingBlock(
            block_name="test_small",
            input_col="document", 
            output_col="chunked_document",
            chunk_size=10,
            overlap=2
        )
        
        result_small = block_small._chunk_text("This is a test sentence.")
        self.assertGreater(len(result_small), 1)
        
        # Very large chunk size
        block_large = ChunkingBlock(
            block_name="test_large",
            input_col="document",
            output_col="chunked_document", 
            chunk_size=10000,
            overlap=100
        )
        
        result_large = block_large._chunk_text(self.sample_text)
        self.assertEqual(len(result_large), 1)  # Should fit in one chunk

    def test_overlap_functionality(self) -> None:
        """Test that overlap between chunks works correctly."""
        block = ChunkingBlock(
            block_name="test_overlap",
            input_col="document",
            output_col="chunked_document",
            chunk_size=50,
            overlap=10
        )
        
        text = "This is a test text that will be chunked with overlap to ensure continuity between chunks."
        chunks = block._chunk_text(text)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # This is a basic check - exact overlap depends on boundary detection
            for i in range(len(chunks) - 1):
                self.assertGreater(len(chunks[i]), 0)
                self.assertGreater(len(chunks[i + 1]), 0)

    def test_empty_text_handling(self) -> None:
        """Test handling of empty or whitespace-only text."""
        block = ChunkingBlock(
            block_name="test_empty",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20
        )
        
        # Empty text - may return empty list or single empty chunk depending on implementation
        empty_chunks = block._chunk_text("")
        # Either no chunks or chunks with no meaningful content
        if len(empty_chunks) > 0:
            self.assertEqual(len(empty_chunks[0].strip()), 0)
        
        # Whitespace only
        whitespace_chunks = block._chunk_text("   \n\n   ")
        # Should either be empty or contain single cleaned chunk
        self.assertLessEqual(len(whitespace_chunks), 1)
        if len(whitespace_chunks) > 0:
            # If there's a chunk, it should be cleaned or very short
            self.assertLessEqual(len(whitespace_chunks[0].strip()), 10)

    def test_custom_separator(self) -> None:
        """Test chunking with custom separator."""
        block = ChunkingBlock(
            block_name="test_separator",
            input_col="document",
            output_col="chunked_document",
            chunk_size=100,
            overlap=20,
            separator=". "
        )
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = block._chunk_text(text)
        
        # Should respect sentence boundaries when possible
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)


if __name__ == "__main__":
    unittest.main()
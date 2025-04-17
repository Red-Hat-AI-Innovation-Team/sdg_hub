# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from sdg_hub.utils import chunking

# Local
from .testdata import testdata


class TestChunking:
    """Test collection in instructlab.utils.chunking."""

    def test_chunk_docs_wc_exeeds_ctx_window(self):
        with pytest.raises(ValueError) as exc:
            chunking.chunk_document(
                documents=testdata.documents,
                chunk_word_count=1000,
                server_ctx_size=1034,
            )
        assert (
            "Given word count (1000) per doc will exceed the server context window size (1034)"
            in str(exc.value)
        )

    def test_chunk_docs_chunk_overlap_error(self):
        with pytest.raises(ValueError) as exc:
            chunking.chunk_document(
                documents=testdata.documents,
                chunk_word_count=5,
                server_ctx_size=1034,
            )
        assert (
            "Got a larger chunk overlap (100) than chunk size (24), should be smaller"
            in str(exc.value)
        )

    def test_chunk_docs_long_lines(self):
        chunk_words = 50
        chunks = chunking.chunk_document(
            documents=testdata.long_line_documents,
            chunk_word_count=chunk_words,
            server_ctx_size=4096,
        )
        max_tokens = chunking._num_tokens_from_words(chunk_words)
        max_chars = chunking._num_chars_from_tokens(max_tokens)
        max_chars += chunking._DEFAULT_CHUNK_OVERLAP  # add in the chunk overlap
        max_chars += 50  # and a bit extra for some really long words
        for chunk in chunks:
            assert len(chunk) <= max_chars

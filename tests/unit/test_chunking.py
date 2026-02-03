"""
Unit tests for smart chunking module.
Tests block classification, merging, deduplication, and filtering.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.core.pdf_parser import TextBlock
from app.core.chunking import (
    SmartChunker,
    ProcessedChunk,
    ChunkMetadata,
    create_smart_chunker,
)


class TestSmartChunker:
    """Tests for SmartChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a SmartChunker with default settings."""
        return SmartChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            skip_headers_footers=True,
            deduplicate=True,
        )

    @pytest.fixture
    def chunker_no_filter(self):
        """Create a SmartChunker without header/footer filtering."""
        return SmartChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            skip_headers_footers=False,
            deduplicate=False,
        )

    # ============ Block Classification Tests ============

    def test_classify_page_number(self, chunker):
        """Test that standalone numbers are classified as page numbers."""
        block = TextBlock(
            text="42",
            page_number=1,
            x0=300, y0=750, x1=320, y1=760,
            line_start=1, line_end=1,
        )
        chunk_type = chunker._classify_block_type(block, page_height=792)
        assert chunk_type == "page_number"

    def test_classify_page_number_with_dashes(self, chunker):
        """Test that numbers with dashes are classified as page numbers."""
        for text in ["- 5 -", "-- 10 --", "   42   "]:
            block = TextBlock(
                text=text,
                page_number=1,
                x0=300, y0=750, x1=320, y1=760,
                line_start=1, line_end=1,
            )
            chunk_type = chunker._classify_block_type(block, page_height=792)
            assert chunk_type == "page_number", f"Failed for text: {text}"

    def test_classify_header_by_position(self, chunker):
        """Test that short text at the top of page is classified as header."""
        block = TextBlock(
            text="Chapter 1",
            page_number=1,
            x0=72, y0=50, x1=200, y1=70,  # Top 10% of page
            line_start=1, line_end=1,
        )
        chunk_type = chunker._classify_block_type(block, page_height=792)
        assert chunk_type == "header"

    def test_classify_header_by_keyword(self, chunker):
        """Test that text with header keywords is classified as header."""
        for keyword in ["Confidential", "Draft", "Copyright 2024"]:
            block = TextBlock(
                text=keyword,
                page_number=1,
                x0=72, y0=50, x1=200, y1=70,
                line_start=1, line_end=1,
            )
            chunk_type = chunker._classify_block_type(block, page_height=792)
            assert chunk_type == "header", f"Failed for keyword: {keyword}"

    def test_classify_footer_by_position(self, chunker):
        """Test that short text at the bottom of page is classified as footer."""
        block = TextBlock(
            text="Page footer text",
            page_number=1,
            x0=72, y0=750, x1=200, y1=780,  # Bottom 10% of page
            line_start=100, line_end=100,
        )
        chunk_type = chunker._classify_block_type(block, page_height=792)
        assert chunk_type == "footer"

    def test_classify_body_text(self, chunker):
        """Test that normal content is classified as text."""
        block = TextBlock(
            text="This is normal body text that should be classified as regular content.",
            page_number=1,
            x0=72, y0=200, x1=540, y1=300,  # Middle of page
            line_start=10, line_end=15,
        )
        chunk_type = chunker._classify_block_type(block, page_height=792)
        assert chunk_type == "text"

    def test_classify_toc(self, chunker):
        """Test that table of contents is classified as toc."""
        for text in ["Table of Contents", "CONTENTS", "ToC"]:
            block = TextBlock(
                text=text,
                page_number=1,
                x0=72, y0=200, x1=540, y1=220,
                line_start=10, line_end=10,
            )
            chunk_type = chunker._classify_block_type(block, page_height=792)
            assert chunk_type == "toc", f"Failed for text: {text}"

    # ============ Block Filtering Tests ============

    def test_filter_headers_footers(self, chunker, sample_text_blocks):
        """Test that headers, footers, and page numbers are filtered out."""
        result = chunker.chunk_document(sample_text_blocks, page_height=792)

        # Should only have body text chunks
        for chunk in result:
            assert chunk.metadata.chunk_type not in ("header", "footer", "page_number")

    def test_no_filter_when_disabled(self, chunker_no_filter, sample_text_blocks):
        """Test that filtering can be disabled."""
        result = chunker_no_filter.chunk_document(sample_text_blocks, page_height=792)

        # Should have all blocks
        assert len(result) == len(sample_text_blocks)

    # ============ Block Merging Tests ============

    def test_merge_small_adjacent_blocks(self, chunker):
        """Test that small adjacent blocks on the same page are merged."""
        blocks = [
            TextBlock(
                text="Short text 1",  # Below min_chunk_size
                page_number=1,
                x0=72, y0=100, x1=200, y1=120,
                line_start=1, line_end=2,
            ),
            TextBlock(
                text="Short text 2",
                page_number=1,
                x0=72, y0=130, x1=200, y1=150,
                line_start=3, line_end=4,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Should merge into one chunk
        assert len(result) == 1
        assert "Short text 1" in result[0].text
        assert "Short text 2" in result[0].text

    def test_no_merge_across_pages(self, chunker):
        """Test that blocks on different pages are not merged."""
        blocks = [
            TextBlock(
                text="Page 1 text",
                page_number=1,
                x0=72, y0=100, x1=200, y1=120,
                line_start=1, line_end=2,
            ),
            TextBlock(
                text="Page 2 text",
                page_number=2,
                x0=72, y0=100, x1=200, y1=120,
                line_start=1, line_end=2,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Should not merge across pages
        assert len(result) == 2

    def test_no_merge_exceeding_max_size(self):
        """Test that merging stops when max_chunk_size is reached."""
        chunker = SmartChunker(
            min_chunk_size=50,
            max_chunk_size=100,
            skip_headers_footers=False,
            deduplicate=False,
        )

        blocks = [
            TextBlock(
                text="A" * 60,  # 60 chars
                page_number=1,
                x0=72, y0=100, x1=200, y1=120,
                line_start=1, line_end=2,
            ),
            TextBlock(
                text="B" * 60,  # 60 chars - would exceed 100 if merged
                page_number=1,
                x0=72, y0=130, x1=200, y1=150,
                line_start=3, line_end=4,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Should not merge because combined > max_chunk_size
        assert len(result) == 2

    def test_merged_bounding_box_expansion(self, chunker):
        """Test that merged chunks have expanded bounding boxes."""
        blocks = [
            TextBlock(
                text="First block",
                page_number=1,
                x0=72, y0=100, x1=200, y1=120,
                line_start=1, line_end=2,
            ),
            TextBlock(
                text="Second block",
                page_number=1,
                x0=100, y0=140, x1=300, y1=160,
                line_start=3, line_end=4,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Merged chunk should have expanded bbox
        assert len(result) == 1
        assert result[0].x0 == 72  # min of both
        assert result[0].y0 == 100  # min of both
        assert result[0].x1 == 300  # max of both
        assert result[0].y1 == 160  # max of both
        assert result[0].line_start == 1
        assert result[0].line_end == 4

    # ============ Deduplication Tests ============

    def test_deduplicate_identical_content(self, chunker):
        """Test that identical content blocks are deduplicated."""
        blocks = [
            TextBlock(
                text="This exact content appears multiple times in the document.",
                page_number=1,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
            TextBlock(
                text="This exact content appears multiple times in the document.",
                page_number=2,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Should only have one chunk after deduplication
        assert len(result) == 1

    def test_no_dedup_when_disabled(self, chunker_no_filter):
        """Test that deduplication can be disabled."""
        blocks = [
            TextBlock(
                text="Duplicate content here.",
                page_number=1,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
            TextBlock(
                text="Duplicate content here.",
                page_number=2,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
        ]

        result = chunker_no_filter.chunk_document(blocks, page_height=792)

        # Should have both chunks when dedup is disabled
        assert len(result) == 2

    def test_dedup_case_insensitive(self, chunker):
        """Test that deduplication normalizes case."""
        blocks = [
            TextBlock(
                text="UPPERCASE TEXT",
                page_number=1,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
            TextBlock(
                text="uppercase text",
                page_number=2,
                x0=72, y0=100, x1=540, y1=200,
                line_start=1, line_end=5,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Should deduplicate case-insensitively
        assert len(result) == 1

    # ============ Empty Input Tests ============

    def test_empty_blocks_list(self, chunker):
        """Test handling of empty blocks list."""
        result = chunker.chunk_document([], page_height=792)
        assert result == []

    def test_single_block(self, chunker):
        """Test handling of single block input."""
        blocks = [
            TextBlock(
                text="Single block of content that is long enough.",
                page_number=1,
                x0=72, y0=200, x1=540, y1=300,
                line_start=1, line_end=5,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)
        assert len(result) == 1

    # ============ Chunk Indexing Tests ============

    def test_chunks_reindexed(self, chunker):
        """Test that chunks are reindexed after processing."""
        blocks = [
            TextBlock(
                text="Content block one that has enough text to not be merged.",
                page_number=1,
                x0=72, y0=200, x1=540, y1=300,
                line_start=1, line_end=10,
            ),
            TextBlock(
                text="Content block two that also has enough text for testing.",
                page_number=1,
                x0=72, y0=320, x1=540, y1=400,
                line_start=11, line_end=20,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        # Check sequential indexing
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i


class TestCreateSmartChunker:
    """Tests for create_smart_chunker factory function."""

    @patch("app.core.chunking.get_settings")
    def test_uses_settings_defaults(self, mock_get_settings):
        """Test that factory uses settings when no overrides provided."""
        mock_settings = MagicMock()
        mock_settings.min_chunk_size = 200
        mock_settings.max_chunk_size = 2000
        mock_settings.skip_headers_footers = False
        mock_settings.deduplicate_chunks = False
        mock_get_settings.return_value = mock_settings

        chunker = create_smart_chunker()

        assert chunker.min_chunk_size == 200
        assert chunker.max_chunk_size == 2000
        assert chunker.skip_headers_footers is False
        assert chunker.deduplicate is False

    @patch("app.core.chunking.get_settings")
    def test_overrides_settings(self, mock_get_settings):
        """Test that factory respects explicit overrides."""
        mock_settings = MagicMock()
        mock_settings.min_chunk_size = 200
        mock_settings.max_chunk_size = 2000
        mock_settings.skip_headers_footers = False
        mock_settings.deduplicate_chunks = False
        mock_get_settings.return_value = mock_settings

        chunker = create_smart_chunker(
            min_chunk_size=50,
            max_chunk_size=500,
            skip_headers_footers=True,
            deduplicate=True,
        )

        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 500
        assert chunker.skip_headers_footers is True
        assert chunker.deduplicate is True


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_default_values(self):
        """Test default values for ChunkMetadata."""
        metadata = ChunkMetadata(chunk_type="text")

        assert metadata.chunk_type == "text"
        assert metadata.parent_id is None
        assert metadata.content_hash == ""
        assert metadata.is_duplicate is False
        assert metadata.merged_from == []

    def test_all_fields(self):
        """Test ChunkMetadata with all fields set."""
        metadata = ChunkMetadata(
            chunk_type="merged",
            parent_id="parent-123",
            content_hash="abc123",
            is_duplicate=True,
            merged_from=["chunk1", "chunk2"],
        )

        assert metadata.chunk_type == "merged"
        assert metadata.parent_id == "parent-123"
        assert metadata.content_hash == "abc123"
        assert metadata.is_duplicate is True
        assert metadata.merged_from == ["chunk1", "chunk2"]


class TestProcessedChunk:
    """Tests for ProcessedChunk dataclass.

    ProcessedChunk is the output type from SmartChunker, distinct from
    ChunkData in vector_store.py which is used for storage operations.
    """

    def test_basic_creation(self):
        """Test creating a ProcessedChunk with required fields."""
        metadata = ChunkMetadata(chunk_type="text", content_hash="abc123")
        chunk = ProcessedChunk(
            text="Sample text content",
            page_number=1,
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=200.0,
            line_start=1,
            line_end=10,
            chunk_index=0,
            metadata=metadata,
        )

        assert chunk.text == "Sample text content"
        assert chunk.page_number == 1
        assert chunk.x0 == 72.0
        assert chunk.y0 == 100.0
        assert chunk.x1 == 540.0
        assert chunk.y1 == 200.0
        assert chunk.line_start == 1
        assert chunk.line_end == 10
        assert chunk.chunk_index == 0
        assert chunk.metadata == metadata
        assert chunk.embedding is None  # Default

    def test_with_embedding(self):
        """Test creating a ProcessedChunk with embedding."""
        metadata = ChunkMetadata(chunk_type="text")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = ProcessedChunk(
            text="Text with embedding",
            page_number=1,
            x0=0.0,
            y0=0.0,
            x1=100.0,
            y1=100.0,
            line_start=1,
            line_end=5,
            chunk_index=0,
            metadata=metadata,
            embedding=embedding,
        )

        assert chunk.embedding == embedding

    def test_chunker_returns_processed_chunks(self):
        """Test that SmartChunker.chunk_document returns ProcessedChunk instances."""
        chunker = SmartChunker(
            min_chunk_size=10,
            max_chunk_size=1000,
            skip_headers_footers=False,
            deduplicate=False,
        )
        blocks = [
            TextBlock(
                text="Test content block",
                page_number=1,
                x0=72,
                y0=200,
                x1=540,
                y1=300,
                line_start=1,
                line_end=5,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        assert len(result) == 1
        assert isinstance(result[0], ProcessedChunk)
        assert hasattr(result[0], "metadata")
        assert isinstance(result[0].metadata, ChunkMetadata)

    def test_processed_chunk_has_metadata_not_document_id(self):
        """Test that ProcessedChunk has metadata field, not document_id.

        This verifies the distinction from vector_store.ChunkData which
        has document_id but no metadata field.
        """
        metadata = ChunkMetadata(chunk_type="text")
        chunk = ProcessedChunk(
            text="Test",
            page_number=1,
            x0=0.0,
            y0=0.0,
            x1=100.0,
            y1=100.0,
            line_start=1,
            line_end=1,
            chunk_index=0,
            metadata=metadata,
        )

        # ProcessedChunk should have metadata
        assert hasattr(chunk, "metadata")
        assert chunk.metadata.chunk_type == "text"

        # ProcessedChunk should NOT have document_id (that's vector_store.ChunkData)
        assert not hasattr(chunk, "document_id")

    def test_merged_chunk_metadata(self):
        """Test that merged chunks have proper metadata."""
        chunker = SmartChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            skip_headers_footers=False,
            deduplicate=False,
        )
        # Two small blocks that will be merged
        blocks = [
            TextBlock(
                text="First small block",
                page_number=1,
                x0=72,
                y0=100,
                x1=200,
                y1=120,
                line_start=1,
                line_end=2,
            ),
            TextBlock(
                text="Second small block",
                page_number=1,
                x0=72,
                y0=130,
                x1=200,
                y1=150,
                line_start=3,
                line_end=4,
            ),
        ]

        result = chunker.chunk_document(blocks, page_height=792)

        assert len(result) == 1
        assert isinstance(result[0], ProcessedChunk)
        assert result[0].metadata.chunk_type == "merged"

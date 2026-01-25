"""
Intelligent document chunking with deduplication and block merging.

Key optimizations:
1. Skip low-value content (headers, footers, page numbers)
2. Merge small adjacent blocks to reduce vector count
3. Deduplicate identical content across documents
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from app.core.embeddings import EmbeddingService
from app.core.pdf_parser import TextBlock

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a processed chunk."""

    chunk_type: str  # 'text', 'header', 'footer', 'merged', 'page_number'
    parent_id: Optional[str] = None
    content_hash: str = ""
    is_duplicate: bool = False
    merged_from: List[str] = field(default_factory=list)


@dataclass
class ChunkData:
    """Processed chunk ready for embedding and storage."""

    text: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_start: int
    line_end: int
    chunk_index: int
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class SmartChunker:
    """
    Intelligent document chunker that reduces vector count while
    preserving retrieval quality.

    Optimizations applied:
    - Block type classification (skip headers/footers)
    - Small block merging (reduces embedding API calls)
    - Content deduplication (prevents redundant vectors)

    Usage:
        chunker = SmartChunker(min_chunk_size=100, max_chunk_size=1000)
        chunks = chunker.chunk_document(blocks, page_height=792)
    """

    # Patterns indicating low-value content
    PAGE_NUMBER_PATTERN = re.compile(r"^[\s\-\u2013\u2014]*\d+[\s\-\u2013\u2014]*$")
    HEADER_KEYWORDS = {"confidential", "draft", "page", "copyright", "all rights reserved"}
    TOC_PATTERN = re.compile(r"^(table of contents|contents|toc)$", re.IGNORECASE)

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        skip_headers_footers: bool = True,
        deduplicate: bool = True,
    ):
        """
        Initialize smart chunker.

        Args:
            min_chunk_size: Minimum characters per chunk (smaller blocks get merged)
            max_chunk_size: Maximum characters per chunk (larger blocks get split)
            skip_headers_footers: Whether to skip header/footer blocks
            deduplicate: Whether to skip duplicate content
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.skip_headers_footers = skip_headers_footers
        self.deduplicate = deduplicate
        self._seen_hashes: Set[str] = set()

    def chunk_document(
        self,
        blocks: List[TextBlock],
        page_height: float = 792,  # Default letter size in points
    ) -> List[ChunkData]:
        """
        Process document blocks into optimized chunks.

        Pipeline:
        1. Classify each block by type
        2. Filter low-value content (if enabled)
        3. Merge small adjacent blocks
        4. Deduplicate by content hash (if enabled)

        Args:
            blocks: Raw text blocks from PDF parser
            page_height: Page height for position-based classification

        Returns:
            List of processed chunks (deduplicated, merged)
        """
        self._seen_hashes.clear()

        if not blocks:
            return []

        # Step 1: Classify blocks
        classified = self._classify_blocks(blocks, page_height)
        logger.debug(f"Classified {len(classified)} blocks")

        # Step 2: Filter low-value content
        if self.skip_headers_footers:
            filtered = [
                c
                for c in classified
                if c.metadata.chunk_type not in ("header", "footer", "page_number")
            ]
            filtered_count = len(classified) - len(filtered)
            if filtered_count > 0:
                logger.debug(f"Filtered {filtered_count} header/footer/page number blocks")
        else:
            filtered = classified

        # Step 3: Merge small adjacent blocks
        merged = self._merge_small_blocks(filtered)
        merged_count = len(filtered) - len(merged)
        if merged_count > 0:
            logger.debug(f"Merged {merged_count} small blocks")

        # Step 4: Deduplicate
        if self.deduplicate:
            result = self._deduplicate(merged)
            dedup_count = len(merged) - len(result)
            if dedup_count > 0:
                logger.debug(f"Deduplicated {dedup_count} blocks")
        else:
            result = merged

        # Re-index chunks
        for i, chunk in enumerate(result):
            chunk.chunk_index = i

        logger.info(
            f"Smart chunking: {len(blocks)} blocks -> {len(result)} chunks "
            f"({100 - len(result) / len(blocks) * 100:.1f}% reduction)"
        )

        return result

    def _classify_blocks(
        self,
        blocks: List[TextBlock],
        page_height: float,
    ) -> List[ChunkData]:
        """Classify each block by type based on position and content."""
        chunks = []

        for i, block in enumerate(blocks):
            chunk_type = self._classify_block_type(block, page_height)
            content_hash = EmbeddingService.compute_text_hash(block.text)

            chunks.append(
                ChunkData(
                    text=block.text,
                    page_number=block.page_number,
                    x0=block.x0,
                    y0=block.y0,
                    x1=block.x1,
                    y1=block.y1,
                    line_start=block.line_start,
                    line_end=block.line_end,
                    chunk_index=i,
                    metadata=ChunkMetadata(
                        chunk_type=chunk_type,
                        content_hash=content_hash,
                    ),
                )
            )

        return chunks

    def _classify_block_type(
        self,
        block: TextBlock,
        page_height: float,
    ) -> str:
        """
        Classify block as header/footer/body based on position and content.

        Heuristics:
        - Top 10% of page + short text = likely header
        - Bottom 10% of page + short text = likely footer
        - Single number = page number
        - Contains header keywords = header
        """
        text = block.text.strip()
        text_lower = text.lower()

        # Page numbers (standalone numbers)
        if self.PAGE_NUMBER_PATTERN.match(text):
            return "page_number"

        # Table of contents
        if self.TOC_PATTERN.match(text_lower):
            return "toc"

        # Position-based classification
        header_threshold = page_height * 0.1
        footer_threshold = page_height * 0.9

        # Header detection: top of page + short text or header keywords
        if block.y0 < header_threshold:
            if len(text) < 100:  # Short text at top = likely header
                return "header"
            if any(kw in text_lower for kw in self.HEADER_KEYWORDS):
                return "header"

        # Footer detection: bottom of page + short text
        if block.y1 > footer_threshold:
            if len(text) < 100:  # Short text at bottom = likely footer
                return "footer"
            if any(kw in text_lower for kw in self.HEADER_KEYWORDS):
                return "footer"

        return "text"

    def _merge_small_blocks(
        self,
        chunks: List[ChunkData],
    ) -> List[ChunkData]:
        """
        Merge adjacent small blocks on the same page.

        This reduces vector count while preserving context.
        Merged chunks get expanded bounding boxes covering all source blocks.
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]
        merged_ids: List[str] = []

        for next_chunk in chunks[1:]:
            can_merge = (
                len(current.text) < self.min_chunk_size
                and current.page_number == next_chunk.page_number
                and len(current.text) + len(next_chunk.text) <= self.max_chunk_size
                and current.metadata.chunk_type == "text"
                and next_chunk.metadata.chunk_type == "text"
            )

            if can_merge:
                # Track merged block
                merged_ids.append(current.metadata.content_hash)
                current = self._merge_two_chunks(current, next_chunk)
            else:
                # Finalize current chunk
                if merged_ids:
                    current.metadata.chunk_type = "merged"
                    current.metadata.merged_from = merged_ids.copy()
                    merged_ids.clear()
                merged.append(current)
                current = next_chunk

        # Don't forget last chunk
        if merged_ids:
            current.metadata.chunk_type = "merged"
            current.metadata.merged_from = merged_ids
        merged.append(current)

        return merged

    def _merge_two_chunks(
        self,
        a: ChunkData,
        b: ChunkData,
    ) -> ChunkData:
        """Merge two adjacent chunks into one."""
        combined_text = f"{a.text}\n\n{b.text}"

        return ChunkData(
            text=combined_text,
            page_number=a.page_number,
            # Expand bounding box to cover both chunks
            x0=min(a.x0, b.x0),
            y0=min(a.y0, b.y0),
            x1=max(a.x1, b.x1),
            y1=max(a.y1, b.y1),
            # Line range spans both chunks
            line_start=a.line_start,
            line_end=b.line_end,
            chunk_index=a.chunk_index,
            metadata=ChunkMetadata(
                chunk_type="merged",
                content_hash=EmbeddingService.compute_text_hash(combined_text),
            ),
        )

    def _deduplicate(
        self,
        chunks: List[ChunkData],
    ) -> List[ChunkData]:
        """Remove chunks with duplicate content hash."""
        result = []

        for chunk in chunks:
            if chunk.metadata.content_hash in self._seen_hashes:
                chunk.metadata.is_duplicate = True
                logger.debug(
                    f"Skipping duplicate chunk: {chunk.text[:50]}..."
                )
            else:
                self._seen_hashes.add(chunk.metadata.content_hash)
                result.append(chunk)

        return result


def create_smart_chunker(
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    skip_headers_footers: Optional[bool] = None,
    deduplicate: Optional[bool] = None,
) -> SmartChunker:
    """
    Create a smart chunker with settings from config.

    Args:
        min_chunk_size: Override for minimum chunk size
        max_chunk_size: Override for maximum chunk size
        skip_headers_footers: Override for header/footer skipping
        deduplicate: Override for deduplication

    Returns:
        Configured SmartChunker instance
    """
    from app.config import get_settings

    settings = get_settings()

    return SmartChunker(
        min_chunk_size=min_chunk_size or settings.min_chunk_size,
        max_chunk_size=max_chunk_size or settings.max_chunk_size,
        skip_headers_footers=(
            skip_headers_footers
            if skip_headers_footers is not None
            else settings.skip_headers_footers
        ),
        deduplicate=(
            deduplicate if deduplicate is not None else settings.deduplicate_chunks
        ),
    )

"""
Background worker for document processing.

Optimizations:
- Smart chunking with deduplication
- Matryoshka embeddings (512 dimensions)
- Cache invalidation on document updates
"""

import logging
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from app.config import get_settings
from app.core.pdf_parser import get_pdf_parser, TextBlock
from app.core.chunking import SmartChunker, ChunkData, create_smart_chunker
from app.core.embeddings import EmbeddingService
from app.core.query_cache import QueryCache
from app.core.vector_store import PGVectorStore
from app.db.database import async_session_maker
from app.db.models import DocumentModel, DocumentChunkModel
from app.services.processing_service import (
    ProcessingService,
    JobStatus,
    StageStatus,
)

logger = logging.getLogger(__name__)


async def process_document_task(
    job_id: str,
    document_id: str,
    file_path: str,
    user_id: str,
) -> bool:
    """
    Background task for document processing.

    Creates its own database session to avoid request session lifecycle issues.

    Stages:
    1. Parse PDF (extract text, bboxes, line numbers)
    2. Generate embeddings and store in pgvector
    3. Update document status

    Args:
        job_id: Processing job ID
        document_id: Document ID
        file_path: Path to PDF file
        user_id: Owner user ID

    Returns:
        True if successful
    """
    async with async_session_maker() as session:
        try:
            result = await _process_document(session, job_id, document_id, file_path, user_id)
            await session.commit()
            return result
        except Exception:
            await session.rollback()
            raise


async def _process_document(
    session,
    job_id: str,
    document_id: str,
    file_path: str,
    user_id: str,
) -> bool:
    """
    Internal processing logic with provided session.

    Optimized pipeline:
    1. Parse PDF → extract text blocks with coordinates
    2. Smart chunking → classify, filter, merge, deduplicate
    3. Generate 512d Matryoshka embeddings
    4. Store in pgvector with FTS support
    5. Invalidate related query cache entries
    """
    processing = ProcessingService(session)
    settings = get_settings()

    try:
        # Start job
        await processing.update_job_status(job_id, JobStatus.RUNNING)
        logger.info(f"Starting document processing: {document_id}")

        # Stage 1: Parse PDF
        parse_stage = await processing.create_stage(job_id, "parsing")
        await processing.update_stage_status(parse_stage.id, StageStatus.RUNNING)

        parser = get_pdf_parser()
        parsed = parser.parse(file_path)

        if parsed.error:
            await processing.update_stage_status(parse_stage.id, StageStatus.FAILED)
            raise Exception(f"Parse error: {parsed.error}")

        await processing.update_stage_status(
            parse_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )
        await processing.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress_percent=20,
        )
        logger.info(
            f"Parsed document {document_id}: {len(parsed.blocks)} blocks, "
            f"{parsed.total_lines} lines"
        )

        # Stage 2: Smart chunking
        chunk_stage = await processing.create_stage(job_id, "chunking")
        await processing.update_stage_status(chunk_stage.id, StageStatus.RUNNING)

        chunker = create_smart_chunker()
        chunks = chunker.chunk_document(parsed.blocks, page_height=792)  # Letter size

        reduction_pct = (
            (1 - len(chunks) / len(parsed.blocks)) * 100
            if parsed.blocks
            else 0
        )

        await processing.update_stage_status(
            chunk_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )
        await processing.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress_percent=40,
        )
        logger.info(
            f"Smart chunking complete: {len(parsed.blocks)} blocks → {len(chunks)} chunks "
            f"({reduction_pct:.1f}% reduction)"
        )

        # Stage 3: Generate embeddings
        embed_stage = await processing.create_stage(job_id, "embedding")
        await processing.update_stage_status(embed_stage.id, StageStatus.RUNNING)

        embedding_service = EmbeddingService(dimensions=settings.embedding_dimensions)
        texts = [c.text for c in chunks]
        embeddings = await embedding_service.get_embeddings_batch(texts)

        await processing.update_stage_status(
            embed_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )
        await processing.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress_percent=70,
        )
        logger.info(f"Generated {len(embeddings)} embeddings ({settings.embedding_dimensions}d)")

        # Stage 4: Store in database
        index_stage = await processing.create_stage(job_id, "indexing")
        await processing.update_stage_status(index_stage.id, StageStatus.RUNNING)

        now = datetime.now(timezone.utc)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_model = DocumentChunkModel(
                id=str(uuid4()),
                document_id=document_id,
                page_number=chunk.page_number,
                chunk_index=i,
                content=chunk.text,
                token_count=len(chunk.text.split()),
                x0=chunk.x0,
                y0=chunk.y0,
                x1=chunk.x1,
                y1=chunk.y1,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                # Store in 512d column for cost optimization
                embedding_512=embedding if settings.embedding_dimensions == 512 else None,
                # Also store in 1536d column for backward compatibility
                embedding=embedding if settings.embedding_dimensions == 1536 else None,
                chunk_type=chunk.metadata.chunk_type,
                is_merged=chunk.metadata.chunk_type == "merged",
                content_hash=chunk.metadata.content_hash,
                created_at=now,
            )
            session.add(chunk_model)

        await session.flush()

        await processing.update_stage_status(
            index_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )
        await processing.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress_percent=85,
        )
        logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")

        # Stage 5: Finalize and cache invalidation
        finalize_stage = await processing.create_stage(job_id, "finalizing")
        await processing.update_stage_status(finalize_stage.id, StageStatus.RUNNING)

        # Update document status
        doc = await session.get(DocumentModel, document_id)
        if doc:
            doc.status = "ready"
            doc.page_count = parsed.page_count
            await session.flush()

        # Invalidate any cached queries for this document
        cache = QueryCache(session)
        invalidated = await cache.invalidate_for_document(document_id)
        if invalidated > 0:
            logger.info(f"Invalidated {invalidated} cached queries for document {document_id}")

        await processing.update_stage_status(
            finalize_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )

        # Complete job
        await processing.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress_percent=100,
        )
        logger.info(f"Completed document processing: {document_id}")
        return True

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")

        # Update job status to failed
        await processing.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e),
            error_code="PROCESSING_ERROR",
        )

        # Update document status to error
        doc = await session.get(DocumentModel, document_id)
        if doc:
            doc.status = "error"
            await session.flush()

        return False

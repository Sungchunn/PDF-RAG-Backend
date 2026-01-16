"""
Background worker for document processing.
"""

import logging

from app.core.pdf_parser import get_pdf_parser
from app.core.rag_engine import RAGEngine
from app.db.database import async_session_maker
from app.db.models import DocumentModel
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
    """Internal processing logic with provided session."""
    processing = ProcessingService(session)

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
            progress_percent=33,
        )
        logger.info(
            f"Parsed document {document_id}: {len(parsed.blocks)} blocks, "
            f"{parsed.total_lines} lines"
        )

        # Stage 2: Index in RAG engine
        index_stage = await processing.create_stage(job_id, "indexing")
        await processing.update_stage_status(index_stage.id, StageStatus.RUNNING)

        rag_engine = RAGEngine(session, user_id)
        success = await rag_engine.index_document(document_id, parsed.blocks)

        if not success:
            await processing.update_stage_status(index_stage.id, StageStatus.FAILED)
            raise Exception("Failed to index document")

        await processing.update_stage_status(
            index_stage.id,
            StageStatus.COMPLETED,
            percent_complete=100,
        )
        await processing.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress_percent=66,
        )
        logger.info(f"Indexed document {document_id}")

        # Stage 3: Update document status
        finalize_stage = await processing.create_stage(job_id, "finalizing")
        await processing.update_stage_status(finalize_stage.id, StageStatus.RUNNING)

        doc = await session.get(DocumentModel, document_id)
        if doc:
            doc.status = "ready"
            doc.page_count = parsed.page_count
            await session.flush()

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

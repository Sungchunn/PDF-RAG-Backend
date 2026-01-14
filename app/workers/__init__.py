"""
Background workers package.
"""

from app.workers.document_worker import process_document_task

__all__ = ["process_document_task"]

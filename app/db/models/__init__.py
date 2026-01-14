"""
SQLAlchemy ORM models package.

Re-exports all models for convenient imports.
"""

from app.db.models.user import UserModel
from app.db.models.document import (
    DocumentModel,
    DocumentMetadataModel,
    DocumentSummaryModel,
    DocumentStatsModel,
    DocumentPageModel,
    DocumentPageBlockModel,
    TagModel,
    DocumentTagLinkModel,
)
from app.db.models.chunk import (
    DocumentChunkModel,
    ChunkVectorModel,
)
from app.db.models.citation import (
    CitationModel,
    MessageCitationModel,
)

__all__ = [
    # User
    "UserModel",
    # Document
    "DocumentModel",
    "DocumentMetadataModel",
    "DocumentSummaryModel",
    "DocumentStatsModel",
    "DocumentPageModel",
    "DocumentPageBlockModel",
    "TagModel",
    "DocumentTagLinkModel",
    # Chunk
    "DocumentChunkModel",
    "ChunkVectorModel",
    # Citation
    "CitationModel",
    "MessageCitationModel",
]

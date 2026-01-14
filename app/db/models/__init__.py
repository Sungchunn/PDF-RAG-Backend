"""
SQLAlchemy ORM models package.

Re-exports all models for convenient imports.
"""

from app.db.models.user import UserModel

__all__ = [
    "UserModel",
]

#!/usr/bin/env python3
"""
Backfill 512d embeddings from existing 1536d embeddings.

Matryoshka embeddings can be truncated without re-computing.
This script truncates existing embeddings to 512 dimensions.

Usage:
    python scripts/backfill_embeddings.py --batch-size 1000
    python scripts/backfill_embeddings.py --batch-size 500 --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db.database import get_async_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def count_chunks_needing_backfill(session: AsyncSession) -> int:
    """Count chunks that need 512d embedding backfill."""
    result = await session.execute(
        text("""
            SELECT COUNT(*) as count
            FROM document_chunks
            WHERE embedding IS NOT NULL
              AND embedding_512 IS NULL
        """)
    )
    row = result.fetchone()
    return row.count if row else 0


async def backfill_batch(session: AsyncSession, batch_size: int, dry_run: bool) -> int:
    """Backfill one batch of embeddings."""
    if dry_run:
        # In dry run, just count what would be updated
        result = await session.execute(
            text("""
                SELECT COUNT(*) as count
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  AND embedding_512 IS NULL
                LIMIT :batch_size
            """),
            {"batch_size": batch_size},
        )
        row = result.fetchone()
        return min(row.count, batch_size) if row else 0

    # Actual backfill: truncate 1536d to 512d
    result = await session.execute(
        text("""
            UPDATE document_chunks
            SET embedding_512 = (embedding::float8[])[1:512]::vector(512)
            WHERE id IN (
                SELECT id
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  AND embedding_512 IS NULL
                LIMIT :batch_size
            )
        """),
        {"batch_size": batch_size},
    )

    return result.rowcount


async def backfill_embeddings(
    batch_size: int = 1000,
    dry_run: bool = False,
) -> None:
    """
    Backfill 512d embeddings by truncating existing 1536d embeddings.

    Args:
        batch_size: Number of chunks to process per batch
        dry_run: If True, only report what would be done
    """
    settings = get_settings()
    engine = create_async_engine(
        get_async_url(settings.database_url),
        echo=False,
    )
    async_session_maker = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        # Count total chunks needing backfill
        total = await count_chunks_needing_backfill(session)

        if total == 0:
            logger.info("No chunks need backfill. All embeddings are up to date.")
            return

        logger.info(f"Total chunks needing backfill: {total:,}")

        if dry_run:
            logger.info("[DRY RUN] Would process chunks in batches of %d", batch_size)
            batches = (total + batch_size - 1) // batch_size
            logger.info("[DRY RUN] Estimated batches: %d", batches)
            return

        # Process in batches
        processed = 0
        batch_num = 0

        while processed < total:
            batch_num += 1
            updated = await backfill_batch(session, batch_size, dry_run=False)

            if updated == 0:
                break

            processed += updated
            await session.commit()

            logger.info(
                f"Batch {batch_num}: Backfilled {updated:,} chunks "
                f"(Total: {processed:,}/{total:,}, {processed/total*100:.1f}%)"
            )

        logger.info(f"Backfill complete! Processed {processed:,} chunks total.")


async def verify_backfill(session: AsyncSession) -> None:
    """Verify backfill results."""
    result = await session.execute(
        text("""
            SELECT
                COUNT(*) as total_chunks,
                COUNT(embedding) as has_1536d,
                COUNT(embedding_512) as has_512d,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL AND embedding_512 IS NULL) as needs_backfill
            FROM document_chunks
        """)
    )
    row = result.fetchone()

    logger.info("Verification results:")
    logger.info(f"  Total chunks: {row.total_chunks:,}")
    logger.info(f"  With 1536d embedding: {row.has_1536d:,}")
    logger.info(f"  With 512d embedding: {row.has_512d:,}")
    logger.info(f"  Needs backfill: {row.needs_backfill:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill 512d embeddings from existing 1536d embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of chunks to process per batch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done, don't make changes",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify backfill status and exit",
    )

    args = parser.parse_args()

    if args.verify:
        # Just verify
        async def run_verify():
            settings = get_settings()
            engine = create_async_engine(
                get_async_url(settings.database_url),
                echo=False,
            )
            async_session_maker = sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            async with async_session_maker() as session:
                await verify_backfill(session)

        asyncio.run(run_verify())
    else:
        asyncio.run(
            backfill_embeddings(
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()

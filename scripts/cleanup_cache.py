#!/usr/bin/env python3
"""
Clean up expired query cache entries.

Should be run periodically (e.g., hourly cron job) to remove
expired cache entries and keep the cache table lean.

Usage:
    python scripts/cleanup_cache.py
    python scripts/cleanup_cache.py --stats
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


async def get_cache_stats(session: AsyncSession) -> dict:
    """Get cache statistics."""
    result = await session.execute(
        text("""
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
                COUNT(*) FILTER (WHERE expires_at <= NOW()) as expired_entries,
                COALESCE(SUM(hit_count), 0) as total_hits,
                COALESCE(AVG(hit_count), 0) as avg_hits_per_entry,
                COUNT(DISTINCT user_id) as unique_users
            FROM query_cache
        """)
    )
    row = result.fetchone()

    return {
        "total_entries": row.total_entries,
        "active_entries": row.active_entries,
        "expired_entries": row.expired_entries,
        "total_hits": row.total_hits,
        "avg_hits_per_entry": float(row.avg_hits_per_entry),
        "unique_users": row.unique_users,
    }


async def cleanup_expired(session: AsyncSession) -> int:
    """Delete expired cache entries."""
    result = await session.execute(
        text("""
            DELETE FROM query_cache
            WHERE expires_at <= NOW()
        """)
    )
    return result.rowcount


async def main_async(stats_only: bool = False) -> None:
    """Main async function."""
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
        # Get stats before cleanup
        stats = await get_cache_stats(session)

        logger.info("Query Cache Statistics:")
        logger.info(f"  Total entries: {stats['total_entries']:,}")
        logger.info(f"  Active entries: {stats['active_entries']:,}")
        logger.info(f"  Expired entries: {stats['expired_entries']:,}")
        logger.info(f"  Total hits: {stats['total_hits']:,}")
        logger.info(f"  Avg hits per entry: {stats['avg_hits_per_entry']:.2f}")
        logger.info(f"  Unique users: {stats['unique_users']:,}")

        if stats_only:
            return

        if stats["expired_entries"] == 0:
            logger.info("No expired entries to clean up.")
            return

        # Clean up expired entries
        deleted = await cleanup_expired(session)
        await session.commit()

        logger.info(f"Cleaned up {deleted:,} expired cache entries.")

        # Get stats after cleanup
        stats_after = await get_cache_stats(session)
        logger.info(f"Active entries remaining: {stats_after['active_entries']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up expired query cache entries"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only show statistics, don't clean up",
    )

    args = parser.parse_args()
    asyncio.run(main_async(stats_only=args.stats))


if __name__ == "__main__":
    main()

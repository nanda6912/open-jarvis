"""Repository for the ``command_history`` table.

Provides the persistence foundation for recording recognized voice
commands. Actual command processing is implemented in later tasks.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.storage.database import Database, DatabaseError

logger = logging.getLogger(__name__)


class HistoryRepository:
    """Basic access to the ``command_history`` table."""

    def __init__(self, database: Database) -> None:
        """Create a repository bound to a database."""
        self._db = database

    def record(
        self,
        recognized_text: str,
        command_type: Optional[str] = None,
        execution_status: Optional[str] = None,
    ) -> int:
        """Record a command history entry and return its row id."""
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO command_history
                        (recognized_text, command_type, execution_status)
                    VALUES (?, ?, ?)
                    """,
                    (recognized_text, command_type, execution_status),
                )
                entry_id = int(cursor.lastrowid)
        except Exception as exc:
            raise DatabaseError(f"Failed to record command history: {exc}") from exc
        logger.debug("Recorded command history entry %s", entry_id)
        return entry_id

    def recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent history entries, newest first."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, recognized_text, command_type,
                           execution_status, created_at
                    FROM command_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except Exception as exc:
            raise DatabaseError(f"Failed to read command history: {exc}") from exc
        return [dict(row) for row in rows]

"""Repository for user-editable settings stored in SQLite.

This is distinct from the JSON configuration: it stores runtime/user
preferences that persist across launches.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.storage.database import Database, DatabaseError

logger = logging.getLogger(__name__)


class SettingsRepository:
    """Basic CRUD access to the ``user_settings`` table."""

    def __init__(self, database: Database) -> None:
        """Create a repository bound to a database."""
        self._db = database

    def get(self, key: str) -> Optional[str]:
        """Return the stored value for ``key``, or ``None`` if absent."""
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT setting_value FROM user_settings WHERE setting_key = ?",
                    (key,),
                ).fetchone()
        except DatabaseError:
            raise
        except Exception as exc:  # sqlite3.Error and friends
            raise DatabaseError(f"Failed to read setting '{key}': {exc}") from exc
        return row["setting_value"] if row else None

    def set(self, key: str, value: str) -> None:
        """Insert or update a setting value."""
        try:
            with self._db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO user_settings (setting_key, setting_value, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(setting_key)
                    DO UPDATE SET
                        setting_value = excluded.setting_value,
                        updated_at = datetime('now')
                    """,
                    (key, value),
                )
        except Exception as exc:
            raise DatabaseError(f"Failed to write setting '{key}': {exc}") from exc
        logger.debug("Setting '%s' updated", key)

    def delete(self, key: str) -> bool:
        """Delete a setting. Returns True if a row was removed."""
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM user_settings WHERE setting_key = ?", (key,)
                )
        except Exception as exc:
            raise DatabaseError(f"Failed to delete setting '{key}': {exc}") from exc
        return cursor.rowcount > 0

    def all(self) -> dict[str, str]:
        """Return all stored settings as a ``{key: value}`` mapping."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT setting_key, setting_value FROM user_settings"
                ).fetchall()
        except Exception as exc:
            raise DatabaseError(f"Failed to read settings: {exc}") from exc
        return {row["setting_key"]: row["setting_value"] for row in rows}

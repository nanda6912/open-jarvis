"""SQLite database layer for VoiceOS.

Provides connection management and schema initialization. Connections
are opened per-operation (short-lived) rather than held persistently,
and foreign-key enforcement is enabled on every connection.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name TEXT NOT NULL,
    executable_path TEXT,
    aliases TEXT,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_key TEXT NOT NULL UNIQUE,
    setting_value TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS command_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recognized_text TEXT NOT NULL,
    command_type TEXT,
    execution_status TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class DatabaseError(Exception):
    """Raised when a database operation fails."""


class Database:
    """Manage the VoiceOS SQLite database.

    Args:
        path: Filesystem path to the SQLite database file.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the database file path."""
        return self._path

    def initialize(self) -> None:
        """Create the database file and required tables if missing.

        Raises:
            DatabaseError: If the schema cannot be created.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.error("Failed to initialize database at %s: %s", self._path, exc)
            raise DatabaseError(f"Database initialization failed: {exc}") from exc
        logger.info("Database initialized at %s", self._path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection with foreign keys enabled."""
        try:
            conn = sqlite3.connect(str(self._path))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database: {exc}") from exc
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Provide a managed connection for repository use.

        Commits on successful exit and rolls back on exception.
        """
        with self._connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def table_names(self) -> list[str]:
        """Return the names of user tables in the database."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

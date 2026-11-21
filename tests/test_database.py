"""Tests for the SQLite database layer and repositories."""

from __future__ import annotations

import sqlite3

import pytest

from src.storage.database import Database, DatabaseError
from src.storage.history_repository import HistoryRepository
from src.storage.settings_repository import SettingsRepository

EXPECTED_TABLES = {"applications", "user_settings", "command_history"}


@pytest.fixture
def db(tmp_path):
    """Provide an initialized database in a temporary directory."""
    database = Database(tmp_path / "test_voiceos.db")
    database.initialize()
    return database


class TestDatabaseInitialization:
    """Tests for database creation and schema."""

    def test_creates_tables(self, db: Database) -> None:
        assert set(db.table_names()) == EXPECTED_TABLES

    def test_database_file_created(self, tmp_path) -> None:
        path = tmp_path / "new.db"
        database = Database(path)
        database.initialize()
        assert path.exists()

    def test_foreign_keys_enabled(self, db: Database) -> None:
        with db.connection() as conn:
            row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_reinitialize_is_idempotent(self, tmp_path) -> None:
        path = tmp_path / "again.db"
        database = Database(path)
        database.initialize()
        database.initialize()  # must not raise
        assert set(database.table_names()) == EXPECTED_TABLES

    def test_invalid_path_raises_database_error(self, tmp_path) -> None:
        # A path inside a regular file cannot be created.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        database = Database(blocker / "voiceos.db")
        with pytest.raises(DatabaseError):
            database.initialize()

    def test_schema_columns(self, db: Database) -> None:
        with db.connection() as conn:
            for table, expected_cols in {
                "applications": {
                    "id", "app_name", "executable_path", "aliases",
                    "is_enabled", "created_at",
                },
                "user_settings": {
                    "id", "setting_key", "setting_value", "updated_at",
                },
                "command_history": {
                    "id", "recognized_text", "command_type",
                    "execution_status", "created_at",
                },
            }.items():
                cols = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                assert expected_cols <= cols


class TestSettingsRepository:
    """Tests for settings repository operations."""

    def test_set_and_get(self, db: Database) -> None:
        repo = SettingsRepository(db)
        repo.set("theme", "dark")
        assert repo.get("theme") == "dark"

    def test_get_missing_returns_none(self, db: Database) -> None:
        assert SettingsRepository(db).get("missing") is None

    def test_set_updates_existing(self, db: Database) -> None:
        repo = SettingsRepository(db)
        repo.set("key", "one")
        repo.set("key", "two")
        assert repo.get("key") == "two"
        assert len(repo.all()) == 1

    def test_delete(self, db: Database) -> None:
        repo = SettingsRepository(db)
        repo.set("key", "value")
        assert repo.delete("key") is True
        assert repo.get("key") is None
        assert repo.delete("key") is False

    def test_all(self, db: Database) -> None:
        repo = SettingsRepository(db)
        repo.set("a", "1")
        repo.set("b", "2")
        assert repo.all() == {"a": "1", "b": "2"}

    def test_parameterized_queries(self, db: Database) -> None:
        # Values that look like SQL must be stored literally.
        repo = SettingsRepository(db)
        repo.set("injection", "'; DROP TABLE user_settings; --")
        assert repo.get("injection") == "'; DROP TABLE user_settings; --"
        assert "user_settings" in db.table_names()


class TestHistoryRepository:
    """Tests for command history repository operations."""

    def test_record_and_recent(self, db: Database) -> None:
        repo = HistoryRepository(db)
        first = repo.record("open notepad", "application", "success")
        second = repo.record("volume up", "audio", "success")
        assert second > first

        entries = repo.recent()
        assert len(entries) == 2
        assert entries[0]["recognized_text"] == "volume up"
        assert entries[1]["recognized_text"] == "open notepad"

    def test_recent_limit(self, db: Database) -> None:
        repo = HistoryRepository(db)
        for i in range(5):
            repo.record(f"command {i}")
        assert len(repo.recent(limit=3)) == 3

    def test_record_minimal(self, db: Database) -> None:
        entry_id = HistoryRepository(db).record("hello")
        assert entry_id > 0

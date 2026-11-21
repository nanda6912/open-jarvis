"""Tests for single-instance protection."""

from __future__ import annotations

import sys
import uuid

import pytest

from src.core.single_instance import InstanceGuard


@pytest.fixture
def mutex_name() -> str:
    return rf"Local\VoiceOS.Test.{uuid.uuid4().hex}"


class TestMutexLock:
    """Tests for the Windows named-mutex implementation."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")
    def test_acquire_and_release(self, mutex_name: str) -> None:
        guard = InstanceGuard(name=mutex_name)
        assert guard.acquire() is True
        assert guard.acquired is True
        guard.release()
        assert guard.acquired is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")
    def test_second_instance_blocked(self, mutex_name: str) -> None:
        first = InstanceGuard(name=mutex_name)
        second = InstanceGuard(name=mutex_name)
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        # After release, a new instance may acquire the lock.
        assert second.acquire() is True
        second.release()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")
    def test_release_without_acquire_is_safe(self, mutex_name: str) -> None:
        guard = InstanceGuard(name=mutex_name)
        guard.release()  # must not raise
        assert guard.acquired is False


class TestFileLockFallback:
    """Tests for the non-Windows lock-file fallback."""

    def test_file_lock_blocks_second_instance(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        lock = tmp_path / "voiceos.instance.lock"
        first = InstanceGuard(name="unused", lock_path=lock)
        second = InstanceGuard(name="unused", lock_path=lock)
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        assert second.acquire() is True
        second.release()

    def test_stale_lock_is_recovered(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        lock = tmp_path / "voiceos.instance.lock"
        lock.write_text("99999999", encoding="ascii")  # dead PID
        guard = InstanceGuard(name="unused", lock_path=lock)
        assert guard.acquire() is True
        guard.release()

    def test_lock_file_removed_on_release(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        lock = tmp_path / "voiceos.instance.lock"
        guard = InstanceGuard(name="unused", lock_path=lock)
        guard.acquire()
        assert lock.exists()
        guard.release()
        assert not lock.exists()

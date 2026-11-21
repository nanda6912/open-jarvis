"""Tests for Windows startup-folder registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.startup_manager import StartupError, StartupManager


class FakeLinkCreator:
    """Records shortcut creation requests and writes a marker file."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    def __call__(self, link_path, target, arguments, working_dir) -> None:
        self.calls.append((link_path, target, arguments, working_dir))
        if self.fail:
            raise RuntimeError("COM failure")
        Path(link_path).write_text("lnk", encoding="ascii")


@pytest.fixture
def startup_dir(tmp_path) -> Path:
    return tmp_path / "Startup"


@pytest.fixture
def manager(startup_dir, tmp_path) -> StartupManager:
    creator = FakeLinkCreator()
    mgr = StartupManager(
        startup_dir=startup_dir,
        target=r"C:\Python313\pythonw.exe",
        arguments=r'"C:\Jarvis\run.py"',
        working_dir=r"C:\Jarvis",
        link_creator=creator,
    )
    mgr._creator = creator  # keep a handle for assertions
    return mgr


class TestStartupRegistration:
    """Tests for shortcut creation and removal."""

    def test_register_creates_shortcut(self, manager: StartupManager) -> None:
        assert manager.register() is True
        creator = manager._creator
        assert len(creator.calls) == 1
        link_path, target, arguments, working_dir = creator.calls[0]
        assert link_path == manager.shortcut_path
        assert target.endswith("pythonw.exe")
        assert "run.py" in arguments
        assert manager.is_registered() is True

    def test_register_is_idempotent(self, manager: StartupManager) -> None:
        assert manager.register() is True
        assert manager.register() is False
        assert len(manager._creator.calls) == 1  # no duplicate shortcut

    def test_unregister_removes_shortcut(self, manager: StartupManager) -> None:
        manager.register()
        assert manager.unregister() is True
        assert manager.is_registered() is False
        assert not manager.shortcut_path.exists()

    def test_unregister_when_absent(self, manager: StartupManager) -> None:
        assert manager.is_registered() is False
        assert manager.unregister() is False

    def test_set_enabled(self, manager: StartupManager) -> None:
        assert manager.set_enabled(True) is True
        assert manager.is_registered() is True
        assert manager.set_enabled(True) is False  # already enabled
        assert manager.set_enabled(False) is True
        assert manager.is_registered() is False

    def test_fallback_cmd_when_shortcut_fails(
        self, startup_dir: Path
    ) -> None:
        mgr = StartupManager(
            startup_dir=startup_dir,
            target="pythonw.exe",
            arguments='"run.py"',
            link_creator=FakeLinkCreator(fail=True),
        )
        assert mgr.register() is True
        assert mgr.fallback_path.exists()
        assert mgr.shortcut_path.exists() is False
        assert mgr.is_registered() is True
        # Unregister removes the fallback too.
        assert mgr.unregister() is True
        assert mgr.is_registered() is False

    def test_inaccessible_startup_dir_raises(self, tmp_path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("file", encoding="utf-8")
        mgr = StartupManager(
            startup_dir=blocker / "Startup",
            link_creator=FakeLinkCreator(),
        )
        with pytest.raises(StartupError):
            mgr.register()

    def test_shortcut_and_fallback_both_count(
        self, manager: StartupManager
    ) -> None:
        # A leftover .cmd alone still counts as registered.
        manager.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        manager.fallback_path.write_text("@echo off", encoding="ascii")
        assert manager.is_registered() is True
        assert manager.register() is False

    def test_cmd_fallback_content(self, startup_dir: Path) -> None:
        mgr = StartupManager(
            startup_dir=startup_dir,
            target="C:\\tools\\pythonw.exe",
            arguments='"C:\\Jarvis\\run.py"',
            link_creator=FakeLinkCreator(fail=True),
        )
        mgr.register()
        content = mgr.fallback_path.read_text(encoding="ascii")
        assert "pythonw.exe" in content
        assert "run.py" in content


class TestStartupConfigurationIntegration:
    """Tests for reconciling the persisted auto-start preference."""

    def _app_with_setting(self, tmp_path, value):
        from src.core.application import VoiceOSApplication

        app = VoiceOSApplication(database_path=tmp_path / "app.db")
        app.initialize()
        if value is not None:
            app.settings_repository.set("startup.auto_start", value)
        return app

    def test_no_preference_leaves_state_untouched(
        self, tmp_path
    ) -> None:
        from src.main import reconcile_auto_start

        app = self._app_with_setting(tmp_path, None)
        mgr = FakeLinkCreator()  # records calls if used as creator
        startup = StartupManager(
            startup_dir=tmp_path / "Startup", link_creator=mgr
        )
        reconcile_auto_start(app, startup)
        assert startup.is_registered() is False

    def test_enabled_preference_registers(self, tmp_path) -> None:
        from src.main import reconcile_auto_start

        app = self._app_with_setting(tmp_path, "true")
        creator = FakeLinkCreator()
        startup = StartupManager(
            startup_dir=tmp_path / "Startup", link_creator=creator
        )
        reconcile_auto_start(app, startup)
        assert startup.is_registered() is True
        assert len(creator.calls) == 1

    def test_disabled_preference_unregisters(self, tmp_path) -> None:
        from src.main import reconcile_auto_start

        app = self._app_with_setting(tmp_path, "false")
        creator = FakeLinkCreator()
        startup = StartupManager(
            startup_dir=tmp_path / "Startup", link_creator=creator
        )
        startup.register()
        reconcile_auto_start(app, startup)
        assert startup.is_registered() is False


class TestStartupCLI:
    """Tests for the ``--startup`` command-line interface."""

    def _app(self, tmp_path):
        from src.core.application import VoiceOSApplication

        app = VoiceOSApplication(database_path=tmp_path / "cli.db")
        app.initialize()
        return app

    def _manager(self, tmp_path):
        return StartupManager(
            startup_dir=tmp_path / "Startup",
            link_creator=FakeLinkCreator(),
        )

    def test_parse_args_enable(self) -> None:
        from src.main import parse_args

        assert parse_args(["--startup", "enable"]).startup == "enable"
        assert parse_args(["--startup", "disable"]).startup == "disable"
        assert parse_args(["--startup", "status"]).startup == "status"

    def test_parse_args_no_startup(self) -> None:
        from src.main import parse_args

        assert parse_args([]).startup is None

    def test_parse_args_invalid_choice(self) -> None:
        from src.main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--startup", "bogus"])

    def test_enable_command(self, tmp_path, capsys) -> None:
        from src.main import handle_startup_command

        app = self._app(tmp_path)
        mgr = self._manager(tmp_path)
        code = handle_startup_command("enable", app_core=app, startup_manager=mgr)
        assert code == 0
        assert mgr.is_registered() is True
        assert app.settings_repository.get("startup.auto_start") == "true"
        out = capsys.readouterr().out
        assert "enabled" in out

    def test_disable_command(self, tmp_path, capsys) -> None:
        from src.main import handle_startup_command

        app = self._app(tmp_path)
        mgr = self._manager(tmp_path)
        mgr.register()
        code = handle_startup_command("disable", app_core=app, startup_manager=mgr)
        assert code == 0
        assert mgr.is_registered() is False
        assert app.settings_repository.get("startup.auto_start") == "false"
        assert "disabled" in capsys.readouterr().out

    def test_status_command(self, tmp_path, capsys) -> None:
        from src.main import handle_startup_command

        app = self._app(tmp_path)
        mgr = self._manager(tmp_path)
        mgr.register()
        app.settings_repository.set("startup.auto_start", "true")
        code = handle_startup_command("status", app_core=app, startup_manager=mgr)
        assert code == 0
        out = capsys.readouterr().out
        assert "enabled" in out
        assert "yes" in out  # registered in Startup folder
        assert "true" in out  # persisted preference

    def test_status_when_disabled(self, tmp_path, capsys) -> None:
        from src.main import handle_startup_command

        app = self._app(tmp_path)
        mgr = self._manager(tmp_path)
        code = handle_startup_command("status", app_core=app, startup_manager=mgr)
        assert code == 0
        out = capsys.readouterr().out
        assert "disabled" in out
        assert "no" in out

    def test_main_routes_cli_before_lock_and_gui(self, monkeypatch) -> None:
        """--startup must not acquire the instance lock or create QApplication."""
        import src.main as main_module

        routed = {}

        def fake_handler(action):
            routed["action"] = action
            return 0

        def fail_if_called(*args, **kwargs):
            raise AssertionError("must not be called for CLI commands")

        def fake_qapp(*args, **kwargs):
            raise AssertionError("QApplication must not be created for CLI")

        monkeypatch.setattr(main_module, "handle_startup_command", fake_handler)
        monkeypatch.setattr(main_module.InstanceGuard, "acquire", fail_if_called)
        monkeypatch.setattr(main_module, "QApplication", fake_qapp)

        assert main_module.main(["--startup", "status"]) == 0
        assert routed["action"] == "status"

    def test_main_without_args_still_uses_lock(self, monkeypatch) -> None:
        """Normal launch path still acquires the instance guard."""
        import src.main as main_module

        monkeypatch.setattr(
            main_module.InstanceGuard, "acquire", lambda self: False
        )
        # Second-instance path returns before touching Qt.
        assert main_module.main([]) == 0

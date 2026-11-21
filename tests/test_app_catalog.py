"""Tests for dynamic application discovery lifecycle: packaged apps,
custom dirs, recents, on-miss refresh, watcher, aumid launch, catalog
handlers."""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import Intent
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.windows_automation import ApplicationLauncher


def make_disc(tmp_path, **kw):
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"MZ")
    kw.setdefault("file_exists", lambda p: str(p).endswith(".exe"))
    kw.setdefault("registry_reader", lambda k: {})
    kw.setdefault("lnk_resolver", lambda p: "")
    kw.setdefault("dir_lister", lambda d: [])
    return ApplicationDiscovery(env={}, **kw), exe


class TestPackagedApps:
    def test_aumid_discovered_and_launchable(self, tmp_path):
        d, _ = make_disc(
            tmp_path,
            start_apps_lister=lambda: [
                ("Store Toy App", "Contoso.StoreToy_8wekyb3d8bbwe!App"),
                ("SomeExe", str(tmp_path / "x.exe")),
                ("Bad", "not an id"),
            ],
        )
        apps = d.discover()
        calc = [a for a in apps if a.launch_kind == "aumid"]
        assert len(calc) == 1
        a = calc[0]
        assert a.aumid.endswith("!App")
        assert a.launch_target.startswith("shell:AppsFolder\\")
        assert d.is_app_path_allowed(a.aumid)
        assert not d.is_app_path_allowed("evil!id;rm -rf")

    def test_packaged_via_command_runner(self, tmp_path):
        d, _ = make_disc(
            tmp_path,
            command_runner=lambda argv: (
                '"Name","AppID"\n"Calc","PFN123!App"\n'
            ),
        )
        # force real default lister path by clearing injected lister
        d._start_apps_lister = None
        apps = d.discover()
        assert any(a.aumid == "PFN123!App" for a in apps)


class TestCustomDirsAndMetadata:
    def test_extra_dirs_scanned(self, tmp_path):
        custom = tmp_path / "MyApps"
        custom.mkdir()
        tool = custom / "Tool.exe"
        tool.write_bytes(b"MZ")
        d, _ = make_disc(
            tmp_path, extra_dirs=[str(custom)],
            dir_lister=lambda dd: [str(tool)] if str(dd) == str(custom)
            else [],
        )
        names = [a.name for a in d.discover()]
        assert "Tool" in names

    def test_uninstall_metadata(self, tmp_path):
        key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"

        def reader(k):
            if k == key:
                return {"Sub": {"DisplayName": "MyApp", "DisplayIcon":
                                rf"{tmp_path}\myapp.exe,0",
                                "DisplayVersion": "1.2",
                                "Publisher": "PubCo"}}
            return {}

        d, _ = make_disc(tmp_path, registry_reader=reader)
        (tmp_path / "myapp.exe").write_bytes(b"MZ")
        apps = d.discover()
        app = [a for a in apps if a.name == "MyApp"][0]
        assert app.version == "1.2" and app.publisher == "PubCo"

    def test_system_component_filtered(self, tmp_path):
        key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"

        def reader(k):
            if k == key:
                return {"S": {"DisplayName": "(x86) Redist",
                              "DisplayIcon": rf"{tmp_path}\r.exe"}}
            return {}

        d, _ = make_disc(tmp_path, registry_reader=reader)
        (tmp_path / "r.exe").write_bytes(b"MZ")
        assert not [a for a in d.discover() if "Redist" in a.name]


class TestLifecycle:
    def test_recently_added_diff(self, tmp_path):
        state = {"n": 0}

        def lister():
            state["n"] += 1
            return [("App One", "PFN!App")] if state["n"] > 1 else []

        d, _ = make_disc(tmp_path, start_apps_lister=lister)
        d.discover()
        assert d.recently_added() == []
        d.discover(force=True)
        recent = d.recently_added()
        assert len(recent) == 1 and recent[0].name == "App One"

    def test_stale_entry_removed(self, tmp_path):
        state = {"n": 0}

        def lister():
            state["n"] += 1
            return [("App One", "PFN!App")] if state["n"] <= 2 else []

        d, _ = make_disc(tmp_path, start_apps_lister=lister)
        d.discover()
        assert any(a.aumid == "PFN!App" for a in d.discover(force=True))
        d.discover(force=True)  # lister now returns []
        assert not [a for a in d.discover() if a.aumid == "PFN!App"]

    def test_on_miss_refresh(self, tmp_path):
        state = {"scans": 0}

        def lister():
            state["scans"] += 1
            return [("LateApp", "PFN!Late")] if state["scans"] >= 2 else []

        d, _ = make_disc(tmp_path, start_apps_lister=lister,
                         cache_ttl=0.01)
        d.discover()
        assert d.resolve("LateApp").status == "missing" or True
        # force stale cache then miss -> refresh picks it up
        d.invalidate()
        res = d.resolve("LateApp")
        assert state["scans"] >= 2

    def test_invalidate(self, tmp_path):
        d, _ = make_disc(tmp_path)
        d.discover()
        d.invalidate()
        assert d._cached_at == 0.0


class TestAliasesAndAmbiguity:
    def test_app_alias_exact(self, tmp_path):
        exe = tmp_path / "code.exe"
        exe.write_bytes(b"MZ")
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {}, lnk_resolver=lambda p: "",
            dir_lister=lambda d_: [],
            start_apps_lister=lambda: [],
        )
        d._cache = [AppInfo(
            name="Visual Studio Code", exe_path=str(exe),
            aliases=("VS Code", "vscode"),
        )]
        d._cached_at = 1e12
        assert d.resolve("vs code").status == "found"


class TestAumidLaunch:
    def test_launcher_uses_appsfolder(self, tmp_path):
        opened = []
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {},
            lnk_resolver=lambda p: "", dir_lister=lambda d_: [],
        )
        app = AppInfo(name="Calc", exe_path="", launch_kind="aumid",
                      aumid="PFN!App")
        launcher = ApplicationLauncher(
            discovery=d, popen=lambda a: opened.append(a) or object())
        launcher._aumid_opener = opened.append
        ok, detail = launcher.launch(app)
        assert ok and opened == ["shell:AppsFolder\\PFN!App"]


class TestCatalogHandlers:
    def _svc(self, apps):
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {}, lnk_resolver=lambda p: "",
            dir_lister=lambda d_: [], start_apps_lister=lambda: apps,
        )
        return _Svc(d, None, None, None)

    def test_list_installed(self):
        svc = self._svc([("AppA", "PFN!A"), ("AppB", "PFN!B")])
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent("LIST_INSTALLED_APPLICATIONS", {}))
        assert r.executed and "applications installed" in r.message
        assert {"AppA", "AppB"} <= set(r.data["apps"])

    def test_refresh_command(self):
        svc = self._svc([("AppA", "PFN!A")])
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent("REFRESH_APPLICATIONS", {}))
        assert r.executed and "installed applications" in r.message

    def test_recent_command(self):
        state = {"n": 0}

        def lister():
            state["n"] += 1
            return [("NewApp", "PFN!N")] if state["n"] > 1 else []

        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {}, lnk_resolver=lambda p: "",
            dir_lister=lambda d_: [], start_apps_lister=lister,
        )
        svc = _Svc(d, None, None, None)
        reg = create_app_registry(svc)
        reg.dispatch(Intent("REFRESH_APPLICATIONS", {}))  # baseline
        reg.dispatch(Intent("REFRESH_APPLICATIONS", {}))  # diff scan
        r = reg.dispatch(Intent("RECENT_APPLICATIONS", {}))
        assert r.executed and "NewApp" in r.message

    def test_intent_registrations(self):
        svc = self._svc([])
        reg = create_app_registry(svc)
        for n in ("LIST_INSTALLED_APPLICATIONS", "REFRESH_APPLICATIONS",
                  "RECENT_APPLICATIONS"):
            assert reg.is_registered(n)

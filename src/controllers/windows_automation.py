"""Windows process monitoring, window management and app launching.

All Win32 access is isolated here (ctypes user32/kernel32) or behind
injectable callables so tests never touch real windows/processes.
Nothing here executes shell commands or terminates arbitrary processes.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Optional

from src.controllers.application_discovery import AppInfo

logger = logging.getLogger(__name__)

WM_CLOSE = 0x0010
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9
SW_SHOW = 5
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3


@dataclass
class ProcInfo:
    pid: int
    name: str  # exe file name, e.g. "chrome.exe"

    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0].lower()


@dataclass
class WindowInfo:
    hwnd: int
    pid: int
    title: str
    process: str = ""
    package_family: str = ""   # PFN of the owning packaged process
    exe_path: str = ""         # full path of the owning process exe
    visible: bool = True       # IsWindowVisible at enumeration time


@dataclass
class FocusResult:
    """Outcome of a switch/focus request."""

    status: str          # "ok" | "not_running" | "focus_failed"
    title: str = ""
    hwnd: int = 0


# Top-level IME windows hosted inside app processes — never real
# application windows.
_IME_WINDOW_TITLES = {"msctfime ui", "default ime"}


# ----------------------------------------------------------------------
# Process monitor (Toolhelp32 snapshot)
# ----------------------------------------------------------------------
class ProcessMonitor:
    """Lists running processes via the Win32 Toolhelp snapshot."""

    def __init__(
        self,
        lister: Optional[Callable[[], list]] = None,
        pid_details: Optional[Callable[[int], tuple]] = None,
    ) -> None:
        self._lister = lister or self._snapshot_processes
        self._pid_details = pid_details or self._pid_details_win32
        self._details_cache: dict[int, tuple] = {}

    def list_processes(self) -> list[ProcInfo]:
        try:
            return [ProcInfo(pid=p, name=n) for p, n in self._lister()]
        except Exception:  # noqa: BLE001
            logger.exception("Process enumeration failed")
            return []

    def pid_details(self, pid: int) -> tuple:
        """Return ``(exe_path, package_family)`` for ``pid`` (cached)."""
        if not pid:
            return ("", "")
        if pid not in self._details_cache:
            try:
                self._details_cache[pid] = self._pid_details(pid)
            except Exception:  # noqa: BLE001
                self._details_cache[pid] = ("", "")
            # Bound the cache; pids are recycled by Windows.
            if len(self._details_cache) > 4096:
                self._details_cache.clear()
        return self._details_cache[pid]

    def package_family(self, pid: int) -> str:
        return self.pid_details(pid)[1]

    def find(self, query: str) -> list[ProcInfo]:
        """Find processes whose exe name/stem matches ``query``."""
        if not query:
            return []
        q = query.lower().replace(".exe", "").strip()
        return [
            p for p in self.list_processes()
            if p.stem == q or q in p.stem
        ]

    def is_running(self, app: AppInfo) -> bool:
        """True when a process OR a visible window belongs to ``app``.

        Identity-aware: packaged apps match on package family name and
        display-name process candidates, never on PFN fragments.
        """
        from src.controllers import app_identity

        ident = app_identity.identity_for_app(app)
        candidates = set(ident.process_names)
        for p in self.list_processes():
            if not p.stem:
                continue
            stems = {p.stem, p.stem.replace(".", "")}
            stems.update(s for s in p.stem.split(".") if s)
            if stems & candidates:
                return True
        if ident.package_family:
            for p in self.list_processes():
                if (
                    self.package_family(p.pid).lower()
                    == ident.package_family
                ):
                    return True
        return False

    @staticmethod
    def _snapshot_processes() -> list[tuple[int, str]]:
        TH32CS_SNAPPROCESS = 0x00000002

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == wintypes.HANDLE(-1).value:
            return []
        out = []
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                out.append((int(entry.th32ProcessID), entry.szExeFile))
                ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snap)
        return out

    @staticmethod
    def _pid_details_win32(pid: int) -> tuple:
        """Return ``(exe_path, package_family_name)`` for ``pid``.

        Uses ``QueryFullProcessImageNameW`` and
        ``GetPackageFamilyName``; both degrade to "" when unavailable
        (non-Windows, access denied, unpackaged process).
        """
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return ("", "")
        k32 = windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = k32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return ("", "")
        try:
            exe_path = ""
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if k32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            ):
                exe_path = buf.value
            pfn = ""
            pfn_buf = ctypes.create_unicode_buffer(256)
            pfn_len = wintypes.DWORD(256)
            # GetPackageFamilyName(hProcess, &len, buf) -> LONG
            if k32.GetPackageFamilyName(
                handle, ctypes.byref(pfn_len), pfn_buf
            ) == 0:
                pfn = pfn_buf.value
            return (exe_path, pfn)
        except Exception:  # noqa: BLE001
            return ("", "")
        finally:
            k32.CloseHandle(handle)


# ----------------------------------------------------------------------
# Win32 API wrapper (injectable)
# ----------------------------------------------------------------------
class _User32:
    """Thin ctypes wrapper over user32 window functions."""

    def __init__(self) -> None:
        self._u = ctypes.windll.user32
        self._u.EnumWindows.argtypes = [
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        ]
        self._u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._u.GetWindowTextW.argtypes = [
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int
        ]
        self._u.IsWindowVisible.argtypes = [wintypes.HWND]
        self._u.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
        ]
        self._u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self._u.SetForegroundWindow.argtypes = [wintypes.HWND]
        self._u.PostMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        self._u.GetForegroundWindow.restype = wintypes.HWND
        self._u.GetWindowLongPtrW.argtypes = [
            wintypes.HWND, ctypes.c_int
        ]
        self._u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        self._u.keybd_event.argtypes = [
            wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_void_p
        ]
        self._u.GetWindowRect.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.RECT)
        ]
        self._u.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.UINT,
        ]
        self._u.SetWindowPos.restype = wintypes.BOOL

        class WINDOWPLACEMENT(ctypes.Structure):
            _fields_ = [
                ("length", wintypes.UINT),
                ("flags", wintypes.UINT),
                ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT),
                ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT),
            ]

        self._placement_cls = WINDOWPLACEMENT
        self._u.GetWindowPlacement.argtypes = [
            wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)
        ]
        self._u.GetWindowPlacement.restype = wintypes.BOOL
        self._u.IsZoomed.argtypes = [wintypes.HWND]
        self._u.IsZoomed.restype = wintypes.BOOL
        self._u.IsIconic.argtypes = [wintypes.HWND]
        self._u.IsIconic.restype = wintypes.BOOL
        self._u.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._u.AttachThreadInput.argtypes = [
            wintypes.DWORD, wintypes.DWORD, wintypes.BOOL
        ]
        self._u.AttachThreadInput.restype = wintypes.BOOL
        self._k32 = ctypes.windll.kernel32
        self._k32.GetCurrentThreadId.restype = wintypes.DWORD
        self._k32.GetCurrentThreadId.argtypes = []

    def enum_windows(self, callback) -> None:
        self._u.EnumWindows(callback, 0)

    def window_text(self, hwnd: int) -> str:
        n = self._u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        self._u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def is_visible(self, hwnd: int) -> bool:
        return bool(self._u.IsWindowVisible(hwnd))

    def pid_of(self, hwnd: int) -> int:
        pid = wintypes.DWORD(0)
        self._u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def thread_id_of(self, hwnd: int) -> int:
        """Thread id that owns ``hwnd`` (return value of
        GetWindowThreadProcessId)."""
        pid = wintypes.DWORD(0)
        tid = self._u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(tid or 0)

    def current_thread_id(self) -> int:
        return int(self._k32.GetCurrentThreadId() or 0)

    def attach_thread_input(
        self, attach: int, target: int, attach_flag: bool
    ) -> bool:
        return bool(
            self._u.AttachThreadInput(attach, target, bool(attach_flag))
        )

    def tap_alt(self) -> None:
        """Tap ALT — grants the calling process foreground rights."""
        VK_MENU, KEYEVENTF_KEYUP = 0x12, 0x0002
        u = self._u
        u.keybd_event(VK_MENU, 0, 0, None)
        u.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, None)

    def show(self, hwnd: int, cmd: int) -> None:
        self._u.ShowWindow(hwnd, cmd)

    def set_foreground(self, hwnd: int) -> bool:
        return bool(self._u.SetForegroundWindow(hwnd))

    def post_close(self, hwnd: int) -> None:
        self._u.PostMessageW(hwnd, WM_CLOSE, 0, 0)

    def foreground_hwnd(self) -> int:
        return int(self._u.GetForegroundWindow() or 0)

    def ex_style(self, hwnd: int) -> int:
        return int(self._u.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) or 0)

    def window_rect(self, hwnd: int):
        """Return (left, top, right, bottom) screen coordinates."""
        rect = wintypes.RECT()
        if not self._u.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)

    def set_window_rect(self, hwnd: int, rect, activate: bool = False) -> bool:
        """Position/size ``hwnd`` via SetWindowPos (no z-order change)."""
        left, top, right, bottom = rect
        flags = SWP_NOZORDER | (0 if activate else SWP_NOACTIVATE)
        return bool(self._u.SetWindowPos(
            hwnd, 0, int(left), int(top),
            int(right - left), int(bottom - top), flags,
        ))

    def window_show_cmd(self, hwnd: int) -> int:
        """Return the WINDOWPLACEMENT show command (1/2/3)."""
        placement = self._placement_cls()
        placement.length = ctypes.sizeof(self._placement_cls)
        if not self._u.GetWindowPlacement(hwnd, ctypes.byref(placement)):
            return SW_SHOWNORMAL
        return int(placement.showCmd)

    def normal_rect(self, hwnd: int):
        """Return the restored (non-maximized) window rectangle."""
        placement = self._placement_cls()
        placement.length = ctypes.sizeof(self._placement_cls)
        if not self._u.GetWindowPlacement(hwnd, ctypes.byref(placement)):
            return None
        r = placement.rcNormalPosition
        return (r.left, r.top, r.right, r.bottom)

    def show_desktop(self) -> None:
        """Toggle 'show desktop' (Win+D)."""
        VK_LWIN, VK_D = 0x5B, 0x44
        KEYEVENTF_KEYUP = 0x0002
        u = self._u
        u.keybd_event(VK_LWIN, 0, 0, None)
        u.keybd_event(VK_D, 0, 0, None)
        u.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, None)
        u.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, None)


# ----------------------------------------------------------------------
# Window manager
# ----------------------------------------------------------------------
class WindowManager:
    """Enumerate and manipulate visible top-level windows."""

    def __init__(
        self,
        api: Optional[_User32] = None,
        monitor: Optional[ProcessMonitor] = None,
    ) -> None:
        self._api = api if api is not None else _User32()
        self._monitor = monitor or ProcessMonitor()

    # -- enumeration ------------------------------------------------------
    def list_windows(self, include_hidden: bool = False) -> list[WindowInfo]:
        """Return visible (or all, with ``include_hidden``), titled,
        non-tool top-level windows."""
        procs = self._monitor.list_processes()
        proc_names = {p.pid: p.name for p in procs}
        pid_details = getattr(self._monitor, "pid_details", None)
        details_cache: dict[int, tuple] = {}

        def _details(pid: int) -> tuple:
            if pid not in details_cache:
                if pid_details is not None:
                    try:
                        details_cache[pid] = pid_details(pid)
                    except Exception:  # noqa: BLE001
                        details_cache[pid] = ("", "")
                else:
                    details_cache[pid] = ("", "")
            return details_cache[pid]

        out: list[WindowInfo] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _lp):
            try:
                visible = bool(self._api.is_visible(hwnd))
                if not visible and not include_hidden:
                    return True
                if self._api.ex_style(hwnd) & WS_EX_TOOLWINDOW:
                    return True
                title = self._api.window_text(hwnd)
                if not title:
                    return True
                # IME candidate-UI surfaces are owned by app processes
                # but are never user-facing application windows.
                if title.strip().lower() in _IME_WINDOW_TITLES:
                    return True
                pid = self._api.pid_of(hwnd)
                exe_path, pfn = _details(pid)
                out.append(
                    WindowInfo(
                        hwnd=int(hwnd),
                        pid=pid,
                        title=title,
                        process=proc_names.get(pid, ""),
                        package_family=pfn,
                        exe_path=exe_path,
                        visible=visible,
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return True

        try:
            self._api.enum_windows(_cb)
        except Exception:  # noqa: BLE001
            logger.exception("Window enumeration failed")
        return out

    def windows_for(self, target: str) -> list[WindowInfo]:
        """Visible windows whose process name or title matches ``target``."""
        if not target:
            return []
        q = target.lower().replace(".exe", "").strip()
        return [
            w for w in self.list_windows()
            if q in w.process.lower()
            or q in w.process.lower().rsplit(".", 1)[0]
            or q in w.title.lower()
        ]

    def windows_for_app(
        self, app: AppInfo, include_hidden: bool = False
    ) -> list[WindowInfo]:
        """Top-level windows belonging to ``app``.

        Identity-aware matching (process name, package family, exe
        path, framework-host title fallback) — works for packaged apps
        whose windows are hosted by ApplicationFrameHost or owned by
        a differently-named packaged process (e.g. WhatsApp.Root.exe).
        """
        from src.controllers import app_identity

        return app_identity.windows_for_app(
            self.list_windows(include_hidden=include_hidden), app
        )

    def _windows_for_target(self, target) -> list[WindowInfo]:
        """``windows_for`` for strings, ``windows_for_app`` for AppInfo."""
        if isinstance(target, AppInfo):
            return self.windows_for_app(target)
        return self.windows_for(str(target or ""))

    # -- operations --------------------------------------------------------
    def minimize(self, hwnd: int) -> bool:
        self._api.show(hwnd, SW_MINIMIZE)
        return True

    def maximize(self, hwnd: int) -> bool:
        self._api.show(hwnd, SW_MAXIMIZE)
        return True

    def restore(self, hwnd: int) -> bool:
        self._api.show(hwnd, SW_RESTORE)
        return True

    def focus(self, hwnd: int) -> bool:
        """Restore and foreground ``hwnd``; True only if verified."""
        self._api.show(hwnd, SW_RESTORE)
        if self._set_foreground(hwnd):
            return True
        # Workaround 1: an ALT tap grants this process foreground
        # rights under the Windows foreground lock.
        try:
            tap = getattr(self._api, "tap_alt", None)
            if tap is not None:
                tap()
                if self._set_foreground(hwnd):
                    return True
        except Exception:  # noqa: BLE001
            logger.debug("ALT-tap foreground workaround failed",
                         exc_info=True)
        # Workaround 2: attach our input thread to the foreground
        # thread so SetForegroundWindow is permitted, then retry.
        try:
            fg = self._api.foreground_hwnd()
            cur = getattr(self._api, "current_thread_id", lambda: 0)()
            tgt = getattr(self._api, "thread_id_of", lambda h: 0)(fg)
            attach = getattr(self._api, "attach_thread_input", None)
            if attach is not None and cur and tgt:
                attach(cur, tgt, True)
                try:
                    self._set_foreground(hwnd)
                finally:
                    attach(cur, tgt, False)
        except Exception:  # noqa: BLE001
            logger.debug("AttachThreadInput fallback failed", exc_info=True)
        # Final honest verification.
        fg_fn = getattr(self._api, "foreground_hwnd", None)
        return bool(fg_fn and fg_fn() == hwnd)

    def _set_foreground(self, hwnd: int) -> bool:
        """SetForegroundWindow plus a bounded verification poll."""
        try:
            if not self._api.set_foreground(hwnd):
                return False
        except Exception:  # noqa: BLE001
            return False
        fg_fn = getattr(self._api, "foreground_hwnd", None)
        if fg_fn is None:
            return True  # cannot verify; trust the API result
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            if fg_fn() == hwnd:
                return True
            time.sleep(0.02)
        return False

    def focus_target(self, target) -> FocusResult:
        """Focus the topmost visible window for ``target``.

        ``target`` may be an :class:`AppInfo` (identity-aware) or a
        plain string matched against process/title.

        When no *visible* window matches an AppInfo target, hidden
        top-level windows (e.g. apps minimized to tray) are restored
        and retried before reporting ``not_running``.
        """
        wins = self._windows_for_target(target)
        if wins and isinstance(target, AppInfo) and len(wins) > 1:
            # Prefer the window whose title evidences the app — tray
            # surfaces like "H.NotifyIcon_..." share the process but
            # are not the application's window.
            wins = sorted(
                wins, key=lambda w: -self._title_score(w, target)
            )
        if not wins and isinstance(target, AppInfo):
            # Packaged apps own several hidden utility windows (IME,
            # notify icons); try each until one becomes visible.
            # Title-evidenced candidates (the real window) go first so
            # a "H.NotifyIcon_..." surface is never restored instead.
            cands = self.windows_for_app(target, include_hidden=True)[:8]
            cands = sorted(
                cands,
                key=lambda w: -self._title_score(w, target),
            )
            for w in cands:
                try:
                    self._api.show(w.hwnd, SW_RESTORE)
                    time.sleep(0.05)
                    if self._api.is_visible(w.hwnd):
                        wins = [w]
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not wins:
            return FocusResult("not_running")
        hwnd = wins[0].hwnd
        if self.focus(hwnd):
            return FocusResult("ok", title=wins[0].title, hwnd=hwnd)
        return FocusResult("focus_failed", title=wins[0].title, hwnd=hwnd)

    @staticmethod
    def _title_score(window, app: AppInfo) -> int:
        """Shared display-name/title token count (0 = no evidence)."""
        try:
            from src.controllers.application_discovery import _normalize
            name_tokens = set(_normalize(app.name).split())
            title_tokens = set(
                _normalize(getattr(window, "title", "") or "").split()
            )
            return len(name_tokens & title_tokens)
        except Exception:  # noqa: BLE001
            return 0

    def focus_app(self, target: str) -> tuple[bool, str]:
        """Focus the first visible window matching ``target``."""
        res = self.focus_target(target)
        if res.status == "ok":
            return True, res.title
        return False, res.status

    def windows_for_pid(self, pid: int) -> list[WindowInfo]:
        """Visible top-level windows owned by ``pid``."""
        if not pid:
            return []
        return [w for w in self.list_windows() if w.pid == pid]

    # -- geometry ---------------------------------------------------------
    def window_rect(self, hwnd: int):
        """(left, top, right, bottom) or None."""
        fn = getattr(self._api, "window_rect", None)
        return fn(hwnd) if fn else None

    def is_minimized(self, hwnd: int) -> bool:
        cmd = getattr(self._api, "window_show_cmd", lambda h: SW_SHOWNORMAL)(hwnd)
        return cmd == SW_SHOWMINIMIZED

    def is_maximized(self, hwnd: int) -> bool:
        cmd = getattr(self._api, "window_show_cmd", lambda h: SW_SHOWNORMAL)(hwnd)
        return cmd == SW_SHOWMAXIMIZED

    def is_zoomed(self, hwnd: int) -> bool:
        """Authoritative ``IsZoomed`` check (maximized state)."""
        fn = getattr(self._api, "is_zoomed", None)
        if fn is not None:
            return bool(fn(hwnd))
        return self.is_maximized(hwnd)

    def is_iconic(self, hwnd: int) -> bool:
        """Authoritative ``IsIconic`` check (minimized state)."""
        fn = getattr(self._api, "is_iconic", None)
        if fn is not None:
            return bool(fn(hwnd))
        return self.is_minimized(hwnd)

    def normal_rect(self, hwnd: int):
        fn = getattr(self._api, "normal_rect", None)
        return fn(hwnd) if fn else None

    def set_rect(self, hwnd: int, rect, activate: bool = False) -> bool:
        fn = getattr(self._api, "set_window_rect", None)
        return bool(fn(hwnd, rect, activate)) if fn else False

    @staticmethod
    def fit_rect_in_work(
        width: int, height: int, work, margin: int = 0
    ) -> tuple:
        """Return a rect of ``width``x``height`` centered in ``work``.

        ``work`` is a (left, top, right, bottom) tuple or any object
        with left/top/right/bottom attributes. The result is
        clamped inside the work area so the window never lands under
        the taskbar; oversized windows are shrunk to fit.
        """
        if hasattr(work, "left"):
            wl, wt, wr, wb = work.left, work.top, work.right, work.bottom
        else:
            wl, wt, wr, wb = work
        w = max(1, min(int(width), wr - wl - 2 * margin))
        h = max(1, min(int(height), wb - wt - 2 * margin))
        left = wl + max(0, ((wr - wl) - w) // 2)
        top = wt + max(0, ((wb - wt) - h) // 2)
        return (left, top, left + w, top + h)

    def move_to_work_area(
        self, hwnd: int, work, margin: int = 0
    ) -> tuple:
        """Move ``hwnd`` inside work area ``work``, preserving size.

        Handles minimized/maximized windows: restores first, re-applies
        maximized afterwards. Returns (ok, maximized).
        """
        api = self._api
        maximized = self.is_maximized(hwnd)
        minimized = self.is_minimized(hwnd)
        hidden = False
        is_visible = getattr(api, "is_visible", None)
        if is_visible is not None:
            try:
                hidden = not is_visible(hwnd)
            except Exception:  # noqa: BLE001
                hidden = False
        if minimized or maximized or hidden:
            api.show(hwnd, SW_RESTORE)
            # Give the window manager a beat to apply the restore.
            import time as _time
            _time.sleep(0.05)
        rect = self.window_rect(hwnd)
        if rect is None:
            # Fall back to the normal (restored) rect.
            rect = self.normal_rect(hwnd)
        if rect is None:
            return False, maximized
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        target = self.fit_rect_in_work(width, height, work, margin)
        ok = self.set_rect(hwnd, target)
        if ok and maximized:
            # Let the move settle before re-maximizing — otherwise
            # Windows can maximize onto the monitor the window is
            # leaving, undoing the move.
            import time as _time
            _time.sleep(0.2)
            api.show(hwnd, SW_MAXIMIZE)
        return ok, maximized

    def foreground_window(self):
        """WindowInfo for the foreground window, or None."""
        hwnd = self._api.foreground_hwnd()
        if not hwnd:
            return None
        for w in self.list_windows():
            if w.hwnd == hwnd:
                return w
        pid = self._api.pid_of(hwnd)
        return WindowInfo(hwnd=hwnd, pid=pid, title="", process="")

    def show_desktop(self) -> None:
        self._api.show_desktop()

    def close_windows(self, target) -> int:
        """Post WM_CLOSE to every visible window matching ``target``.

        ``target`` may be an :class:`AppInfo` (identity-aware) or a
        plain string. Returns the number of windows signalled. Only
        top-level windows belonging to the target are touched.
        """
        count = 0
        for w in self._windows_for_target(target):
            try:
                self._api.post_close(w.hwnd)
                count += 1
            except Exception:  # noqa: BLE001
                logger.debug("WM_CLOSE failed for hwnd %s", w.hwnd)
        return count

    def close_window(self, hwnd: int) -> bool:
        """Post WM_CLOSE to a single verified top-level ``hwnd``.

        Callers must confirm the hwnd still belongs to a live window;
        this method only performs the close request.
        """
        try:
            self._api.post_close(int(hwnd))
            return True
        except Exception:  # noqa: BLE001
            logger.debug("WM_CLOSE failed for hwnd %s", hwnd)
            return False


# ----------------------------------------------------------------------
# Workstation lock
# ----------------------------------------------------------------------
class WorkstationLocker:
    """Lock the Windows workstation via LockWorkStation (user32).

    ``lock()`` returns True only when the API reports success — callers
    must never claim the workstation was locked otherwise.
    """

    def __init__(self, lock_fn: Optional[Callable[[], object]] = None) -> None:
        self._lock = lock_fn or self._lock_workstation

    def lock(self) -> bool:
        try:
            return bool(self._lock())
        except Exception:  # noqa: BLE001
            logger.exception("LockWorkStation failed")
            return False

    @staticmethod
    def _lock_workstation():
        import ctypes

        if not hasattr(ctypes, "windll"):
            return False
        return bool(ctypes.windll.user32.LockWorkStation())


# ----------------------------------------------------------------------
# Application launcher
# ----------------------------------------------------------------------
class ApplicationLauncher:
    """Launch validated executables — never through a shell."""

    def __init__(
        self,
        discovery,
        popen: Optional[Callable] = None,
        monitor: Optional[ProcessMonitor] = None,
    ) -> None:
        self._discovery = discovery
        self._popen = popen or subprocess.Popen
        self._monitor = monitor or ProcessMonitor()

    def resolve(self, query: str):
        """Resolve ``query`` to a validated AppInfo or None."""
        app = self._discovery.find(query)
        if app is None:
            return None
        if not self._discovery.is_app_path_allowed(app.exe_path):
            logger.warning("Rejected invalid exe path: %s", app.exe_path)
            return None
        return app

    def launch(self, app: AppInfo):
        """Launch ``app``. Returns (ok, detail)."""
        ok, _proc, detail = self.launch_process(app)
        return ok, detail

    def launch_process(self, app: AppInfo):
        """Launch ``app`` and return (ok, process, detail).

        ``process`` is the object returned by ``popen`` (a
        ``subprocess.Popen`` in production, a fake in tests); it may be
        None when the popen callable does not return one. The pid is
        read via ``getattr(proc, "pid", None)``.
        """
        launch_id = app.exe_path or getattr(app, "aumid", "")
        if not self._discovery.is_app_path_allowed(launch_id):
            return False, None, "invalid_path"
        try:
            if getattr(app, "launch_kind", "exe") == "aumid":
                # Packaged app: launch via the AppsFolder identifier,
                # never a shell command.
                self._open_aumid(app)
                logger.info("Launched packaged app %s", app.name)
                return True, None, "ok"
            proc = self._popen([app.exe_path])  # list form: no shell
            logger.info("Launched %s (%s)", app.name, app.exe_path)
            return True, proc, "ok"
        except FileNotFoundError:
            return False, None, "not_found"
        except PermissionError:
            return False, None, "permission_denied"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Launch failed for %s", app.exe_path)
            return False, None, str(exc)

    def _open_aumid(self, app: AppInfo) -> None:
        """Open a packaged app via shell:AppsFolder (no shell exec)."""
        import os

        target = app.launch_target
        opener = getattr(self, "_aumid_opener", None) or os.startfile
        opener(target)

    def launch_pid(self, app: AppInfo):
        """Launch ``app`` and return its pid (or None)."""
        ok, proc, _detail = self.launch_process(app)
        if not ok or proc is None:
            return None
        return getattr(proc, "pid", None)

    def running(self, app: AppInfo) -> bool:
        return self._monitor.is_running(app)

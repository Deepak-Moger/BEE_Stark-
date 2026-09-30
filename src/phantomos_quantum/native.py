"""Native input dispatch with explicit dry-run and platform backends.

No PyAutoGUI/pynput is used. The privileged path is opt-in and isolated behind the
`NativeBackend` protocol, which makes the gesture engine testable without OS access.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import platform
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .display import MultiMonitorLayout
from .errors import DependencyUnavailable, PermissionDenied
from .models import DispatchResult, Intent, IntentKind


class NativeBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def button(self, button: str, pressed: bool) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def key(self, key: str, pressed: bool) -> None: ...
    def text(self, text: str) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class AuditEntry:
    action: str
    payload: dict[str, Any]
    timestamp: float


class DryRunBackend:
    """Records exactly what would have been sent to the OS."""

    def __init__(self, max_entries: int = 1000) -> None:
        self.events: deque[AuditEntry] = deque(maxlen=max_entries)

    def _record(self, action: str, **payload: Any) -> None:
        self.events.append(AuditEntry(action, payload, time.monotonic()))

    def move(self, x: int, y: int) -> None:
        self._record("move", x=x, y=y)

    def button(self, button: str, pressed: bool) -> None:
        self._record("button", button=button, pressed=pressed)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx=dx, dy=dy)

    def key(self, key: str, pressed: bool) -> None:
        self._record("key", key=key, pressed=pressed)

    def text(self, text: str) -> None:
        self._record("text", text=text)

    def close(self) -> None:
        pass


# Windows SendInput ---------------------------------------------------------
class _WinMouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _WinKeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _WinInputUnion(ctypes.Union):
    _fields_ = [("mi", _WinMouseInput), ("ki", _WinKeyboardInput)]


class _WinInput(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", _WinInputUnion)]


_WIN_KEYS = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "escape": 0x1B, "space": 0x20,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "delete": 0x2E,
}
_WIN_KEYS.update({chr(ord("a") + i): 0x41 + i for i in range(26)})
_WIN_KEYS.update({str(i): 0x30 + i for i in range(10)})


class WindowsSendInputBackend:
    """Direct user32.SendInput backend; instantiated only on Windows."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise DependencyUnavailable("Windows SendInput is only available on Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = (wintypes.UINT, ctypes.POINTER(_WinInput), ctypes.c_int)
        self._send_input.restype = wintypes.UINT

    def _send(self, item: _WinInput) -> None:
        if self._send_input(1, ctypes.byref(item), ctypes.sizeof(_WinInput)) != 1:
            raise PermissionDenied(f"SendInput failed: {ctypes.get_last_error()}")

    def move(self, x: int, y: int) -> None:
        # The dispatcher passes physical pixels; SendInput expects normalized coordinates.
        width = max(1, int(self._user32.GetSystemMetrics(0)) - 1)
        height = max(1, int(self._user32.GetSystemMetrics(1)) - 1)
        nx = max(0, min(65535, round(x * 65535 / width)))
        ny = max(0, min(65535, round(y * 65535 / height)))
        item = _WinInput(type=0, mi=_WinMouseInput(nx, ny, 0, 0x8000 | 0x0001, 0, None))
        self._send(item)

    def button(self, button: str, pressed: bool) -> None:
        flags = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
        down, up = flags.get(button, flags["left"])
        item = _WinInput(type=0, mi=_WinMouseInput(0, 0, 0, down if pressed else up, 0, None))
        self._send(item)

    def scroll(self, dx: int, dy: int) -> None:
        # WHEEL_DELTA is 120. Horizontal scroll uses MOUSEEVENTF_HWHEEL.
        if dy:
            self._send(_WinInput(type=0, mi=_WinMouseInput(0, 0, int(dy * 120), 0x0800, 0, None)))
        if dx:
            self._send(_WinInput(type=0, mi=_WinMouseInput(0, 0, int(dx * 120), 0x01000, 0, None)))

    def key(self, key: str, pressed: bool) -> None:
        code = _WIN_KEYS.get(key.lower())
        if code is None and len(key) == 1:
            code = ord(key.upper())
        if code is None:
            raise ValueError(f"unsupported Windows key: {key}")
        flags = 0 if pressed else 0x0002
        self._send(_WinInput(type=1, ki=_WinKeyboardInput(code, 0, flags, 0, None)))

    def text(self, text: str) -> None:
        for char in text:
            self.key(char, True)
            self.key(char, False)

    def close(self) -> None:
        pass


# macOS CoreGraphics -------------------------------------------------------
_MAC_KEYS = {"backspace": 51, "enter": 36, "escape": 53, "space": 49, "tab": 48}
_MAC_KEYS.update({chr(ord("a") + i): code for i, code in enumerate(
    (0, 11, 8, 2, 14, 3, 5, 4, 34, 38, 40, 37, 46, 45, 31, 35, 12, 15, 1, 17, 32, 9, 13, 7, 16, 6)
)})


class MacCoreGraphicsBackend:
    """Direct CoreGraphics CGEvent backend; instantiated only on macOS."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise DependencyUnavailable("CoreGraphics is only available on macOS")
        self._cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        self._cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
        self._cg.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
        self._cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self._cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        self._cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
        self._cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
        self._cg.CGEventCreateScrollWheelEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
        self._cg.CFRelease.argtypes = [ctypes.c_void_p]
        self._location = (0.0, 0.0)

    def _post_mouse(self, event_type: int, button: int = 0) -> None:
        event = self._cg.CGEventCreateMouseEvent(None, event_type, *self._location, button)
        if not event:
            raise PermissionDenied("CGEventCreateMouseEvent failed; check Accessibility permission")
        self._cg.CGEventPost(0, event)
        self._cg.CFRelease(event)

    def move(self, x: int, y: int) -> None:
        self._location = (float(x), float(y))
        self._post_mouse(5, 0)  # kCGEventMouseMoved

    def button(self, button: str, pressed: bool) -> None:
        button_id = {"left": 0, "right": 1, "middle": 2}.get(button, 0)
        event_type = {("left", True): 1, ("left", False): 2, ("right", True): 3, ("right", False): 4,
                      ("middle", True): 25, ("middle", False): 26}.get((button, pressed), 1)
        self._post_mouse(event_type, button_id)

    def scroll(self, dx: int, dy: int) -> None:
        event = self._cg.CGEventCreateScrollWheelEvent(None, 0, 2, int(dy), int(dx))
        self._cg.CGEventPost(0, event)
        self._cg.CFRelease(event)

    def key(self, key: str, pressed: bool) -> None:
        code = _MAC_KEYS.get(key.lower())
        if code is None:
            raise ValueError(f"unsupported macOS key: {key}")
        event = self._cg.CGEventCreateKeyboardEvent(None, code, pressed)
        self._cg.CGEventPost(0, event)
        self._cg.CFRelease(event)

    def text(self, text: str) -> None:
        # Keycode synthesis is intentionally limited; callers should use voice/text
        # integration for Unicode, which can use the platform pasteboard safely.
        for char in text:
            self.key(char, True)
            self.key(char, False)

    def close(self) -> None:
        pass


# Linux uinput --------------------------------------------------------------
_EV_SYN, _EV_KEY, _EV_REL = 0, 1, 2
_REL_X, _REL_Y, _REL_WHEEL, _REL_HWHEEL = 0, 1, 8, 6
_BTN_LEFT, _BTN_RIGHT, _BTN_MIDDLE = 272, 273, 274


def _ioc(direction: int, type_: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (type_ << 8) | number


_IOC_WRITE, _IOC_READ = 1, 2
_UI_SET_EVBIT = _ioc(_IOC_WRITE, ord("U"), 100, struct.calcsize("i"))
_UI_SET_KEYBIT = _ioc(_IOC_WRITE, ord("U"), 101, struct.calcsize("i"))
_UI_SET_RELBIT = _ioc(_IOC_WRITE, ord("U"), 102, struct.calcsize("i"))
_UI_DEV_CREATE = _ioc(0, ord("U"), 1, 0)
_UI_DEV_DESTROY = _ioc(0, ord("U"), 2, 0)


class LinuxUInputBackend:
    """Direct `/dev/uinput` backend. It never falls back to desktop automation."""

    def __init__(self, device_path: str = "/dev/uinput") -> None:
        if not sys.platform.startswith("linux"):
            raise DependencyUnavailable("uinput is only available on Linux")
        try:
            import fcntl
            self._fcntl = fcntl
            self._fd = os.open(device_path, os.O_WRONLY | os.O_NONBLOCK)
            for ioctl, value in ((_UI_SET_EVBIT, _EV_KEY), (_UI_SET_EVBIT, _EV_REL),
                                 (_UI_SET_RELBIT, _REL_X), (_UI_SET_RELBIT, _REL_Y),
                                 (_UI_SET_RELBIT, _REL_WHEEL), (_UI_SET_RELBIT, _REL_HWHEEL),
                                 (_UI_SET_KEYBIT, _BTN_LEFT), (_UI_SET_KEYBIT, _BTN_RIGHT),
                                 (_UI_SET_KEYBIT, _BTN_MIDDLE)):
                fcntl.ioctl(self._fd, ioctl, value)
            # uinput_user_dev: name[80], id[4xH], ff_effects_max, abs arrays.
            header = struct.pack("80sHHHHI", b"PhantomOS Quantum", 0x03, 0x1234, 1, 1, 0)
            self._fd_write(header + bytes(64 * 4 * 4))
            fcntl.ioctl(self._fd, _UI_DEV_CREATE)
        except (OSError, ImportError) as exc:
            try:
                os.close(self._fd)
            except Exception:
                pass
            raise PermissionDenied("cannot create /dev/uinput device; check permissions") from exc

    def _fd_write(self, data: bytes) -> None:
        os.write(self._fd, data)

    def _event(self, event_type: int, code: int, value: int) -> None:
        now = time.time()
        sec, usec = int(now), int((now - int(now)) * 1_000_000)
        self._fd_write(struct.pack("llHHi", sec, usec, event_type, code, value))

    def _sync(self) -> None:
        self._event(_EV_SYN, 0, 0)

    def move(self, x: int, y: int) -> None:
        self._event(_EV_REL, _REL_X, int(x))
        self._event(_EV_REL, _REL_Y, int(y))
        self._sync()

    def button(self, button: str, pressed: bool) -> None:
        code = {"left": _BTN_LEFT, "right": _BTN_RIGHT, "middle": _BTN_MIDDLE}.get(button, _BTN_LEFT)
        self._event(_EV_KEY, code, int(pressed))
        self._sync()

    def scroll(self, dx: int, dy: int) -> None:
        if dx:
            self._event(_EV_REL, _REL_HWHEEL, int(dx))
        if dy:
            self._event(_EV_REL, _REL_WHEEL, int(dy))
        self._sync()

    def key(self, key: str, pressed: bool) -> None:
        # Linux key codes for common controls; printable text is intentionally small.
        keycodes = {"backspace": 14, "enter": 28, "escape": 1, "space": 57, "tab": 15,
                    "ctrl": 29, "shift": 42, "alt": 56}
        keycodes.update(dict(zip("abcdefghijklmnopqrstuvwxyz",
                                (30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50,
                                 49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44))))
        code = keycodes.get(key.lower())
        if code is None:
            raise ValueError(f"unsupported Linux key: {key}")
        self._event(_EV_KEY, code, int(pressed))
        self._sync()

    def text(self, text: str) -> None:
        for char in text:
            self.key(char, True)
            self.key(char, False)

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            try:
                self._fcntl.ioctl(self._fd, _UI_DEV_DESTROY)
            finally:
                os.close(self._fd)
                self._fd = None


def create_native_backend(name: str = "auto", dry_run: bool = True) -> NativeBackend:
    """Create a native backend, keeping dry-run the safe and deterministic default."""
    if dry_run or name == "dry-run":
        return DryRunBackend()
    selected = name.lower()
    if selected == "auto":
        selected = "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux"
    if selected in {"windows", "win32"}:
        return WindowsSendInputBackend()
    if selected in {"mac", "macos", "darwin"}:
        return MacCoreGraphicsBackend()
    if selected in {"linux", "uinput"}:
        return LinuxUInputBackend()
    if selected in {"compiled", "ctypes"}:
        return CompiledCtypesBackend()
    raise ValueError(f"unknown native backend: {name}")


class CompiledCtypesBackend:
    """Load the optional C++17 C ABI built from `native/phantom_input.cpp`."""

    def __init__(self, library_path: str | None = None) -> None:
        path = library_path or os.getenv("PHANTOMOS_NATIVE_LIB")
        if not path:
            raise DependencyUnavailable("set PHANTOMOS_NATIVE_LIB to a compiled dispatcher library")
        self._lib = ctypes.CDLL(path)
        for name, args in {
            "phantom_move": [ctypes.c_int, ctypes.c_int],
            "phantom_button": [ctypes.c_int, ctypes.c_int],
            "phantom_scroll": [ctypes.c_int, ctypes.c_int],
            "phantom_key": [ctypes.c_int, ctypes.c_int],
        }.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, ctypes.c_int
        self._keycodes = {"backspace": 8, "enter": 13, "escape": 27, "space": 32, "tab": 9,
                          "ctrl": 17, "shift": 16, "alt": 18}

    def _call(self, name: str, *args: int) -> None:
        if getattr(self._lib, name)(*args) != 0:
            raise PermissionDenied(f"compiled native call {name} failed")

    def move(self, x: int, y: int) -> None:
        self._call("phantom_move", x, y)

    def button(self, button: str, pressed: bool) -> None:
        self._call("phantom_button", {"left": 0, "right": 1, "middle": 2}.get(button, 0), int(pressed))

    def scroll(self, dx: int, dy: int) -> None:
        self._call("phantom_scroll", dx, dy)

    def key(self, key: str, pressed: bool) -> None:
        code = self._keycodes.get(key.lower(), ord(key.upper()) if len(key) == 1 else None)
        if code is None:
            raise ValueError(f"unsupported compiled key: {key}")
        self._call("phantom_key", code, int(pressed))

    def text(self, text: str) -> None:
        for char in text:
            self.key(char, True)
            self.key(char, False)

    def close(self) -> None:
        pass


class NativeDispatcher:
    """Policy-aware conversion from normalized intents to OS events."""

    def __init__(self, backend: NativeBackend, width: int, height: int,
                 allowed_actions: frozenset[str], min_confidence: float = 0.72,
                 require_unlock: bool = False, layout: MultiMonitorLayout | None = None,
                 action_handlers: Mapping[str, Callable[[Intent], None]] | None = None) -> None:
        self.backend = backend
        self.width, self.height = width, height
        self.allowed_actions = allowed_actions
        self.min_confidence = min_confidence
        self.require_unlock = require_unlock
        self.layout = layout
        self.action_handlers = dict(action_handlers or {})
        self.enabled = True
        self.unlocked = not require_unlock

    def dispatch(self, intent: Intent) -> DispatchResult:
        if not self.enabled:
            return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, "kill switch")
        if intent.confidence < self.min_confidence:
            return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, "low confidence")
        action_allowed = intent.action in self.allowed_actions
        if intent.kind is IntentKind.KEY:
            action_allowed = action_allowed or "key" in self.allowed_actions
        if intent.kind is IntentKind.BUTTON:
            action_allowed = action_allowed or "button" in self.allowed_actions
        if not action_allowed:
            return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, "action not allowed")
        if (self.require_unlock or intent.kind in {IntentKind.SYSTEM, IntentKind.WINDOW}) and not self.unlocked:
            return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, "workspace locked")
        try:
            if intent.kind is IntentKind.POINTER_MOVE and intent.point:
                if self.layout is not None:
                    x, y = self.layout.map(intent.point)
                else:
                    x, y = round(intent.point.x * (self.width - 1)), round(intent.point.y * (self.height - 1))
                self.backend.move(x, y)
            elif intent.kind is IntentKind.BUTTON:
                self.backend.button(str(intent.metadata.get("button", "left")), bool(intent.metadata.get("pressed", True)))
            elif intent.kind is IntentKind.SCROLL:
                self.backend.scroll(round(intent.metadata.get("dx", 0)), round(intent.value))
            elif intent.kind is IntentKind.KEY:
                key = intent.action if intent.action != "key" else intent.text
                modifiers = tuple(str(item) for item in intent.metadata.get("shortcut", ()))
                for modifier in modifiers:
                    self.backend.key(modifier, True)
                self.backend.key(key, bool(intent.metadata.get("pressed", True)))
                if modifiers:
                    self.backend.key(key, False)
                    for modifier in reversed(modifiers):
                        self.backend.key(modifier, False)
            elif intent.kind is IntentKind.TEXT:
                self.backend.text(intent.text)
            elif intent.action in self.action_handlers:
                self.action_handlers[intent.action](intent)
            else:
                return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, "unsupported intent")
        except (OSError, PermissionDenied, ValueError) as exc:
            return DispatchResult(False, isinstance(self.backend, DryRunBackend), intent.action, str(exc))
        return DispatchResult(True, isinstance(self.backend, DryRunBackend), intent.action)

    def close(self) -> None:
        self.backend.close()

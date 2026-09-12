"""Recognize the native Smart Downloads window, invisible to Android captures.

Only the unique local HD-Player process from the selected ADB installation is
eligible. The popup's own unnamed, top-right button must support Invoke; neither
Enable nor the player's window-close control is ever used as a fallback.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

TITLE = 'BlueStacks Smart Downloads'


@dataclass(frozen=True)
class PopupWindow:
    handle: int
    process: int


@dataclass(frozen=True)
class PopupControl:
    window: PopupWindow
    popup_id: tuple
    button_id: tuple
    bounds: tuple[int, int, int, int]


def close_button_layout(popup, button):
    """All bounds are UI Automation screen coordinates, independent of DPI."""
    left, top, right, bottom = popup
    x, y, x2, y2 = button
    width, height = right-left, bottom-top
    return (width >= 300 and height >= 200 and left <= x < x2 <= right
            and top <= y < y2 <= bottom and x > left+width*.90
            and y2 < top+height*.16 and 12 <= x2-x <= width*.09
            and 12 <= y2-y <= height*.14 and .65 <= (x2-x)/(y2-y) <= 1.5)


class SmartDownloadsDesktop:
    def __init__(self, adb_executable):
        from ctypes import wintypes as wt

        self.player = Path(adb_executable).resolve().with_name('HD-Player.exe')
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.callback_type = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        signatures = {
            'EnumWindows': ([self.callback_type, wt.LPARAM], wt.BOOL),
            'EnumChildWindows': ([wt.HWND, self.callback_type, wt.LPARAM], wt.BOOL),
            'IsWindowVisible': ([wt.HWND], wt.BOOL),
            'GetWindowTextW': ([wt.HWND, wt.LPWSTR, ctypes.c_int], ctypes.c_int),
            'GetWindowThreadProcessId': ([wt.HWND, ctypes.POINTER(wt.DWORD)], wt.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.user, name)
            function.argtypes, function.restype = args, result
        self.kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.kernel.OpenProcess.restype = wt.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [
            wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = wt.BOOL
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.kernel.CloseHandle.restype = wt.BOOL
        self.uia = self.types = self.com = None

    def _windows(self, parent=None):
        result = []

        @self.callback_type
        def visit(handle, _):
            result.append(handle)
            return True

        if parent is None:
            if not self.user.EnumWindows(visit, 0):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            self.user.EnumChildWindows(parent, visit, 0)
        return result

    def _process(self, handle):
        from ctypes import wintypes as wt

        pid = wt.DWORD()
        self.user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return pid.value

    def _is_player(self, pid):
        from ctypes import wintypes as wt

        process = self.kernel.OpenProcess(0x1000, False, pid)  # query limited information
        if not process:
            return False
        try:
            name, size = ctypes.create_unicode_buffer(32768), wt.DWORD(32768)
            return bool(self.kernel.QueryFullProcessImageNameW(
                process, 0, name, ctypes.byref(size))) and Path(name.value) == self.player
        finally:
            self.kernel.CloseHandle(process)

    def locate(self):
        roots, processes = [], {}
        # Include hidden/minimized players in the ambiguity check. Never select
        # another emulator just because it is the only visible one.
        for handle in self._windows():
            pid = self._process(handle)
            if pid not in processes:
                processes[pid] = self._is_player(pid)
            if processes[pid]:
                roots.append((handle, pid))
        if len({pid for _, pid in roots}) != 1:
            return None
        found = set()
        for root, pid in roots:
            for handle in (root, *self._windows(root)):
                if not self.user.IsWindowVisible(handle) or self._process(handle) != pid:
                    continue
                title = ctypes.create_unicode_buffer(256)
                self.user.GetWindowTextW(handle, title, len(title))
                if title.value == TITLE:
                    found.add(PopupWindow(handle, pid))
        if len(found) > 1:
            raise RuntimeError('More than one Smart Downloads popup is visible.')
        return next(iter(found), None)

    def visible(self, window):
        if (not self.user.IsWindowVisible(window.handle)
                or self._process(window.handle) != window.process):
            return False
        title = ctypes.create_unicode_buffer(256)
        self.user.GetWindowTextW(window.handle, title, len(title))
        return title.value == TITLE

    def _automation(self):
        if self.uia is None:
            import comtypes
            from comtypes.client import CreateObject, GetModule

            comtypes.CoInitialize()
            self.com = comtypes
            self.types = GetModule('UIAutomationCore.dll')
            self.uia = CreateObject(self.types.CUIAutomation8, interface=self.types.IUIAutomation2)
            self.uia.ConnectionTimeout = 1000
            self.uia.TransactionTimeout = 1000
        return self.uia

    @staticmethod
    def _rect(element):
        rect = element.CurrentBoundingRectangle
        return rect.left, rect.top, rect.right, rect.bottom

    def _inspect(self, window):
        uia = self._automation()
        popup = uia.ElementFromHandle(window.handle)
        if (popup.CurrentName != TITLE or popup.CurrentProcessId != window.process
                or popup.CurrentIsOffscreen or not popup.CurrentIsEnabled):
            raise RuntimeError('The Smart Downloads window changed before inspection.')
        condition = uia.CreatePropertyCondition(self.types.UIA_ControlTypePropertyId,
                                                self.types.UIA_ButtonControlTypeId)
        buttons = popup.FindAll(self.types.TreeScope_Children, condition)
        controls = [buttons.GetElement(i) for i in range(buttons.Length)]
        # The observed dialog contains exactly Enable and its unnamed X.
        enable = [b for b in controls if b.CurrentName == 'Enable']
        close = [b for b in controls if not b.CurrentName and b.CurrentIsEnabled
                 and not b.CurrentIsOffscreen and b.CurrentProcessId == window.process
                 and close_button_layout(self._rect(popup), self._rect(b))]
        if len(controls) != 2 or len(enable) != 1 or len(close) != 1:
            raise RuntimeError('The Smart Downloads close button could not be identified.')
        button = close[0]
        observation = PopupControl(window, tuple(popup.GetRuntimeId()),
                                   tuple(button.GetRuntimeId()), self._rect(button))
        return observation, button

    def observe(self, window):
        return self._inspect(window)[0]

    def dismiss(self, expected):
        # Fresh identity AND geometry checks immediately before invoking the X.
        if self.locate() != expected.window:
            raise RuntimeError('The Smart Downloads window changed before closing.')
        current, button = self._inspect(expected.window)
        if current != expected:
            raise RuntimeError('The Smart Downloads close button moved before input.')
        button.GetCurrentPattern(self.types.UIA_InvokePatternId).QueryInterface(
            self.types.IUIAutomationInvokePattern).Invoke()

    def close(self):
        self.uia = self.types = None
        if self.com is not None:
            self.com.CoUninitialize()
            self.com = None

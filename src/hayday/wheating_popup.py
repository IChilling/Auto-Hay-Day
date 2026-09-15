"""Dismiss a confirmed host popup before obtaining the next game screenshot."""
from __future__ import annotations

import os
import re
import time

from hayday.adb import AdbClient


class WheatHostPopup:
    def __init__(self, run, desktop=None, clock=time.monotonic):
        self.run, self.desktop, self.clock = run, desktop, clock
        self.next_check = 0.
        self.uncertain = False
        if desktop is None and os.name == 'nt' and isinstance(run.client, AdbClient):
            local = re.fullmatch(r'emulator-\d+|(?:127\.0\.0\.1|localhost|\[::1\]):\d+', run.serial)
            executable = run.client.executable
            if (local and executable.name.lower() == 'hd-adb.exe'
                    and executable.with_name('HD-Player.exe').is_file() and run.client.is_bluestacks()):
                from hayday.bluestacks_popup import SmartDownloadsDesktop
                self.desktop = SmartDownloadsDesktop(executable)

    def process(self):
        self.run.check()
        if self.uncertain:
            self.run.block('The Smart Downloads close action is unconfirmed; no repeated input was sent.')
        if self.desktop is None or self.clock() < self.next_check:
            return
        self.next_check = self.clock()+1.
        try:
            window = self.desktop.locate()
            if window is None:
                return
            if not self.run._lock_file:
                self.run.block('Smart Downloads dismissal requires the active device session lock.')
            first = self.desktop.observe(window)
            self.run.publish('Wheating: closing the BlueStacks Smart Downloads popup.')
            self.run.wait(.10)
            current = self.desktop.locate()
            if current is None:
                return
            if current != window or self.desktop.observe(current) != first:
                self.run.block('The Smart Downloads popup changed before its close button could be confirmed.')
            self.run.check()
            self.uncertain = True
            self.desktop.dismiss(first)
            for _ in range(10):
                self.run.wait(.10)
                if not self.desktop.visible(window):
                    self.uncertain = False
                    self.run.publish('Wheating: Smart Downloads closed. Resuming the game.')
                    return
            self.run.block('Smart Downloads remained visible after closing; no repeated input was sent.')
        except Exception as exc:
            # Cancellation must retain its normal result, including if it
            # arrived while a native accessibility call was in progress.
            self.run.check()
            self.run.block(str(exc))

    def close(self):
        if self.desktop is not None:
            self.desktop.close()

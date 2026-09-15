"""Service integration kept separate from the order workflow."""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime

from hayday.adb import AdbClient, AdbError


class WheatingServices:
    def _init_wheating(self):
        self._wheating_runner = None
        self.last_wheating_run = None
        self.wheating_progress = {}
        self.wheating_preview_key = None

    def select_wheating_instances(self, keys):
        with self._lock:
            self._ensure_open()
            if self._wheating_runner:
                raise AdbError('Stop Wheating before changing its selected instances.')
            if not set(keys) <= {i.key for i in self.instances}:
                raise AdbError('An instance is no longer available. Discover again.')
            self.save_settings(replace(self.settings, wheating_instances=tuple(keys)))

    def select_wheating_preview(self, key):
        with self._lock:
            states = self.wheating_progress.get('instances', {})
            if key not in states:
                return
            self.wheating_preview_key = key
            state = states[key]
            if state.get('frame'):
                self.frame, self.frame_serial = state['frame'], state['serial']

    def _start_wheating_fleet(self, cancel_event, progress, max_seconds, reset_restart_cooldown):
        from hayday.wheating_fleet import WheatingFleet

        with self._lock:
            self._ensure_open()
            if self._wheating_runner or self._order_runner or self._board_runner or self._captures_active:
                raise AdbError('Stop the other workflow before starting Wheating.')
            choices = {i.key: i for i in self.instances}
            if not set(self.settings.wheating_instances) <= choices.keys():
                raise AdbError('A selected instance is unavailable. Discover and review the selection.')
            selected = [choices[key] for key in self.settings.wheating_instances]
            self.wheating_preview_key = selected[0].key
            self.wheating_progress = {}

            def publish(update):
                with self._lock:
                    if self._closed or update['sequence'] <= self.wheating_progress.get('sequence', 0):
                        return
                    current = update['instances'].get(self.wheating_preview_key, {})
                    if current.get('frame'):
                        self.frame, self.frame_serial = current['frame'], current['serial']
                    self.wheating_progress = dict(update)
                    output = {**update, 'frame': current.get('frame')}
                if progress:
                    progress(output)

            fleet = WheatingFleet(selected, self.data.root/'diagnostics'/'wheating',
                progress=publish, cancel_event=cancel_event, max_seconds=max_seconds,
                reset_restart_cooldown=reset_restart_cooldown)
            self._wheating_runner = fleet
        try:
            result = fleet.run()
            with self._lock:
                if not self._closed:
                    self.last_wheating_run = result
                    for key, outcome in result.results.items():
                        self.data.log('INFO' if outcome.success else 'WARNING',
                            f'{choices[key].display_name}: {outcome.message} Evidence: {outcome.diagnostics}', 'wheating')
            return result
        finally:
            with self._lock:
                if self._wheating_runner is fleet:
                    self._wheating_runner = None

    def start_wheating(self, cancel_event: threading.Event | None = None, progress=None, max_seconds=None,
                      *, reset_restart_cooldown=False):
        if self.settings.wheating_instances:
            return self._start_wheating_fleet(cancel_event, progress, max_seconds, reset_restart_cooldown)
        from hayday.wheating import WheatingRunner

        with self._lock:
            self._ensure_open()
            if self._wheating_runner or self._order_runner or self._board_runner or self._captures_active:
                raise AdbError('Another game workflow or capture is already running. Stop it first.')
            original_client, serial = self.client, self.connected_serial
            if not original_client or not serial:
                raise AdbError('Select an online emulator device first.')
            client = AdbClient(self.settings.adb_path, serial=serial, timeout_seconds=30)

            def publish(update):
                with self._lock:
                    if self._closed or self.client is not original_client or self.connected_serial != serial:
                        return
                    self.wheating_progress = dict(update)
                    if update.get('frame'):
                        self.frame, self.frame_serial = update['frame'], serial
                if progress:
                    progress(dict(update))

            try:
                runner = WheatingRunner(client, self.data.root/'diagnostics'/'wheating',
                    cancel_event=cancel_event, progress=publish, max_seconds=max_seconds,
                    reset_restart_cooldown=reset_restart_cooldown)
            except Exception:
                client.close()
                raise
            self._wheating_runner = runner
            self.wheating_progress = {'message': 'Starting Wheating…', 'listed': 0, 'planted': 0}
        try:
            result = runner.run()
            with self._lock:
                if self._closed:
                    return result
                self.last_wheating_run = result
                if self.client is original_client and self.connected_serial == serial:
                    if result.frame:
                        self.frame, self.frame_serial = result.frame, serial
                    self.wheating_progress = {'message': result.message, 'status': result.status,
                        'planted': result.planted, 'listed': result.listed,
                        'collected': result.collected, 'advertisements': result.advertisements}
                self.data.log('INFO' if result.success else 'WARNING', result.message, 'wheating')
                if result.diagnostics:
                    self.data.log('INFO', f'Wheating evidence: {result.diagnostics}', 'wheating')
            return result
        finally:
            client.close()
            with self._lock:
                if self._wheating_runner is runner:
                    self._wheating_runner = None

    def cancel_wheating(self, key=None):
        with self._lock:
            if self._wheating_runner:
                if key is None:
                    self._wheating_runner.cancel()
                elif hasattr(self._wheating_runner, 'instances'):
                    self._wheating_runner.cancel(key)

    def reset_wheating(self) -> str:
        """Reset the selected device's Wheating state and relaunch Hay Day.

        Persistent state is moved into a timestamped archive instead of being
        deleted.  This makes the button safe to use when a run is wedged while
        ensuring the next run starts with no stale crop, shop, or restart
        intent.  The Android package is restarted through ADB, so Windows focus
        and the user's other applications are untouched.
        """
        with self._lock:
            self._ensure_open()
            if self._order_runner or self._board_runner or self._captures_active:
                raise AdbError('Stop the other game workflow before resetting Wheating.')
            runner = self._wheating_runner
            if runner:
                runner.cancel()
            serial = self.connected_serial
            original_client = self.client

        # Cancellation closes the worker's ADB client, but its owning thread
        # still has to finish its finally block before state can be archived.
        deadline = time.monotonic()+15.
        while True:
            with self._lock:
                active = self._wheating_runner
            if active is None:
                break
            if time.monotonic() >= deadline:
                raise AdbError('Wheating did not stop cleanly; the reset was not applied.')
            time.sleep(.05)

        device_root = None
        if serial:
            device = 'device_'+hashlib.sha256(serial.encode()).hexdigest()[:16]
            device_root = self.data.root/'diagnostics'/'wheating'/device
            if device_root.exists():
                stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')
                archive = device_root.parent/'reset_archive'/f'{stamp}_{device}'
                archive.mkdir(parents=True, exist_ok=True)
                for child in list(device_root.iterdir()):
                    shutil.move(str(child), str(archive/child.name))

        # Restart only the Hay Day package.  A direct Android launch avoids
        # opening Recent Apps or changing the desktop's foreground window.
        if serial and original_client:
            client = AdbClient(self.settings.adb_path, serial=serial, timeout_seconds=30)
            try:
                if client.hay_day_running():
                    client.force_stop_hay_day()
                    stop_deadline = time.monotonic()+10.
                    while client.hay_day_running():
                        if time.monotonic() >= stop_deadline:
                            raise AdbError('Hay Day did not stop during the manual reset.')
                        time.sleep(.2)
                client.home()
                client.launch_hay_day()
            finally:
                client.close()

        with self._lock:
            self.last_wheating_run = None
            self.wheating_progress = {
                'message': 'Wheating reset. Hay Day is restarting with a clean state.'
            }
            self.data.log('INFO', 'Wheating state and device session reset.', 'wheating')
        return 'Wheating state reset. Hay Day is restarting with a clean state.'

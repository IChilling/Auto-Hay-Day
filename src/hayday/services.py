"""Device sessions, board diagnostics and explicitly started order workflows."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from hayday.adb import (
    AdbClient,
    AdbError,
    Device,
    Screenshot,
    discover_adb_executables,
    discover_bluestacks_instances,
)
from hayday.emulators import EmulatorConnection, adb_provider, discover_mumu_instances
from hayday.storage import AppData, Settings
from hayday.wheating_services import WheatingServices


class Services(WheatingServices):
    def __init__(self, root: Path | None = None):
        self.data = AppData(root)
        self.settings = self.data.load_settings()
        self.devices: list[Device] = []
        self.instances: list[EmulatorConnection] = []
        self.client: AdbClient | None = None
        self.frame: Screenshot | None = None
        self.frame_serial = ""
        self.connected_serial = ""
        self._lock = threading.RLock()
        self._closed = False
        self._board_runner = None
        self._order_runner = None
        self._init_wheating()
        self._captures_active = 0
        self.last_board_test = None
        self.last_order_run = None
        self.order_progress: dict = {}
        self.data.log(
            "INFO",
            "Workspace opened. Order sessions and board diagnostics start only on request.",
        )
        for warning in self.data.warnings:
            self.data.log("WARNING", warning, "settings")

    def discover(self) -> str:
        # Keep local discovery and client creation together so disconnect cannot
        # occur between them and then be undone by a delayed discovery result.
        with self._lock:
            self._ensure_open()
            executables = discover_adb_executables()
            bluestacks = discover_bluestacks_instances()
            settings = self.settings
            self.instances = discover_mumu_instances(Path(settings.adb_path) if settings.adb_path else None)
            blue_adb = next((item.path for item in executables if 'BlueStacks' in item.source), None)
            if blue_adb:
                self.instances.extend(EmulatorConnection('BlueStacks', item.name, item.display_name,
                                      blue_adb, item.endpoint) for item in bluestacks)
            providers = {adb_provider(item.path) for item in executables}
            new_path = not settings.adb_path
            if new_path and executables and len(providers) == 1:
                settings = replace(settings, adb_path=str(executables[0].path))
            matching = [item for item in self.instances
                        if settings.adb_path and item.adb_path == Path(settings.adb_path).resolve()]
            if new_path and len(matching) == 1 and not settings.selected_serial:
                settings = replace(settings, endpoint=matching[0].endpoint)
            self.save_settings(settings)
            if not settings.adb_path:
                return "Choose a discovered emulator instance or browse to an ADB executable."
            serial = self.connected_serial
            client = self._new_client(serial)
        self._refresh(client, serial)
        return f"Discovery complete. {len(self.devices)} device(s) found."

    def connect_instance(self, key: str) -> str:
        """Select the installation and connect only the explicitly chosen instance."""
        with self._lock:
            self._ensure_open()
            instance = next((item for item in self.instances if item.key == key), None)
            if instance is None:
                raise AdbError('That emulator instance is no longer available. Discover again.')
            self.disconnect()
            self.save_settings(replace(self.settings, adb_path=str(instance.adb_path),
                                       endpoint=instance.endpoint, selected_serial=''))
            client = self._new_client()
        try:
            client.connect(instance.endpoint)
            self._refresh(client, '')
            with self._lock:
                self._ensure_current(client)
                if not any(device.serial == instance.endpoint and device.online for device in self.devices):
                    raise AdbError('The selected emulator endpoint is not online.')
                client.serial = instance.endpoint
            frame = client.capture()
            with self._lock:
                self._ensure_current(client)
                self.save_settings(replace(self.settings, selected_serial=instance.endpoint))
                self.frame, self.frame_serial = frame, instance.endpoint
                self.connected_serial = instance.endpoint
            return f'Connected to {instance.emulator}: {instance.display_name} ({instance.endpoint}).'
        except Exception:
            self._discard_client(client)
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise AdbError("Workspace is closed")

    def _ensure_current(self, client: AdbClient) -> None:
        self._ensure_open()
        if self.client is not client:
            raise AdbError("ADB operation cancelled because the session changed.")

    def _new_client(self, serial: str = "") -> AdbClient:
        with self._lock:
            self._ensure_open()
            self.disconnect()
            self.devices = []
            if not self.settings.adb_path:
                raise AdbError("Choose an ADB executable on the Emulators page first.")
            self.client = AdbClient(self.settings.adb_path, serial=serial)
            return self.client

    def _discard_client(self, client: AdbClient) -> None:
        with self._lock:
            # An older worker's failure must not disconnect a newer selection.
            if self.client is client:
                self.disconnect()

    def _refresh(self, client: AdbClient, serial: str) -> None:
        try:
            devices = client.list_devices()
            with self._lock:
                self._ensure_current(client)
                self.devices = devices
                self.connected_serial = (
                    serial
                    if any(device.serial == serial and device.online for device in devices)
                    else ""
                )
                client.serial = self.connected_serial
        except Exception:
            self._discard_client(client)
            raise

    def refresh_devices(self) -> str:
        with self._lock:
            serial = self.connected_serial
            client = self._new_client(serial)
        self._refresh(client, serial)
        return f"Found {len(self.devices)} device(s). Select an online instance to use it."

    def connect_endpoint(self, endpoint: str) -> str:
        with self._lock:
            serial = self.connected_serial
            client = self._new_client(serial)
        try:
            result = client.connect(endpoint.strip())
            self._refresh(client, serial)
            with self._lock:
                self._ensure_current(client)
                self.save_settings(replace(self.settings, endpoint=endpoint.strip()))
            return result
        except Exception:
            self._discard_client(client)
            raise

    def select_device(self, serial: str) -> str:
        client = self._new_client(serial)
        try:
            devices = client.list_devices()
            with self._lock:
                self._ensure_current(client)
                self.devices = devices
            if not any(device.serial == serial and device.online for device in devices):
                raise AdbError("That device is not online. Refresh and select an online instance.")
            frame = client.capture()
            with self._lock:
                self._ensure_current(client)
                self.save_settings(replace(self.settings, selected_serial=serial))
                self.frame, self.frame_serial = frame, serial
                self.connected_serial = serial
            return f"Connected to {serial}. Captured {frame.width} × {frame.height}."
        except Exception:
            self._discard_client(client)
            raise

    def capture(self) -> Screenshot:
        with self._lock:
            self._ensure_open()
            if self._board_runner or self._order_runner or self._wheating_runner:
                raise AdbError("A game workflow is running. Stop it before capturing manually.")
            client, serial = self.client, self.connected_serial
            if not serial or not client:
                raise AdbError("Select an online emulator device first.")
            self._captures_active += 1
        try:
            frame = client.capture()
            with self._lock:
                self._ensure_current(client)
                self.frame, self.frame_serial = frame, serial
            return frame
        except Exception:
            self._discard_client(client)
            raise
        finally:
            with self._lock:
                self._captures_active -= 1

    def save_frame(self) -> Path:
        with self._lock:
            self._ensure_open()
            if not self.frame:
                raise ValueError("Capture a device screen before saving.")
            return self.data.save_capture(self.frame.png)

    def save_settings(self, settings: Settings) -> None:
        with self._lock:
            self._ensure_open()
            self.data.save_settings(settings)
            self.settings = settings

    def test_quest_board(self, cancel_event: threading.Event | None = None):
        from hayday.board_test import BoardTestRunner

        with self._lock:
            self._ensure_open()
            if self._board_runner:
                raise AdbError("A board test is already running.")
            if self._order_runner or self._wheating_runner or self._captures_active:
                raise AdbError("Another game workflow or capture is running. Stop it first.")
            serial = self.connected_serial
            original_client = self.client
            if not serial or not original_client:
                raise AdbError("Select an online emulator device first.")
            # The test owns a fixed serial; changing the normal session cancels it.
            client = AdbClient(self.settings.adb_path, serial=serial)
            runner = BoardTestRunner(
                client, self.data.root / "diagnostics" / "quest_board", cancel_event=cancel_event
            )
            self._board_runner = runner
        try:
            result = runner.run()
            with self._lock:
                self._ensure_open()
                self.last_board_test = result
                if (
                    self.client is original_client
                    and self.connected_serial == serial
                    and result.frame
                ):
                    self.frame, self.frame_serial = result.frame, serial
                self.data.log(
                    "INFO" if result.success else "WARNING", result.message, "quest_board"
                )
                if result.diagnostics:
                    self.data.log(
                        "INFO", f"Board test evidence: {result.diagnostics}", "quest_board"
                    )
            return result
        finally:
            with self._lock:
                if self._board_runner is runner:
                    self._board_runner = None

    def cancel_board_test(self) -> None:
        with self._lock:
            if self._board_runner:
                self._board_runner.cancel()

    def start_orders(
        self, cancel_event: threading.Event | None = None,
        progress: Callable[[dict], None] | None = None,
        max_deliveries: int = 10, max_seconds: float = 1200,
    ):
        from hayday.orders import OrderRunner

        with self._lock:
            self._ensure_open()
            if self._order_runner or self._board_runner or self._wheating_runner or self._captures_active:
                raise AdbError("Another game workflow or capture is already running. Stop it first.")
            original_client, serial = self.client, self.connected_serial
            if not serial or not original_client:
                raise AdbError("Select an online emulator device first.")
            client = AdbClient(self.settings.adb_path, serial=serial)

            def publish(update: dict) -> None:
                # Runner callbacks arrive on a worker thread. Store state under
                # the session lock; the UI callback only queues its own update.
                with self._lock:
                    if self._closed or self.client is not original_client or (
                        self.connected_serial != serial
                    ):
                        return
                    self.order_progress = dict(update)
                    if update.get("frame"):
                        self.frame, self.frame_serial = update["frame"], serial
                if progress:
                    progress(dict(update))

            try:
                runner = OrderRunner(
                    client, self.data.root / "diagnostics" / "orders",
                    cancel_event=cancel_event, progress=publish,
                    max_deliveries=max_deliveries, max_seconds=max_seconds,
                )
            except Exception:
                client.close()
                raise
            self._order_runner = runner
            self.order_progress = {"message": "Starting order session…", "deliveries": 0}
        try:
            result = runner.run()
            with self._lock:
                if self._closed:
                    return result
                self.last_order_run = result
                if self.client is original_client and self.connected_serial == serial:
                    if result.frame:
                        self.frame, self.frame_serial = result.frame, serial
                    self.order_progress = {
                        "message": result.message, "deliveries": result.deliveries,
                        "status": result.status,
                    }
                self.data.log("INFO" if result.success else "WARNING", result.message, "orders")
                if result.diagnostics:
                    self.data.log("INFO", f"Order session evidence: {result.diagnostics}", "orders")
            return result
        finally:
            client.close()
            with self._lock:
                if self._order_runner is runner:
                    self._order_runner = None

    def cancel_orders(self) -> None:
        with self._lock:
            if self._order_runner:
                self._order_runner.cancel()

    def disconnect(self) -> None:
        with self._lock:
            self.cancel_board_test()
            self.cancel_orders()
            self.cancel_wheating()
            if self.client:
                self.client.close()
                self.client = None
            self.connected_serial = ""

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.disconnect()
            self.data.close()

r"""Mount every route in the real native Flet client using isolated, fake data.

Run from the project root with ``.venv\Scripts\python.exe scripts\ui_smoke.py``.
The smoke window closes automatically. No ADB command is ever issued.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import flet as ft
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hayday.adb import Device, Screenshot  # noqa: E402
from hayday.emulators import EmulatorConnection  # noqa: E402
from hayday.services import Services  # noqa: E402
from hayday.ui import BG, REFERENCE, MainWindow  # noqa: E402

ROUTES = ("overview", "devices", "captures", "activity", "settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds-per-route", type=float, default=0.8)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--screenshots", type=Path, help="Optional route screenshot directory")
    parser.add_argument("--hidden", action="store_true", help="Keep the native smoke window hidden")
    arguments = parser.parse_args()
    if arguments.seconds_per_route < 0.1:
        parser.error("--seconds-per-route must be at least 0.1")
    report = {"passed": False, "routes": [], "errors": []}
    if arguments.screenshots:
        arguments.screenshots.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hayday-ui-smoke-") as directory:

        async def smoke(page: ft.Page):
            services = Services(Path(directory))
            window = None

            def record_error(event):
                report["errors"].append(str(getattr(event, "data", event)))

            page.on_error = record_error
            try:
                window = MainWindow(page, services)
                screenshot = None
                if arguments.screenshots:
                    page.remove(window.shell)
                    screenshot = ft.Screenshot(
                        content=ft.Container(window.shell, bgcolor=BG, expand=True), expand=True
                    )
                    page.add(screenshot)
                await asyncio.sleep(arguments.seconds_per_route)
                await window.initialize()
                for state in ("empty", "populated", "multi-running"):
                    if state == "populated":
                        buffer = io.BytesIO()
                        Image.new("RGB", (960, 540), "#377651").save(buffer, format="PNG")
                        services.frame = Screenshot(
                            buffer.getvalue(), 960, 540, "2026-09-06T04:00:00+00:00"
                        )
                        services.connected_serial = services.frame_serial = "emulator-5554"
                        services.devices = [
                            Device("emulator-5554", "device", "Smoke test emulator"),
                            Device("127.0.0.1:5565", "offline", "Offline test emulator"),
                        ]
                        services.instances = [
                            EmulatorConnection('MuMu', '0', 'Android Device', Path('mumu/adb.exe'), '127.0.0.1:16384'),
                            EmulatorConnection('BlueStacks', 'Pie64', 'Farm', Path('blue/HD-Adb.exe'), '127.0.0.1:5555'),
                            EmulatorConnection('MuMu', '2', 'Android Device-1-1', Path('mumu/adb.exe'), '127.0.0.1:16448'),
                        ]
                        services.data.save_capture(services.frame.png)
                        window.reference_shown = False
                    if state == 'multi-running':
                        services.select_wheating_instances([i.key for i in services.instances])
                        services.wheating_progress = {'instances': {
                            instance.key: dict(name=instance.display_name, serial=instance.endpoint,
                                status='running', planted=51 if index == 0 else 9,
                                listed=51 if index == 0 else 9, message='Wheat is growing',
                                frame=services.frame)
                            for index, instance in enumerate(services.instances)}}
                        window.wheating_running = True
                        window.wheating_planted = window.wheating_listed = 69
                    for route in ROUTES:
                        window.navigate(route)
                        await asyncio.sleep(arguments.seconds_per_route)
                        if state == 'multi-running' and route == 'overview':
                            assert len(window._fleet_buttons) == 3
                            assert all(not view.disabled and not stop.disabled
                                       for view, stop in window._fleet_buttons.values())
                            assert not window.wheating_reset_button.visible
                            window._fleet_buttons[services.instances[1].key][0].on_click(None)
                            assert services.frame_serial == services.instances[1].endpoint
                        if screenshot:
                            png = await screenshot.capture(pixel_ratio=1.0)
                            (arguments.screenshots / f"{state}-{route}.png").write_bytes(png)
                        report["routes"].append(f"{state}/{route}")
                window.wheating_running = False
                report["passed"] = not report["errors"]
            except Exception:
                report["errors"].append(traceback.format_exc())
            finally:
                try:
                    if window:
                        await window.shutdown()
                    else:
                        services.close()
                except Exception:
                    report["errors"].append(traceback.format_exc())
                    report["passed"] = False
                await page.window.destroy()

        with (
            patch("hayday.services.discover_adb_executables", return_value=[]),
            patch("hayday.services.discover_bluestacks_instances", return_value=[]),
            patch("hayday.services.discover_mumu_instances", return_value=[]),
            patch(
                "hayday.services.AdbClient",
                side_effect=AssertionError("Smoke test must not issue ADB commands"),
            ),
        ):
            ft.run(smoke, assets_dir=str(REFERENCE.parent),
                   view=ft.AppView.FLET_APP_HIDDEN if arguments.hidden else ft.AppView.FLET_APP)

    output = json.dumps(report, indent=2)
    print(output)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(output + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

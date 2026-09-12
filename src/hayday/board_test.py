"""One user-requested board-opening test, with visual checks before and after input."""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from hayday.adb import AdbClient, Screenshot
from hayday.panel import PanelVerifier
from hayday.tutorials import TutorialDismissal
from hayday.vision import BoardDetector


class TestCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class BoardTestResult:
    status: str
    message: str
    tapped: bool = False
    diagnostics: Path | None = None
    frame: Screenshot | None = field(default=None, repr=False)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"opened", "already_open"}


class BoardTestRunner:
    """One board tap, plus separately verified free-fuel tutorial dismissal."""

    def __init__(
        self,
        client: AdbClient,
        diagnostics_root: Path,
        *,
        detector=None,
        verifier=None,
        verification_timeout: float = 8.0,
        poll_interval: float = 0.35,
        cancel_event: threading.Event | None = None,
        tutorials=None,
        reconnect=None,
    ):
        self.client = client
        self.diagnostics_root = diagnostics_root
        self.detector = detector
        self.verifier = verifier
        self.verification_timeout = verification_timeout
        self.poll_interval = poll_interval
        self.tutorials = tutorials
        self.reconnect = reconnect
        self._cancelled = cancel_event if cancel_event is not None else threading.Event()
        self._input_gate = threading.Lock()
        self._tapped = False
        self._maintenance = None
        self._launcher = None
        self._frame: Screenshot | None = None
        self._directory: Path | None = None
        self._details: dict[str, Any] = {"serial": client.serial, "max_taps": 1,
                                       "max_board_taps": 1, "max_fuel_taps_per_appearance": 1}

    def cancel(self) -> None:
        # Mark first, so cancellation can interrupt vision work while a command exits.
        self._cancelled.set()
        self.client.close()

    def _checkpoint(self) -> None:
        if self._cancelled.is_set():
            raise TestCancelled("Board test cancelled.")

    def _capture(self, name: str) -> Screenshot:
        frame = self._capture_runtime(name)
        if self._maintenance is None:
            from hayday.reconnect import MaintenanceRecovery
            self._maintenance = MaintenanceRecovery(
                self.client, capture=lambda: self._capture_runtime('maintenance-observation'),
                cancel_event=self._cancelled, check=self._checkpoint,
                save=lambda label, image: self._save_tutorial_frame(label, image))
        frame = self._maintenance.process(frame)
        self._details['maintenance'] = self._maintenance.events
        if self.reconnect is None:
            from hayday.reconnect import ReconnectRecovery

            self.reconnect = ReconnectRecovery(
                self.client,
                capture=lambda: self._capture_runtime('reconnect-observation'),
                cancel_event=self._cancelled,
                check=self._checkpoint,
                save=lambda label, image: self._save_tutorial_frame(label, image),
            )
        frame = self.reconnect.process(frame)
        self._details['reconnect'] = self.reconnect.events
        if self.tutorials is None:
            self.tutorials = TutorialDismissal(
                self.client, capture=lambda: self._capture_raw('fuel-tutorial-observation'),
                cancel_event=self._cancelled, check=self._checkpoint,
                save=lambda label, image: self._save_tutorial_frame(label, image),
            )
        self._details['tutorials'] = self.tutorials.events
        return self.tutorials.process(frame)

    def _save_tutorial_frame(self, name, frame):
        if self._directory:
            (self._directory / f'{name}.png').write_bytes(frame.png)

    def _capture_runtime(self, name: str) -> Screenshot:
        frame = self._capture_raw(name)
        if self._launcher is None:
            from hayday.launcher import LaunchRecovery
            self._launcher = LaunchRecovery(
                self.client, capture=lambda: self._capture_raw('relaunch-observation'),
                cancel_event=self._cancelled, check=self._checkpoint,
                save=lambda label, image: self._save_tutorial_frame(label, image))
        frame = self._launcher.process(frame)
        self._details['launcher'] = self._launcher.events
        return frame

    def _capture_raw(self, name: str) -> Screenshot:
        self._checkpoint()
        frame = self.client.capture()
        self._checkpoint()
        self._frame = frame
        if self._directory:
            (self._directory / f"{name}.png").write_bytes(frame.png)
        return frame

    def _finish(self, status: str, message: str) -> BoardTestResult:
        tutorial_events = self._details.get('tutorials', [])
        tutorial_taps = sum(event['stage'] == 'tap_attempted' for event in tutorial_events)
        if tutorial_taps:
            message = message.replace('No tap sent', 'No board tap sent').replace(
                'No further taps were sent', 'No further board taps were sent')
            dismissed = sum(event['stage'] == 'dismissed' for event in tutorial_events)
            message += (f' Fuel tutorial dismissal confirmed {dismissed} time(s).'
                        if dismissed == tutorial_taps else ' Fuel tutorial dismissal remains unconfirmed.')
        result = BoardTestResult(
            status, message, self._tapped, self._directory, self._frame, self._details
        )
        if self._directory:
            report = {
                "status": status,
                "message": message,
                "tapped": self._tapped,
                "finished_at": datetime.now(UTC).isoformat(),
                **self._details,
            }
            try:
                (self._directory / "result.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
                )
            except (OSError, ValueError) as exc:
                result = BoardTestResult(
                    status,
                    f"{message} Could not save result report: {exc}",
                    self._tapped,
                    self._directory,
                    self._frame,
                    self._details,
                )
        return result

    def _overlay(self, frame: Screenshot, match) -> None:
        if not self._directory:
            return
        image = Image.open(io.BytesIO(frame.png)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (match.x, match.y, match.x + match.width, match.y + match.height),
            outline="#00FFAA",
            width=max(2, frame.width // 500),
        )
        r = max(5, frame.width // 240)
        draw.ellipse(
            (match.tap_x - r, match.tap_y - r, match.tap_x + r, match.tap_y + r),
            outline="#FF4040",
            width=3,
        )
        draw.text((match.x, max(0, match.y - 20)), f"Board score {match.score:.3f}", fill="#00FFAA")
        image.save(self._directory / "board-detected.png")

    def run(self) -> BoardTestResult:
        try:
            self._checkpoint()
            name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self._directory = self.diagnostics_root / name
            self._directory.mkdir(parents=True)
            self.detector = self.detector or BoardDetector()
            self.verifier = self.verifier or PanelVerifier()
            reference_error = getattr(self.verifier, "reference_error", "")
            if reference_error:
                raise ValueError(reference_error)
            before = self._capture("before")
            panel = self.verifier.verify(before.png, cancel=self._cancelled.is_set)
            self._checkpoint()
            if panel.verified:
                if self._cancelled.wait(self.poll_interval):
                    self._checkpoint()
                again = self._capture("already-open-confirmation")
                confirmation = self.verifier.verify(again.png, cancel=self._cancelled.is_set)
                self._checkpoint()
                if confirmation.verified and (before.width, before.height) == (
                    again.width,
                    again.height,
                ):
                    self._details["panel"] = asdict(confirmation)
                    return self._finish(
                        "already_open",
                        "Truck-order panel is already open and visually confirmed. No tap sent.",
                    )
                return self._finish(
                    "changed",
                    "The screen changed during the test. No tap sent; return to the farm and test again.",
                )
            report = self.detector.detect(before.png, cancel=self._cancelled.is_set)
            self._checkpoint()
            self._details["detection"] = asdict(report)
            if not report.match:
                return self._finish(
                    "not_found", f"No safe board match: {report.reason} No tap sent."
                )
            self._overlay(before, report.match)
            fresh = self._capture("before-tap")
            if (before.width, before.height) != (fresh.width, fresh.height):
                return self._finish(
                    "changed", "Device resolution changed during the test. No tap sent."
                )
            checked = self.detector.revalidate(
                fresh.png, report.match, cancel=self._cancelled.is_set
            )
            self._checkpoint()
            self._details["revalidation"] = asdict(checked)
            match = checked.match
            if not match:
                return self._finish(
                    "changed",
                    "The board moved, became obscured, or could not be revalidated. No tap sent.",
                )
            # The detector's local revalidation must still refer to the original board.
            previous = report.match
            if abs(match.tap_x - previous.tap_x) > max(6, previous.width * 0.08) or abs(
                match.tap_y - previous.tap_y
            ) > max(6, previous.height * 0.08):
                return self._finish(
                    "changed", "Board position changed before tapping. No tap sent."
                )
            with self._input_gate:
                self._checkpoint()
                self._details["tap"] = {"x": match.tap_x, "y": match.tap_y}
                # A command failure can have an uncertain outcome; never retry it.
                self._tapped = True
                self.client.tap(match.tap_x, match.tap_y, width=fresh.width, height=fresh.height)
            self._checkpoint()
            end = time.monotonic() + self.verification_timeout
            confirmations = 0
            attempts = []
            while time.monotonic() < end:
                if self._cancelled.wait(self.poll_interval):
                    self._checkpoint()
                after = self._capture("after")
                if (after.width, after.height) != (fresh.width, fresh.height):
                    return self._finish(
                        "unverified",
                        "Tapped once, but the device resolution changed. Panel opening is unverified.",
                    )
                panel = self.verifier.verify(after.png, cancel=self._cancelled.is_set)
                self._checkpoint()
                attempts.append(asdict(panel))
                self._details["verification"] = attempts
                confirmations = confirmations + 1 if panel.verified else 0
                if confirmations >= 2:
                    self._details["panel"] = asdict(panel)
                    return self._finish(
                        "opened",
                        f"Board found (score {match.score:.3f}), tapped once, and truck-order panel confirmed in two frames.",
                    )
            return self._finish(
                "unverified",
                "Tapped once, but the truck-order panel was not confirmed. No further taps were sent.",
            )
        except Exception as exc:
            if self._cancelled.is_set() or isinstance(exc, TestCancelled):
                return self._finish(
                    "cancelled",
                    "Board test cancelled."
                    + (" A tap had already been attempted." if self._tapped else " No tap sent."),
                )
            return self._finish(
                "error",
                f"Board test failed: {exc}"
                + (
                    " A tap was attempted; its outcome may be uncertain."
                    if self._tapped
                    else " No tap sent."
                ),
            )
        finally:
            self.client.close()

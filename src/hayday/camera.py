"""Observe, zoom, and scan visible farm views to recover the truck order board."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from hayday.adb import AdbClient, Screenshot
from hayday.panel import PanelResult, PanelVerifier
from hayday.vision import BoardDetector, BoardMatch


@dataclass(frozen=True)
class CameraResult:
    status: str
    message: str
    frame: Screenshot | None = None
    match: BoardMatch | None = None
    panel: PanelResult | None = None
    actions: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.status in {"opened", "already_open"}


class CameraCancelled(RuntimeError):
    pass


@dataclass
class FarmScan:
    """Sweep overlapping rows from an observed corner, independent of layout.

    A shortened drag stays on the same row. Two unchanged observations after
    separate drags establish an edge; missing drag origins never count as edges.
    """

    phase: str = 'left_edge'
    row: int = 1
    rightward: bool = True
    unchanged: int = 0

    @property
    def direction(self):
        if self.phase in ('left_edge', 'align_left'):
            return .36, 0.
        if self.phase == 'top_edge':
            return 0., .28
        if self.phase == 'next_row':
            return 0., -.28
        return (-.36 if self.rightward else .36), 0.

    @property
    def description(self):
        return {'left_edge': 'Locating the left camera edge',
                'top_edge': 'Locating the top camera edge',
                'align_left': 'Aligning the first search row',
                'next_row': 'Moving to the next overlapping row'}.get(
                    self.phase, f'Searching farm row {self.row}')

    def observe(self, moved: bool):
        if self.phase == 'done':
            return
        self.unchanged = 0 if moved else self.unchanged+1
        if self.phase == 'next_row' and moved:
            self.phase = 'row'
            self.row += 1
            self.rightward = not self.rightward
        elif self.unchanged >= 2:
            self.unchanged = 0
            self.phase = {'left_edge': 'top_edge', 'top_edge': 'align_left',
                          'align_left': 'row', 'row': 'next_row',
                          'next_row': 'done'}[self.phase]


class CameraNavigator:
    """Bounded observation-driven board recovery; the caller owns the client.

    Gesture distances are viewport fractions, never saved farm coordinates.
    Every candidate must pass the detector's multi-feature revalidation before
    input. This navigator only opens the order panel; it never handles an order.
    """

    def __init__(
        self, client: AdbClient, *, detector=None, verifier=None,
        cancel_event: threading.Event | None = None,
        capture: Callable[[], Screenshot] | None = None,
        progress: Callable[[str], None] | None = None,
        max_zoom_steps: int = 1, max_pan_steps: int = 128,
        settle_seconds: float = 0.35, verification_timeout: float = 8,
        search_timeout: float = 360, allow_dialog_recovery: bool = True,
        dialog_vision=None,
    ):
        if type(allow_dialog_recovery) is not bool:
            raise ValueError("allow_dialog_recovery must be a boolean.")
        for name, value, limit in (("max_zoom_steps", max_zoom_steps, 1), ("max_pan_steps", max_pan_steps, 256)):
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f"{name} must be an integer between 0 and {limit}.")
        for name, value in (("settle_seconds", settle_seconds), ("verification_timeout", verification_timeout), ("search_timeout", search_timeout)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative duration.")
        self.client = client
        self.detector = detector
        self.verifier = verifier
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.capture = capture if capture is not None else client.capture
        self.progress = progress
        self.max_zoom_steps = max_zoom_steps
        self.max_pan_steps = max_pan_steps
        self.settle_seconds = settle_seconds
        self.verification_timeout = verification_timeout
        self.search_timeout = search_timeout
        self.allow_dialog_recovery = allow_dialog_recovery
        self.dialog_vision = dialog_vision
        self._handled_dialogs: set[str] = set()
        self.frame: Screenshot | None = None
        self._serial = client.serial
        self._size: tuple[int, int] | None = None
        self._actions: list[str] = []
        self._deadline = math.inf
        self._tapped = False

    def _check(self):
        if self.cancel_event.is_set():
            raise CameraCancelled("Camera recovery cancelled.")
        if self.client.serial != self._serial or not self._serial:
            raise RuntimeError("Device selection changed during camera recovery.")
        if time.monotonic() >= self._deadline:
            raise TimeoutError("Camera search reached its time limit.")

    def _record(self, message):
        self._actions.append(message)
        if self.progress:
            self.progress(message)

    def _observe(self) -> Screenshot:
        self._check()
        frame = self.capture()
        self._check()
        size = (frame.width, frame.height)
        if self._size is not None and size != self._size:
            raise RuntimeError("Device resolution changed during camera recovery.")
        self._size = size
        self.frame = frame
        return frame

    def _pause(self):
        if self.cancel_event.wait(self.settle_seconds):
            self._check()

    def _settled_observe(self) -> Screenshot:
        """Wait for camera inertia to stop before looking for a tap target."""
        previous = None
        for _ in range(8):
            self._pause()
            frame = self._observe()
            if self._modal_visible(frame):
                return frame
            current = self._view(frame)
            if previous is not None and self._same_view(previous, current):
                return frame
            previous = current
        raise TimeoutError("Camera kept moving after the gesture; no target input was sent.")

    def _result(self, status, message, match=None, panel=None):
        return CameraResult(status, message, self.frame, match, panel, tuple(self._actions))

    @staticmethod
    def _decode(frame: Screenshot) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frame.height, frame.width):
            raise ValueError("Camera recovery requires a valid screenshot with matching dimensions.")
        return image

    @classmethod
    def _view(cls, frame: Screenshot) -> np.ndarray:
        image = cls._decode(frame)
        # Exclude persistent edge HUD controls; blur suppresses crop animation.
        height, width = image.shape[:2]
        image = image[round(height*.16):round(height*.82), round(width*.14):round(width*.86)]
        small = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0)

    @staticmethod
    def _same_view(first: np.ndarray, second: np.ndarray) -> bool:
        difference = np.abs(first.astype(np.float32)-second.astype(np.float32)).mean(axis=2)
        return float(np.median(difference)) < 2.5 and float(np.quantile(difference, .8)) < 7

    @classmethod
    def _modal_visible(cls, frame: Screenshot) -> bool:
        """A large opaque cream dialog is not farm ground, even with grass at its edges."""
        image = cls._decode(frame)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cream = cv2.inRange(hsv, (15, 0, 165), (45, 115, 255))
        h, w = cream.shape
        cream = cv2.morphologyEx(cream, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(cream)
        return any(sw > w*.4 and sh > h*.3 and area > w*h*.16
                   for _, _, sw, sh, area in stats[1:count])

    @classmethod
    def _grass_start(cls, frame: Screenshot, dx: float, dy: float) -> tuple[int, int, int, int] | None:
        """Choose an observed open patch whose entire drag stays away from HUD."""
        image = cls._decode(frame)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (24, 75, 65), (88, 255, 255))
        # Smooth green patches offer a better drag origin than buildings or crops.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 130)
        grass[cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0] = 0
        height, width = grass.shape
        move_x, move_y = round(width*dx), round(height*dy)
        x0, x1 = round(width*.17), round(width*.83)
        y0, y1 = round(height*.19), round(height*.79)
        # Restrict both endpoints; this is viewport geometry, not farm layout.
        left, right = max(x0, x0-move_x), min(x1, x1-move_x)
        top, bottom = max(y0, y0-move_y), min(y1, y1-move_y)
        allowed = np.zeros_like(grass)
        allowed[max(0, top):max(0, bottom), max(0, left):max(0, right)] = 255
        grass &= allowed
        distance = cv2.distanceTransform(grass, cv2.DIST_L2, 3)
        _, radius, _, (x, y) = cv2.minMaxLoc(distance)
        if radius < max(3, min(width, height)*.005):
            return None
        return x, y, x+move_x, y+move_y

    @classmethod
    def _pinch_area(cls, frame: Screenshot) -> tuple[tuple[int, int], int] | None:
        """Place both contacts on observed grass with enough span to zoom."""
        image = cls._decode(frame)
        h, w = image.shape[:2]
        grass = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (24, 75, 65), (88, 255, 255))
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 130)
        grass[cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0] = 0
        allowed = np.zeros_like(grass)
        allowed[round(h*.20):round(h*.75), round(w*.17):round(w*.83)] = 255
        grass &= allowed
        distance = cv2.distanceTransform(grass, cv2.DIST_L2, 3)
        # The middle of a two-finger gesture need not be grass: only its
        # starting contacts can select a building. Tiny pinches can fall below
        # the game's effective zoom threshold despite producing valid events.
        for fraction in (.44, .36, .28, .20, .16):
            span = round(w*fraction)
            if span < 32 or span >= w:
                continue
            clearance = np.minimum(distance[:, :-span], distance[:, span:])
            _, radius, _, (x, y) = cv2.minMaxLoc(clearance)
            if radius >= max(6, min(w, h)*.012):
                return (x+span//2, y), span
        return None

    def _verify_open(self, first_panel: PanelResult | None = None, *, match=None, already=False):
        end = time.monotonic() + self.verification_timeout
        confirmations = 1 if first_panel is not None and first_panel.verified else 0
        while time.monotonic() < end:
            self._pause()
            frame = self._observe()
            panel = self.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            confirmations = confirmations+1 if panel.verified else 0
            if confirmations >= 2:
                return self._result(
                    "already_open" if already else "opened",
                    "Truck-order panel confirmed in two frames.", match, panel,
                )
        return self._result("unverified", "Order panel could not be confirmed. No further board taps were sent.", match)

    def _recover_dialog(self, frame: Screenshot) -> Screenshot | CameraResult:
        """Handle only the two explicitly authorized, freshly verified dialogs."""
        blocked = "An unrecognized dialog covers the farm. Close it before camera recovery."
        if not self.allow_dialog_recovery:
            return self._result("blocked", "Dialog recovery is disabled for this operation. No dialog input sent.")
        if self.dialog_vision is None:
            try:
                from hayday.dialogs import DialogVision

                self.dialog_vision = DialogVision()
            except (ImportError, ValueError):
                return self._result("blocked", blocked)
        first = self.dialog_vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if first is None or first.kind not in {"fuel_tutorial", "connection_lost"}:
            return self._result("blocked", blocked)
        if first.kind in self._handled_dialogs:
            return self._result("blocked", "The same dialog returned after its one recovery attempt. No repeated tap sent.")
        previous = first
        confirmed = None
        for _ in range(4):
            # Tutorials can grow or move during their entrance animation. Wait
            # for two consecutive recognized, stable observations; never reuse
            # a target from before the animation or send a speculative tap.
            self._pause()
            fresh = self._observe()
            current = self.dialog_vision.observe(fresh.png, cancel=self.cancel_event.is_set)
            self._check()
            if current is not None and current.kind != first.kind:
                break
            if current is not None and previous is not None:
                tolerance = max(4, max(previous.target.width, previous.target.height)*.05)
                if (abs(current.target.center[0]-previous.target.center[0]) <= tolerance
                        and abs(current.target.center[1]-previous.target.center[1]) <= tolerance
                        and .96 <= current.target.width/max(1, previous.target.width) <= 1.04
                        and .96 <= current.target.height/max(1, previous.target.height) <= 1.04):
                    confirmed = current
                    break
            previous = current
        if confirmed is None:
            return self._result("changed", "Dialog changed before its recovery control could be confirmed. No tap sent.")
        self._handled_dialogs.add(first.kind)
        self._record("Dismissing the confirmed fuel tutorial." if first.kind == "fuel_tutorial"
                     else "Reconnecting using the confirmed Connection Lost / Try Again control.")
        self._check()
        self.client.tap(*confirmed.target.center, width=fresh.width, height=fresh.height)
        self._check()
        if first.kind == "fuel_tutorial":
            return self._settled_observe()
        # A reconnect can show a loading illustration. Never pan or zoom that
        # picture: wait for the order panel, a detected board, or the known fuel
        # tutorial before returning control to farm navigation.
        end = min(self._deadline, time.monotonic()+20)
        while time.monotonic() < end:
            self._pause()
            frame = self._observe()
            panel = self.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if panel.verified:
                return frame
            if self._modal_visible(frame):
                dialog = self.dialog_vision.observe(frame.png, cancel=self.cancel_event.is_set)
                self._check()
                if dialog is None or dialog.kind != "connection_lost":
                    return frame
                continue
            report = self.detector.detect(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if report.match:
                return frame
        return self._result("blocked", "Reconnect was attempted once, but a recognizable farm or order panel did not return. No camera input sent.")

    def _try_open(self, frame: Screenshot):
        self._check()
        panel = self.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if panel.verified:
            return self._verify_open(panel, already=True)
        if self._modal_visible(frame):
            recovered = self._recover_dialog(frame)
            return recovered if isinstance(recovered, CameraResult) else self._try_open(recovered)
        report = self.detector.detect(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if not report.match:
            return None
        fresh = self._observe()
        if self._modal_visible(fresh):
            recovered = self._recover_dialog(fresh)
            return recovered if isinstance(recovered, CameraResult) else self._try_open(recovered)
        checked = self.detector.revalidate(fresh.png, report.match, cancel=self.cancel_event.is_set)
        self._check()
        match = checked.match
        if not match or abs(match.tap_x-report.match.tap_x) > max(6, report.match.width*.08) or abs(match.tap_y-report.match.tap_y) > max(6, report.match.height*.08):
            return self._result("changed", "Board changed before input; no tap sent.")
        self._record(f"Board found at scale {match.scale:.3f}; opening its order panel.")
        self._check()
        self._tapped = True
        self.client.tap(match.tap_x, match.tap_y, width=fresh.width, height=fresh.height)
        self._check()
        result = self._verify_open(match=match)
        if result.status == "unverified":
            return self._retry_visible_board(match)
        return result

    @staticmethod
    def _same_target(first: BoardMatch, second: BoardMatch) -> bool:
        return (
            abs(first.tap_x-second.tap_x) <= max(8, first.width*.10)
            and abs(first.tap_y-second.tap_y) <= max(8, first.height*.10)
            and .9 <= second.width/max(1, first.width) <= 1.1
            and .9 <= second.height/max(1, first.height) <= 1.1
        )

    def _retry_visible_board(self, previous: BoardMatch) -> CameraResult:
        """Allow one fresh navigation tap when the first collected truck rewards.

        This never retries an order-send action. It requires the original board
        to remain visible and pass a new full search plus local revalidation.
        A failed input command is handled outside this method and never retried.
        """
        frame = self._observe()
        panel = self.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if panel.verified:
            return self._verify_open(panel, match=previous)
        if self._modal_visible(frame):
            recovered = self._recover_dialog(frame)
            if isinstance(recovered, CameraResult):
                return recovered
            frame = recovered
            panel = self.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if panel.verified:
                return self._verify_open(panel, match=previous)
        report = self.detector.detect(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if report.match is None or not self._same_target(previous, report.match):
            return self._result("unverified", "Order panel is unconfirmed and the same board is no longer safely visible. No additional tap sent.", previous)
        fresh = self._observe()
        if self._modal_visible(fresh):
            recovered = self._recover_dialog(fresh)
            if isinstance(recovered, CameraResult):
                return recovered
            fresh = recovered
            panel = self.verifier.verify(fresh.png, cancel=self.cancel_event.is_set)
            self._check()
            if panel.verified:
                return self._verify_open(panel, match=previous)
        checked = self.detector.revalidate(fresh.png, report.match, cancel=self.cancel_event.is_set)
        self._check()
        if checked.match is None or not self._same_target(previous, checked.match):
            return self._result("changed", "Board changed during navigation revalidation. No additional tap sent.", previous)
        match = checked.match
        self._record("The same board remains visible after the first navigation tap; opening it once more.")
        self._check()
        self.client.tap(match.tap_x, match.tap_y, width=fresh.width, height=fresh.height)
        self._check()
        return self._verify_open(match=match)

    def open_board(self) -> CameraResult:
        """Find/open the board, recovering camera position only when needed."""
        self._deadline = time.monotonic() + self.search_timeout
        self._actions = []
        self._size = None
        self._tapped = False
        self._handled_dialogs = set()
        self.frame = None
        try:
            self._check()
            self.detector = self.detector or BoardDetector()
            self.verifier = self.verifier or PanelVerifier()
            error = getattr(self.verifier, "reference_error", "")
            if error:
                raise ValueError(error)
            frame = self._observe()
            result = self._try_open(frame)
            if result:
                return result
            # A smooth grass origin is required before issuing camera gestures.
            # Unknown overlays with no usable farm surface stop recovery.
            if self._grass_start(frame, 0, 0) is None:
                return self._result("blocked", "No clear farm surface is visible for camera recovery.")
            if self.max_zoom_steps:
                pinch = self._pinch_area(frame)
                if pinch is None:
                    self._record("No clear two-finger zoom origin; scanning at the current zoom.")
                else:
                    self._record("Zooming out once before searching the farm.")
                    self._check()
                    self.client.pinch_zoom_out(width=frame.width, height=frame.height,
                                              center=pinch[0], span=pinch[1], cancel_event=self.cancel_event)
                    frame = self._settled_observe()
            result = self._try_open(frame)
            if result:
                return result
            scan = FarmScan()
            for index in range(1, self.max_pan_steps+1):
                self._check()
                if self._modal_visible(frame):
                    recovered = self._recover_dialog(frame)
                    if isinstance(recovered, CameraResult):
                        return recovered
                    frame = recovered
                    result = self._try_open(frame)
                    if result:
                        return result
                dx, dy = scan.direction
                drag = None
                for fraction in (1., .5, .25):
                    drag = self._grass_start(frame, dx*fraction, dy*fraction)
                    if drag is not None:
                        break
                if drag is None:
                    return self._result('blocked', 'No clear drag origin for the next search row; farm scan is incomplete.')
                before_view = self._view(frame)
                self._record(f"{scan.description} ({index}/{self.max_pan_steps}).")
                self._check()
                self.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=250)
                frame = self._settled_observe()
                current = self._view(frame)
                scan.observe(moved=not self._same_view(before_view, current))
                # Search even revisited views: a previously clipped board or
                # one hidden by an animation can now be recognizable.
                result = self._try_open(frame)
                if result:
                    return result
                if scan.phase == 'done':
                    return self._result('not_found', f'Board not found after sweeping {scan.row} rows between observed camera edges.')
            return self._result("not_found", "Board not found before the camera gesture limit; farm scan is incomplete.")
        except Exception as exc:
            if self.cancel_event.is_set() or isinstance(exc, CameraCancelled):
                return self._result("cancelled", "Camera recovery cancelled.")
            status = "unverified" if self._tapped else "blocked" if isinstance(exc, TimeoutError) else "error"
            return self._result(status, f"Camera recovery stopped: {exc}")

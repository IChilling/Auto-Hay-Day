"""Reconnect only through the exact observed Connection Lost / TRY AGAIN dialog."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from hayday.adb import Screenshot
from hayday.dialogs import DialogObservation, DialogVision


class ReconnectBlocked(RuntimeError):
    """The connection dialog could not be handled without uncertain input."""


class ReconnectRecovery:
    """Bounded, visually confirmed TRY AGAIN handling for one fixed device.

    The initial frame is only a trigger. Two later, independently captured
    frames must show the same exact dialog and TRY AGAIN target before every
    tap. A returned tap command may be repeated only while the exact dialog is
    still positively visible, and never more than ``max_attempts`` per popup.
    """

    dialog_kind = 'connection_lost'

    def __init__(
        self,
        client,
        *,
        capture: Callable[[], Screenshot],
        cancel_event: threading.Event | None = None,
        check=None,
        wait=None,
        save=None,
        progress=None,
        vision=None,
        ready=None,
        max_attempts: int = 3,
    ):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError("Reconnect attempts must be an integer between 1 and 5.")
        self.client = client
        self.serial = client.serial
        self.capture = capture
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.check = check or (lambda: None)
        self.wait = wait or self.cancel_event.wait
        self.save = save or (lambda label, frame: None)
        self.progress = progress or (lambda message: None)
        self.vision = vision
        self.ready = ready or self._possible_playable_scene
        self.max_attempts = max_attempts
        self.pause_seconds = 0.5
        self.confirmation_checks = 6
        self.clearance_checks = 8
        self.readiness_checks = 40
        self.events: list[dict] = []
        self._uncertain = False

    def _check(self):
        self.check()
        if self.cancel_event.is_set():
            raise ReconnectBlocked("Reconnection was cancelled.")
        if self.client.serial != self.serial or not self.serial:
            raise ReconnectBlocked("Device selection changed during reconnection.")

    @staticmethod
    def _possible_connection(frame: Screenshot) -> bool:
        """Cheap rejection only; it never authorizes a tap."""
        try:
            image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), cv2.IMREAD_COLOR)
        except (TypeError, ValueError, cv2.error):
            return False
        if image is None or image.shape[:2] != (frame.height, frame.width):
            return False
        ratio = min(1.0, 640 / image.shape[1])
        image = cv2.resize(
            image,
            (round(image.shape[1] * ratio), round(image.shape[0] * ratio)),
            interpolation=cv2.INTER_AREA,
        )
        if image.std(axis=(0, 1)).mean() < 10:
            return False
        cream = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (15, 0, 160), (45, 125, 255))
        cream = cv2.morphologyEx(cream, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(cream)
        height, width = cream.shape
        return any(
            w > width * 0.52 and h > height * 0.48 and area > width * height * 0.25
            for _, _, w, h, area in stats[1:count]
        )

    @staticmethod
    def _possible_playable_scene(frame: Screenshot) -> bool:
        """Reject the observed blue loading illustration before automation resumes.

        This is only a return gate and never authorizes device input. Farm views
        and the post-reconnect fuel tutorial both retain broad green scenery;
        the loading logo does not.
        """
        try:
            image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), cv2.IMREAD_COLOR)
        except (TypeError, ValueError, cv2.error):
            return False
        if image is None or image.shape[:2] != (frame.height, frame.width):
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (24, 55, 45), (90, 255, 255))
        height, width = grass.shape
        region = grass[round(height * .12):round(height * .88),
                       round(width * .08):round(width * .92)]
        if not (region.size and np.count_nonzero(region) / region.size >= .15):
            return False
        from hayday.game_scene import farm_scene_vision
        return farm_scene_vision().ready(frame.png)

    def _observe(self, frame: Screenshot) -> DialogObservation | None:
        self._check()
        if self.vision is None:
            if not self._possible_connection(frame):
                return None
            try:
                self.vision = DialogVision()
            except (ImportError, ValueError) as exc:
                raise ReconnectBlocked(
                    "Connection Lost references are unavailable; TRY AGAIN was not tapped."
                ) from exc
        elif isinstance(self.vision, DialogVision) and not self._possible_connection(frame):
            return None
        result = self.vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        return result if result is not None and result.kind == self.dialog_kind else None

    # Subclasses reuse fresh-frame confirmation and the same device bounds.

    def _fresh(self, previous: Screenshot) -> Screenshot:
        self._check()
        self.wait(self.pause_seconds)
        self._check()
        frame = self.capture()
        self._check()
        if (frame.width, frame.height) != (previous.width, previous.height):
            raise ReconnectBlocked("Device resolution changed during reconnection.")
        return frame

    @staticmethod
    def _fresh_capture(previous: Screenshot, current: Screenshot) -> bool:
        return bool(current.captured_at) and current.captured_at != previous.captured_at

    @staticmethod
    def _same(first: DialogObservation | None, second: DialogObservation | None) -> bool:
        if first is None or second is None or first.kind != second.kind:
            return False
        if len(first.features) != len(second.features) or tuple(
            feature.name for feature in first.features
        ) != tuple(feature.name for feature in second.features):
            return False
        tolerance = max(4, min(first.target.width, first.target.height) * 0.04)
        targets = [(first.target, second.target), (first.bounds, second.bounds)]
        targets.extend(
            (left.target, right.target)
            for left, right in zip(first.features, second.features, strict=True)
        )
        for left, right in targets:
            if isinstance(left, tuple):
                lx, ly, lw, lh = left
                rx, ry, rw, rh = right
                left_center = lx + lw / 2, ly + lh / 2
                right_center = rx + rw / 2, ry + rh / 2
            else:
                lw, lh, rw, rh = left.width, left.height, right.width, right.height
                left_center, right_center = left.center, right.center
            if (
                any(abs(a - b) > tolerance for a, b in zip(left_center, right_center, strict=True))
                or not 0.96 <= rw / max(1, lw) <= 1.04
                or not 0.96 <= rh / max(1, lh) <= 1.04
            ):
                return False
        return True

    @staticmethod
    def _valid_target(observation: DialogObservation, frame: Screenshot) -> bool:
        target = observation.target
        x, y = target.center
        bx, by, bw, bh = observation.bounds
        return (
            target.width > 0
            and target.height > 0
            and target.x >= 0
            and target.y >= 0
            and target.x + target.width <= frame.width
            and target.y + target.height <= frame.height
            and bx <= x < bx + bw
            and by <= y < by + bh
        )

    def _confirm(
        self, frame: Screenshot, initial: DialogObservation
    ) -> tuple[Screenshot, DialogObservation | None]:
        """Require two new matching captures; ``None`` means it cleared itself."""
        previous_frame = frame
        previous_observation = None
        clear = 0
        for _ in range(self.confirmation_checks):
            fresh = self._fresh(previous_frame)
            current = self._observe(fresh)
            distinct = self._fresh_capture(previous_frame, fresh)
            clear = clear + 1 if distinct and current is None else 0
            if clear >= 2:
                return fresh, None
            if (
                distinct
                and previous_observation is not None
                and self._same(previous_observation, current)
                and self._same(initial, current)
                and self._valid_target(current, fresh)
            ):
                return fresh, current
            previous_observation = current if distinct else None
            previous_frame = fresh
        raise ReconnectBlocked(
            "Connection Lost / TRY AGAIN did not settle in two fresh matching frames; no tap was sent."
        )

    def _verify_clear(
        self, frame: Screenshot
    ) -> tuple[Screenshot, DialogObservation | None, bool]:
        """Return two-frame clearance, or the last positively recognized dialog."""
        previous_frame = frame
        last_dialog = None
        clear = 0
        for _ in range(self.clearance_checks):
            fresh = self._fresh(previous_frame)
            current = self._observe(fresh)
            distinct = self._fresh_capture(previous_frame, fresh)
            clear = clear + 1 if distinct and current is None else 0
            if current is not None:
                last_dialog = current
                clear = 0
            if clear >= 2:
                return fresh, None, True
            previous_frame = fresh
        return previous_frame, last_dialog, False

    def _verify_ready(
        self, frame: Screenshot,
    ) -> tuple[Screenshot, DialogObservation | None]:
        """Wait for two playable frames or report a returned outage."""
        previous = frame
        ready = 0
        for _ in range(self.readiness_checks):
            ready = ready + 1 if self.ready(previous) else 0
            if ready >= 2:
                return previous, None
            fresh = self._fresh(previous)
            connection = self._observe(fresh)
            if connection is not None:
                return fresh, connection
            previous = fresh
        raise ReconnectBlocked(
            "Connection Lost cleared, but a playable farm scene did not return within 20 seconds."
        )

    def process(self, frame: Screenshot) -> Screenshot:
        """Handle one observed outage, or return the untouched frame."""
        self._check()
        if self._uncertain:
            raise ReconnectBlocked(
                "An earlier TRY AGAIN action remains uncertain; no repeated tap is allowed."
            )
        first = self._observe(frame)
        if first is None:
            return frame
        self.save("connection_lost_detected", frame)
        attempts = 0
        confirmation_rounds = 0
        max_confirmation_rounds = self.max_attempts + 2
        while attempts < self.max_attempts:
            confirmation_rounds += 1
            if confirmation_rounds > max_confirmation_rounds:
                self.save("connection_reappearance_exhausted", frame)
                raise ReconnectBlocked(
                    "Connection Lost repeatedly cleared and returned; recovery stopped without "
                    "an unbounded TRY AGAIN loop."
                )
            self.progress("Confirming Connection Lost and TRY AGAIN before reconnecting.")
            frame, confirmed = self._confirm(frame, first)
            if confirmed is None:
                frame, returned = self._verify_ready(frame)
                if returned is None:
                    return frame
                first = returned
                continue
            attempts += 1
            self.save(f"connection_retry_{attempts}_before_tap", frame)
            self.events.append(
                {
                    "kind": "connection_lost",
                    "stage": "tap_attempted",
                    "attempt": attempts,
                    "captured_at": frame.captured_at,
                    "target": list(confirmed.target.center),
                }
            )
            self._uncertain = True
            self.progress(
                f"Tapping the confirmed TRY AGAIN control ({attempts}/{self.max_attempts})."
            )
            self._check()
            self.client.tap(*confirmed.target.center, width=frame.width, height=frame.height)
            self._check()
            frame, remaining, cleared = self._verify_clear(frame)
            if cleared:
                frame, returned = self._verify_ready(frame)
                if returned is not None:
                    first = returned
                    if attempts < self.max_attempts:
                        self._uncertain = False
                        continue
                    break
                self._uncertain = False
                self.events.append(
                    {
                        "kind": "connection_lost",
                        "stage": "reconnected",
                        "attempt": attempts,
                        "captured_at": frame.captured_at,
                    }
                )
                self.save("connection_recovered", frame)
                return frame
            if remaining is None:
                self.save("connection_recovery_uncertain", frame)
                raise ReconnectBlocked(
                    "TRY AGAIN was tapped, but two fresh frames could not confirm that the dialog cleared."
                )
            first = remaining
            if attempts < self.max_attempts:
                self._uncertain = False
                self.progress("Connection Lost remains visible; preparing a bounded retry.")
        self.save("connection_retry_exhausted", frame)
        raise ReconnectBlocked(
            f"Connection Lost remained after {self.max_attempts} confirmed TRY AGAIN attempts; no more taps were sent."
        )


class MaintenanceRecovery(ReconnectRecovery):
    """Pause on verified maintenance, retry at most once every two minutes."""

    dialog_kind = 'server_maintenance'

    def __init__(self, *args, retry_seconds=120., max_wait_seconds=3600., clock=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (type(retry_seconds) not in (int, float) or not 30 <= retry_seconds <= 900
                or type(max_wait_seconds) not in (int, float)
                or not retry_seconds <= max_wait_seconds <= 7200):
            raise ValueError('Maintenance requires a 30–900 second retry interval and a bounded wait.')
        self.retry_seconds = float(retry_seconds)
        self.max_wait_seconds = float(max_wait_seconds)
        self.clock = clock or time.monotonic
        self.next_retry = None

    def process(self, frame):
        self._check()
        if self._uncertain:
            raise ReconnectBlocked('An earlier RETRY LOGIN action is uncertain; automatic retry is paused.')
        observed = self._observe(frame)
        if observed is None:
            return frame
        started = self.clock()
        deadline = started+self.max_wait_seconds
        self.next_retry = max(self.next_retry or 0., started+self.retry_seconds)
        self.save('maintenance_detected', frame)
        self.progress(f'Hay Day is under maintenance. Farm work is paused; Retry Login every {self.retry_seconds:g} seconds.')
        missing_since = None
        ready_count = 0
        while self.clock() < deadline:
            self._check()
            if observed is not None:
                missing_since = None
                ready_count = 0
                if self.clock() >= self.next_retry:
                    frame, confirmed = self._confirm(frame, observed)
                    if confirmed is not None:
                        self.next_retry = self.clock()+self.retry_seconds
                        self._uncertain = True
                        self.events.append({'kind': self.dialog_kind, 'stage': 'tap_attempted',
                                            'captured_at': frame.captured_at})
                        self.save('maintenance_retry_before', frame)
                        self.progress('Retrying login after the maintenance wait.')
                        self._check()
                        self.client.tap(*confirmed.target.center, width=frame.width, height=frame.height)
                        self._check()
                        self._uncertain = False  # Command returned; the next attempt remains time-gated.
                    observed = self._observe(frame)
            else:
                missing_since = self.clock() if missing_since is None else missing_since
                playable = self.ready(frame) and not self._possible_connection(frame)
                ready_count = ready_count+1 if playable else 0
                if ready_count >= 2:
                    self.events.append({'kind': self.dialog_kind, 'stage': 'reconnected',
                                        'captured_at': frame.captured_at})
                    self.save('maintenance_recovered', frame)
                    self.progress('The farm is available again. Resuming the current work.')
                    return frame
                if self.clock()-missing_since >= 90:
                    raise ReconnectBlocked('Maintenance cleared, but the farm has not returned. No farm input was sent.')
            previous = frame
            delay = min(10., max(.5, self.next_retry-self.clock())) if observed else 1.
            self.wait(delay)
            self._check()
            frame = self.capture()
            self._check()
            if (frame.width, frame.height) != (previous.width, previous.height):
                raise ReconnectBlocked('Device resolution changed during maintenance recovery.')
            if not self._fresh_capture(previous, frame):
                raise ReconnectBlocked('Maintenance recovery requires fresh screenshots.')
            observed = self._observe(frame)
        raise ReconnectBlocked('The maintenance waiting limit was reached. Farm work remains paused.')

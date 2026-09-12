"""Dismiss only the observed free-fuel tutorial, with fresh visual confirmation."""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy as np

from hayday.adb import Screenshot
from hayday.dialogs import DialogVision


class TutorialBlocked(RuntimeError):
    """A known tutorial could not be handled without repeating uncertain input."""


class TutorialDismissal:
    """One tap per confirmed fuel prompt; other dialogs never acquire input.

    ``capture`` must return a new raw screenshot and must not call this helper.
    The caller retains its device, cancellation, deadline, and diagnostic checks.
    A failed or uncleared tap blocks further attempts for this helper's lifetime.
    """

    def __init__(self, client, *, capture: Callable[[], Screenshot],
                 cancel_event: threading.Event | None = None, check=None, wait=None,
                 save=None, progress=None, vision=None):
        self.client = client
        self.serial = client.serial
        self.capture = capture
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.check = check or (lambda: None)
        self.wait = wait or self.cancel_event.wait
        self.save = save or (lambda label, frame: None)
        self.progress = progress or (lambda message: None)
        self.vision = vision
        self.events: list[dict] = []
        self._uncertain = False
        self.pause_seconds = .3

    def _check(self):
        self.check()
        if self.cancel_event.is_set():
            raise TutorialBlocked('Fuel tutorial handling was cancelled.')
        if self.client.serial != self.serial or not self.serial:
            raise TutorialBlocked('Device selection changed during fuel tutorial handling.')

    @staticmethod
    def _possible_fuel(frame):
        # Cheap rejection only: input still requires every exact DialogVision
        # text feature and its dialog geometry on two independent captures.
        try:
            image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, TypeError, cv2.error):
            return False
        if image is None or image.shape[:2] != (frame.height, frame.width):
            return False
        height, width = image.shape[:2]
        ratio = min(1, 640/width)
        image = cv2.resize(image, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cream = cv2.inRange(hsv, (15, 0, 160), (45, 125, 255))
        cream = cv2.morphologyEx(cream, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(cream)
        height, width = cream.shape
        return any(w > width*.35 and h > height*.22 and area > width*height*.12
                   for _, _, w, h, area in stats[1:count])

    def _observe(self, frame):
        self._check()
        if self.vision is None:
            if not self._possible_fuel(frame):
                return None
            try:
                self.vision = DialogVision()
            except (ImportError, ValueError) as exc:
                raise TutorialBlocked('Fuel tutorial references are unavailable; no dismissal was attempted.') from exc
        elif isinstance(self.vision, DialogVision) and not self._possible_fuel(frame):
            return None
        result = self.vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        return result if result is not None and result.kind == 'fuel_tutorial' else None

    def _fresh(self, previous):
        self._check()
        self.wait(self.pause_seconds)
        self._check()
        frame = self.capture()
        self._check()
        if (frame.width, frame.height) != (previous.width, previous.height):
            raise TutorialBlocked('Device resolution changed during fuel tutorial handling.')
        return frame

    @staticmethod
    def _same(first, second):
        if first is None or second is None:
            return False
        tolerance = max(4, min(first.target.width, first.target.height)*.04)
        return (all(abs(a-b) <= tolerance for a, b in zip(first.target.center, second.target.center, strict=True))
                and .96 <= second.target.width/max(1, first.target.width) <= 1.04
                and .96 <= second.target.height/max(1, first.target.height) <= 1.04)

    @staticmethod
    def _valid_target(observation, frame):
        target = observation.target
        x, y = target.center
        return (target.width > 0 and target.height > 0 and target.x >= 0 and target.y >= 0
                and target.x+target.width <= frame.width and target.y+target.height <= frame.height
                and 0 <= x < frame.width and 0 <= y < frame.height)

    def process(self, frame: Screenshot) -> Screenshot:
        self._check()
        if self._uncertain:
            raise TutorialBlocked('An earlier fuel dismissal remains uncertain; no repeated tap is allowed.')
        first = self._observe(frame)
        if first is None:
            return frame
        self.save('fuel_tutorial_detected', frame)
        self.progress('Confirming the free-fuel tutorial before dismissing it.')
        previous = first
        for _ in range(4):
            fresh = self._fresh(frame)
            current = self._observe(fresh)
            distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
            if current is None and distinct:
                # It disappeared without our input. The caller will validate
                # this new scene before doing any work of its own.
                return fresh
            if distinct and self._same(previous, current) and self._valid_target(current, fresh):
                frame = fresh
                break
            previous = current if distinct else None
            frame = fresh
        else:
            raise TutorialBlocked('The fuel tutorial did not settle in two fresh matching frames; no tap was sent.')
        self._check()
        self.save('fuel_tutorial_before_tap', frame)
        self.events.append({'kind': 'fuel_tutorial', 'stage': 'tap_attempted',
                            'captured_at': frame.captured_at, 'target': list(current.target.center)})
        # Mark uncertainty before input, including command errors and cancellation.
        self._uncertain = True
        self.progress('Dismissing the confirmed free-fuel tutorial once.')
        self._check()
        self.client.tap(*current.target.center, width=frame.width, height=frame.height)
        self._check()
        clear = 0
        for _ in range(6):
            fresh = self._fresh(frame)
            current = self._observe(fresh)
            distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
            clear = clear+1 if distinct and current is None else 0
            frame = fresh
            if clear >= 2:
                self._uncertain = False
                self.events.append({'kind': 'fuel_tutorial', 'stage': 'dismissed', 'captured_at': frame.captured_at})
                self.save('fuel_tutorial_dismissed', frame)
                return frame
        self.save('fuel_tutorial_uncleared', frame)
        raise TutorialBlocked('The fuel tutorial remained after its one dismissal attempt; no repeated tap was sent.')

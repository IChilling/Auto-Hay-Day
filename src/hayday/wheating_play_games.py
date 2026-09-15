"""Dismiss the verified Play Games profile invitation without account setup."""
import re

import cv2

from hayday.farming import FarmingWorker
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingPrerequisite


def profile_cancel_target(frame, lines):
    names = ('cancel', 'google play games', 'next', 'create a play games profile')
    labels = {}
    for line in lines:
        key = re.sub(r'\s+', ' ', line.text).strip().casefold()
        if key not in names:
            continue
        if key in labels:
            return None
        labels[key] = line.bounds
    boxes = [labels.get(name) for name in names]
    if not all(boxes):
        return None
    cancel, title, next_button, profile = [VisualTarget(*box, 1.) for box in boxes]
    if not all(0 < t.x < t.x+t.width < frame.width and 0 < t.y < t.y+t.height < frame.height
               for t in (cancel, title, next_button, profile)):
        return None
    if (not cancel.x+cancel.width < title.x < title.x+title.width < next_button.x
            or max(abs(t.center[1]-title.center[1]) for t in (cancel, next_button)) > title.height*1.5
            or not title.y+title.height < profile.y
            or not cancel.x-title.width < profile.x < next_button.x):
        return None
    return cancel


class WheatPlayGames:
    def __init__(self, run):
        self.run = run
        self.uncertain = False

    def possible(self, frame):
        image = self.run.vision._image_for(frame.png)
        small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        neutral = cv2.inRange(hsv, (0, 0, 190), (179, 45, 255))
        # The invitation can outlive Hay Day and cover the dark launcher.
        # Inspect its own light panel, independent of the scene behind it.
        count, _, stats, _ = cv2.connectedComponentsWithStats(neutral)
        height, width = neutral.shape
        return any(w > width*.38 and h > height*.30 and area > width*height*.20
                   for _, _, w, h, area in stats[1:count])

    def observe(self, frame):
        if not self.possible(frame):
            return None
        text = self.run.vision.text.read(frame.png)
        return profile_cancel_target(frame, text.lines) if text.error is None else None

    def fresh(self):
        self.run.wait(.15)
        # This flag disables Android overlay handlers while they confirm their
        # own screenshots. Launcher waits otherwise use the normal raw pipeline.
        return self.run._capture_raw(handle_notifications=False)

    def process(self, frame):
        foreground = getattr(self.run.client, 'foreground_package', None)
        if not self.possible(frame) or not callable(foreground) or foreground() != 'com.google.android.gms':
            return frame
        target = self.observe(frame)
        if target is None:
            return frame
        if self.uncertain:
            raise WheatingPrerequisite('Play Games dismissal is unconfirmed; no repeated Cancel tap was sent.')
        fresh = self.fresh()
        checked = self.observe(fresh)
        if checked is None and foreground() == 'com.supercell.hayday':
            return fresh
        if (not fresh.captured_at or fresh.captured_at == frame.captured_at
                or not FarmingWorker._same_target(target, checked, fresh)
                or foreground() != 'com.google.android.gms'):
            raise WheatingPrerequisite('The Play Games invitation changed before its Cancel control was confirmed.')
        if self.run.diagnostics:
            (self.run.diagnostics/'play_games_before.png').write_bytes(fresh.png)
        self.uncertain = True
        self.run.publish('Wheating: dismissing the Play Games profile invitation with Cancel.')
        self.run.tap(checked.center, fresh)
        resolver = getattr(self.run.client, 'launcher_package', None)
        launcher = resolver() if callable(resolver) else None
        clear = 0
        for _ in range(10):
            after = self.fresh()
            distinct = bool(after.captured_at) and after.captured_at != fresh.captured_at
            package = foreground()
            returned = package == 'com.supercell.hayday' or bool(launcher and package == launcher)
            clear = clear+1 if distinct and returned else 0
            fresh = after
            if clear >= 2:
                self.uncertain = False
                self.run.publish('Wheating: Play Games dismissed. Waiting for the farm to settle.')
                return after
        raise WheatingPrerequisite('Play Games did not return to Hay Day or its launcher after Cancel; no repeated tap was sent.')

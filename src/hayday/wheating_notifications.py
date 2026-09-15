"""Decline only the verified Android notification request for Hay Day."""
import re

import cv2

from hayday.farming import FarmingWorker
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingPrerequisite

CONTROLLERS = {'com.android.permissioncontroller', 'com.google.android.permissioncontroller'}


def notification_target(frame, lines):
    def normalized(text):
        return re.sub(r'\s+', ' ', text.replace('\u2019', "'")).strip().casefold()
    labels = {normalized(line.text): line.bounds for line in lines}
    question = labels.get('allow hay day to send you notifications?')
    allow, deny = labels.get('allow'), labels.get("don't allow")
    if not question or not allow or not deny:
        return None
    q, a, d = [VisualTarget(*box, 1.) for box in (question, allow, deny)]
    if (not all(0 < t.x < t.x+t.width < frame.width and 0 < t.y < t.y+t.height < frame.height
                for t in (q, a, d))
            or not frame.height*.3 < q.center[1] < a.center[1] < d.center[1] < frame.height*.8
            or max(abs(t.center[0]-frame.width*.5) for t in (q, a, d)) > frame.width*.1):
        return None
    return d


class WheatNotifications:
    def __init__(self, run):
        self.run = run

    def possible(self, frame):
        image = self.run.vision._image_for(frame.png)
        small = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        buttons = cv2.inRange(hsv, (95, 20, 180), (120, 105, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(buttons)
        return sum(120 < w < 360 and 12 < h < 40 and area > w*h*.8
                   and 70 < x < 360 and 90 < y < 210 for x, y, w, h, area in stats[1:]) == 2

    def observe(self, frame):
        if not self.possible(frame):
            return None
        result = self.run.vision.text.read(frame.png)
        return notification_target(frame, result.lines) if result.error is None else None

    def fresh(self):
        self.run.wait(.1)
        return self.run._capture_raw(handle_notifications=False)

    def process(self, frame):
        foreground = getattr(self.run.client, 'foreground_package', None)
        if not self.possible(frame) or not callable(foreground) or foreground() not in CONTROLLERS:
            return frame
        target = self.observe(frame)
        if target is None:
            raise WheatingPrerequisite('An Android permission prompt needs attention. No permission was granted.')
        fresh = self.fresh()
        checked = self.observe(fresh)
        if checked is None and foreground() not in CONTROLLERS:
            return fresh
        if (not fresh.captured_at or frame.captured_at == fresh.captured_at
                or not FarmingWorker._same_target(target, checked, fresh)
                or foreground() not in CONTROLLERS):
            raise WheatingPrerequisite('The Android notification prompt changed before dismissal.')
        if self.run.diagnostics:
            (self.run.diagnostics/'notifications_before.png').write_bytes(fresh.png)
        self.run.tap(checked.center, fresh)
        clear = 0
        for _ in range(6):
            after = self.fresh()
            distinct = bool(after.captured_at) and after.captured_at != fresh.captured_at
            clear = clear+1 if distinct and not self.possible(after) and foreground() not in CONTROLLERS else 0
            fresh = after
            if clear >= 2:
                self.run.publish('Wheating: declined the Android notification request. Resuming wheat work.')
                return after
        raise WheatingPrerequisite('The Android notification dismissal is unconfirmed; no repeated tap was sent.')

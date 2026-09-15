"""Let a recognized achievement animation finish before resuming farm input."""
from pathlib import Path

import cv2

from hayday.resource_vision import _decode


class WheatAchievement:
    def __init__(self, run):
        self.run = run
        self.reference = _decode((Path(__file__).parent/'assets/wheating/achievement_title.png').read_bytes())

    def visible(self, frame):
        image = self.run.vision._image_for(frame.png)
        small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small[105:170, 65:255], cv2.COLOR_BGR2HSV)
        if (cv2.inRange(hsv, (17, 110, 180), (35, 255, 255)) > 0).mean() < .35:
            return False
        scale = frame.height/1080
        ref = cv2.resize(self.reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        crop = image[round(frame.height*.55):, round(frame.width*.2):round(frame.width*.8)]
        if crop.shape[0] < ref.shape[0] or crop.shape[1] < ref.shape[1]:
            return False
        return float(cv2.matchTemplate(crop, ref, cv2.TM_CCOEFF_NORMED).max()) >= .92

    def process(self, frame):
        if not self.visible(frame):
            return frame
        self.run.publish('Wheating: waiting for the achievement animation to clear.')
        clear = 0
        for _ in range(16):
            self.run.wait(.3)
            fresh = self.run._capture_raw()
            distinct = fresh.captured_at and fresh.captured_at != frame.captured_at
            clear = clear+1 if distinct and not self.visible(fresh) else 0
            frame = fresh
            if clear >= 2:
                return frame
        from hayday.wheating import WheatingPrerequisite
        raise WheatingPrerequisite('The achievement overlay is still open. Clear it to resume Wheating.')

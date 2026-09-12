"""Recognize and dismiss the fixed level-up controls without touching rewards."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hayday.dialogs import DialogVision
from hayday.resource_vision import ResourceVision, _decode
from hayday.tutorials import TutorialDismissal


class LevelUpVision(DialogVision):
    def __init__(self):
        root = Path(__file__).parent/'assets/wheating/level_up'
        spec = json.loads((root/'manifest.json').read_text('utf-8'))
        self.specifications = {'level_up': spec}
        self.references = {'level_up': [_decode((root/f['file']).read_bytes(), True)
                                       for f in spec['features']]}
        self._matcher = ResourceVision()
        self._png = self._result = None

    @staticmethod
    def possible(image):
        # Rejection only: the three independent original-pixel features below
        # must still match before Continue can become a target.
        small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (85, 110, 145), (110, 255, 255))
        green = cv2.inRange(hsv[:55], (30, 130, 155), (70, 255, 255))
        return float((blue > 0).mean()) > .15 and float((green > 0).mean()) > .06

    def observe(self, frame, *, image=None, cancel=lambda: False):
        self._check(cancel)
        if frame.png == self._png:
            return self._result
        image = _decode(frame.png) if image is None else image
        result = None
        if self.possible(image):
            base = frame.height/1080
            left, top = round(frame.width*.15), round(frame.height*.65)
            crop = image[top:, left:round(frame.width*.85)]
            reference = self.references['level_up'][0]
            anchors = self._matcher._search(crop, reference, np.array([base]), .90, cancel, 3)
            if not anchors:
                anchors = self._matcher._search(crop, reference,
                    np.geomspace(base*.6, base*1.3, 15), .90, cancel, 3)
            accepted = []
            for anchor in anchors:
                anchor = replace(anchor, x=anchor.x+left, y=anchor.y+top)
                scale = anchor.width/reference.shape[1]
                found = self._candidate(image, 'level_up', anchor, cancel, base)
                found = found or self._candidate(image, 'level_up', anchor, cancel, scale)
                found = found or self._refined(image, 'level_up', anchor, cancel)
                if found:
                    accepted.append(found)
            if len(accepted) == 1:
                result = accepted[0]
        self._check(cancel)
        self._png, self._result = frame.png, result
        return result


class WheatLevelUp:
    def __init__(self, run):
        self.run = run
        self.vision = LevelUpVision()
        self._uncertain = False

    def observe(self, frame):
        self.run.check()
        return self.vision.observe(frame, image=self.run.vision._image_for(frame.png),
                                   cancel=self.run.cancel_event.is_set)

    def _fresh(self):
        self.run.wait(.2)
        return self.run._capture_raw()

    def _save(self, label, frame):
        if self.run.diagnostics:
            (self.run.diagnostics/(label+'.png')).write_bytes(frame.png)

    def process(self, frame):
        self.run.check()
        if self._uncertain:
            self.run.block('The level-up Continue tap is unconfirmed; no repeated tap was sent.')
        previous = self.observe(frame)
        if previous is None:
            return frame
        self._save('level_up_detected', frame)
        self.run.publish('Wheating: confirming the level-up screen before continuing.')
        for _ in range(4):
            fresh = self._fresh()
            current = self.observe(fresh)
            distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
            if distinct and current is None:
                return fresh
            if (distinct and TutorialDismissal._same(previous, current)
                    and TutorialDismissal._valid_target(current, fresh)):
                frame = fresh
                break
            previous = current if distinct else None
            frame = fresh
        else:
            self.run.block('The level-up controls did not settle; no Continue tap was sent.')
        self._save('level_up_before_continue', frame)
        self._uncertain = True
        self.run.tap(current.target.center, frame)
        clear = 0
        for _ in range(6):
            fresh = self._fresh()
            distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
            clear = clear+1 if distinct and self.observe(fresh) is None else 0
            frame = fresh
            if clear >= 2:
                self._uncertain = False
                self._save('level_up_dismissed', frame)
                # Let the runner perform a post-dismissal stable farm capture
                # before it decides whether saved crop work is still valid.
                self.run._level_up_dismissed = True
                self.run.publish('Wheating: level-up screen cleared. Resuming the current wheat action.')
                return frame
        self._save('level_up_uncleared', frame)
        self.run.block('The level-up screen remained after Continue; no repeated tap was sent.')

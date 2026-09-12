"""Find every gold sold receipt from its fixed rooster and SOLD lettering."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, _decode


class SoldReceiptVision:
    def __init__(self, cancel=lambda: False):
        root = Path(__file__).parent/'assets/wheating/sold_receipt'
        self.spec = json.loads((root/'manifest.json').read_text('utf-8'))
        self.label = _decode((root/'label.png').read_bytes(), True)
        rooster = cv2.imdecode(np.frombuffer((root/'rooster.png').read_bytes(), np.uint8), 0)
        self.shape = cv2.resize(rooster, (32, 52), interpolation=cv2.INTER_AREA) > 127
        self.matcher, self.cancel = ResourceVision(), cancel

    def find(self, image):
        height, width = image.shape[:2]
        base = height/self.spec['reference_height']
        left, top = round(width*.10), round(height*.23)
        crop = image[top:round(height*.80), left:round(width*.90)]
        mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (15, 140, 130), (28, 240, 220))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        result = []
        rx1, ry1, rx2, ry2 = self.spec['rooster']
        lx1, ly1, lx2, ly2 = self.spec['label']
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if self.cancel():
                return ()
            if not (45*base <= h <= 160*base and .48 <= w/h <= .75 and .32 <= area/(w*h) <= .56):
                continue
            component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)*255
            shape = cv2.resize(component, (32, 52), interpolation=cv2.INTER_AREA) > 127
            overlap = np.logical_and(shape, self.shape).sum()/np.logical_or(shape, self.shape).sum()
            if overlap < .82:
                continue
            scale = h/(ry2-ry1)
            x0, y0 = x+left+(lx1-rx1)*scale, y+top+(ly1-ry1)*scale
            margin = max(5, round(7*scale))
            x0, y0 = max(0, round(x0)-margin), max(0, round(y0)-margin)
            right = min(width, x0+round((lx2-lx1)*scale)+2*margin)
            bottom = min(height, y0+round((ly2-ly1)*scale)+2*margin)
            # The profile badge shares the rooster. Only an independent SOLD
            # word at the correct relative position can authorize collection.
            hits = self.matcher._search(image[y0:bottom, x0:right], self.label,
                np.array([scale]), .91, self.cancel, 1)
            if not hits:
                hits = self.matcher._search(image[y0:bottom, x0:right], self.label,
                    np.array([base, w/(rx2-rx1), .96*scale, 1.04*scale]), .91, self.cancel, 1)
            if hits:
                result.append(replace(hits[0], x=hits[0].x+x0, y=hits[0].y+y0))
        return tuple(result)

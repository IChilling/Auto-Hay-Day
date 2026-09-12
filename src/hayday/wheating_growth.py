"""Measure the selected wheat countdown once per Wheating run."""
from __future__ import annotations

import base64
import json
import math
import re
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import _decode
from hayday.wheating_numbers import ranking, white_references


def timer_glyphs(image, bar):
    x, y, w, h = bar.box
    top, bottom = y+round(h*.58), y+round(h*1.24)
    left, right = x+round(w*.05), x+round(w*.95)
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        return []
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    # Boosted countdowns have purple interiors; ordinary ones have white.
    mask = cv2.inRange(hsv, (125, 30, 150), (175, 200, 255))
    mask |= cv2.inRange(hsv, (0, 0, 205), (179, 45, 255))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = []
    for label, (gx, gy, gw, gh, area) in enumerate(stats[1:], 1):
        if .055*h <= gh <= .60*h and gw <= .65*h and area >= .003*h*h:
            parts.append(([gx, gy, gx+gw, gy+gh], [label]))
    # Interior shading can separate the top and stem of I.
    # Merge only overlapping columns, never the gaps between separate letters.
    merged = True
    while merged:
        merged = False
        for i, (a, ids) in enumerate(parts):
            for j in range(i+1, len(parts)):
                b, others = parts[j]
                overlap = min(a[2], b[2])-max(a[0], b[0])
                gap = max(0, a[1]-b[3], b[1]-a[3])
                if (max(a[2]-a[0], b[2]-b[0]) < .15*h
                        and overlap > .5*min(a[2]-a[0], b[2]-b[0]) and gap < .05*h):
                    parts[i] = ([min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3])], ids+others)
                    parts.pop(j)
                    merged = True
                    break
            if merged:
                break
    result = []
    for (gx, gy, rx, by), ids in sorted(parts):
        gw, gh = rx-gx, by-gy
        if not (.27*h <= gh <= .60*h and gw <= gh*1.5):
            continue
        piece = np.isin(labels[gy:by, gx:rx], ids).astype(np.uint8)*255
        factor = min(28/gw, 36/gh)
        sw, sh = max(1, round(gw*factor)), max(1, round(gh*factor))
        glyph = np.zeros((40, 32), np.uint8)
        px, py = (32-sw)//2, (40-sh)//2
        glyph[py:py+sh, px:px+sw] = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_AREA)
        result.append(glyph)
    return result


@lru_cache(maxsize=1)
def timer_references():
    path = Path(__file__).parent/'assets/wheating/growth_units.json'
    units = json.loads(path.read_text('utf-8'))['glyphs']
    return tuple((label, glyph) for label, glyph in white_references() if label.isdecimal())+tuple(
        (item['label'], cv2.imdecode(np.frombuffer(base64.b64decode(item['png']), np.uint8), 0))
        for item in units)


def parse_countdown(text):
    match = re.fullmatch(r'(?:(\d{1,2})MIN)?(?:(\d{1,2})SEC)?', text)
    if not match or not any(match.groups()):
        return None
    minutes, seconds = (int(n or 0) for n in match.groups())
    total = minutes*60+seconds
    return total if seconds < 60 and 0 < total <= 3600 else None


def read_countdown(frame, vision):
    bar = vision.growing(frame.png)
    if bar is None:
        return None
    glyphs = timer_glyphs(_decode(frame.png), bar)
    if not 4 <= len(glyphs) <= 10:
        return None
    text = ''
    for glyph in glyphs:
        scores = ranking(glyph, templates=timer_references())
        # The complete MIN/SEC grammar separates letters from numbers. Comparing
        # N against 8 here would reject clear, small timers unnecessarily.
        label, confidence = scores[0]
        alternatives = [score for other, score in scores[1:]
                        if other.isdecimal() == label.isdecimal()]
        if confidence < .90 or confidence-max(alternatives) < .035:
            return None
        text += label
    seconds = parse_countdown(text)
    return (seconds, text, bar) if seconds is not None else None


class WheatGrowthTimer:
    FALLBACK_SECONDS = 120.

    def __init__(self, run):
        self.run = run
        self.seconds = None

    @property
    def duration(self):
        return self.seconds if self.seconds is not None else self.FALLBACK_SECONDS

    def measure(self, worker, frame, before, point, planted_at):
        if self.seconds is not None:
            return frame
        # Reuse the timer opened by releasing the seed drag. If it did not open,
        # select only the freshly planted, landmark-aligned final plot.
        if worker.vision.growing(frame.png) is None:
            target = worker._translated_plot(before, frame, point)
            if target is None:
                return frame
            frame = worker._tap(target, frame)
        previous = None
        for _ in range(5):
            worker._check()
            reading = read_countdown(frame, worker.vision)
            observed_at = time.monotonic()
            if reading:
                remaining, text, bar = reading
                duration = remaining+max(0., observed_at-planted_at)
                if (previous and frame.captured_at and frame.captured_at != previous[0]
                        and 0 <= previous[1]-remaining <= 3
                        and abs(duration-previous[2]) <= 1.5
                        and worker._same_target(previous[3], bar, frame)):
                    self.seconds = float(math.ceil(max(duration, previous[2])))
                    data = dict(seconds=self.seconds, remaining=remaining, text=text,
                                measured_at=datetime.now(UTC).isoformat())
                    self.run.state['wheat_growth'] = data
                    self.run.persist()
                    if self.run.diagnostics:
                        self.run.save_json(self.run.diagnostics/'wheat_growth.json', data)
                        (self.run.diagnostics/'wheat_growth.png').write_bytes(frame.png)
                    self.run.publish(f'Wheating: measured wheat growth at {self.seconds:g}s. '
                                     'Reusing this duration for later planting cycles.')
                    worker.growth_ready_at = planted_at+self.seconds
                    return frame
                previous = (frame.captured_at, remaining, duration, bar)
            else:
                previous = None
            worker._pause(.2)
            frame = worker._quick_frame()
        self.run.publish('Wheating: crop timer was not clear enough to cache; retrying after the next seeding.')
        return frame

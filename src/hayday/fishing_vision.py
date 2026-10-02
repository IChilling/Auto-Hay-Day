"""Observed fishing controls and feedback; no device input is sent here."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.quantities import read_count
from hayday.resource_vision import ResourceVision, VisualTarget, _decode


@dataclass(frozen=True)
class LureMenu:
    tool: tuple[int, int]
    stock: int | None
    scale: float


@dataclass(frozen=True)
class LureWorkbench:
    tool: tuple[int, int]
    stock: int | None
    scale: float
    empty_slots: tuple[VisualTarget, ...]
    title: VisualTarget


class FishingVision:
    def __init__(self):
        root = Path(__file__).resolve().parents[2]/'images/fishing'
        if not root.is_dir():
            root = Path(__file__).parent/'assets/fishing'
        self.manifest = json.loads((root/'manifest.json').read_text('utf-8'))
        self.references = {name: _decode((root/f'{name}.png').read_bytes(), True)
                           for name in ('red_lure', 'blue_lure', 'home',
                                        'workbench_title', 'free_red_recipe', 'queued_red_lure',
                                        'workbench_anchor')}
        self.item = (root/'fish_item.png').read_bytes()
        self.matcher = ResourceVision()
        self.red_title = cv2.imencode('.png', self.matcher._title_crop(
            _decode((root/'red_lure_title.png').read_bytes())))[1].tobytes()

    def supports(self, icon, cancel):
        matches = self.matcher.find_item(icon, self.item, min_scale=.5, max_scale=1.8,
                                         threshold=.94, cancel=cancel)
        return bool(matches and matches[0].score >= .94)

    def _matches(self, png, name, cancel):
        frame = _decode(png)
        base = frame.shape[0]/1080
        return self.matcher._search(frame, self.references[name],
            np.unique(np.r_[np.linspace(base*.8, base*1.2, 13), base]), .94, cancel, max_peaks=3)

    def home(self, png, cancel=lambda: False):
        frame = _decode(png)
        base = frame.shape[0]/1080
        top = round(frame.shape[0]*.8)
        matches = self.matcher._search(frame[top:, :round(frame.shape[1]*.15)],
            self.references['home'], np.unique(np.r_[np.linspace(base*.8, base*1.2, 13), base]),
            .94, cancel, max_peaks=3)
        matches = [VisualTarget(m.x, m.y+top, m.width, m.height, m.score) for m in matches]
        matches = [m for m in matches if m.center[0] < frame.shape[1]*.15
                   and m.center[1] > frame.shape[0]*.8 and m.score >= .94]
        return matches[0] if len(matches) == 1 else None

    def menu(self, png, cancel=lambda: False):
        reds = self._matches(png, 'red_lure', cancel)
        if not reds:
            return None
        blues = self._matches(png, 'blue_lure', cancel)
        from hayday.farming_vision import FarmingVision
        # The workbench displays identical lure artwork. Only the spot picker
        # has paging buttons; producer queues must never authorize a cast.
        if not FarmingVision(cancel=cancel).seed_menu(png):
            return None
        source = self.manifest['features']['red_lure']
        blue_source = self.manifest['features']['blue_lure']
        candidates = []
        for red in reds:
            if red.score < .94:
                continue
            scale = red.width/(source[2]-source[0])
            expected = (red.x+(blue_source[0]-source[0])*scale,
                        red.y+(blue_source[1]-source[1])*scale)
            if not any(blue.score >= .94 and np.hypot(blue.x-expected[0], blue.y-expected[1]) < 12*scale
                       for blue in blues):
                continue
            x, y, w, h = self.manifest['stock_box']
            stock_box = (round(red.x+(x-source[0])*scale), round(red.y+(y-source[1])*scale),
                         round(w*scale), round(h*scale))
            tx, ty = self.manifest['tool_point']
            point = round(red.x+(tx-source[0])*scale), round(red.y+(ty-source[1])*scale)
            candidates.append(LureMenu(point, read_count(png, stock_box), scale))
        return candidates[0] if len(candidates) == 1 else None

    def workbench(self, png, cancel=lambda: False):
        titles = self._matches(png, 'workbench_title', cancel)
        if len(titles) != 1:
            return None
        title = titles[0]
        scale = title.width/self.references['workbench_title'].shape[1]
        reds = [red for red in self._matches(png, 'red_lure', cancel)
                if -430*scale < red.center[0]-title.x < -230*scale
                and 20*scale < red.center[1]-title.y < 155*scale]
        if len(reds) != 1:
            return None
        red = reds[0]
        red_scale = red.width/self.references['red_lure'].shape[1]
        source = self.manifest['features']['red_lure']
        x, y, w, h = self.manifest['stock_box']
        stock_box = (round(red.x+(x-source[0])*red_scale), round(red.y+(y-source[1])*red_scale),
                     round(w*red_scale), round(h*red_scale))
        scene = self.matcher.observe(png, cancel)
        slots = tuple(sorted((slot for slot in scene.empty_slots
            if -260*scale < slot.center[0]-title.x < 180*scale
            and 260*scale < slot.center[1]-title.y < 410*scale), key=lambda slot: slot.x))
        return LureWorkbench(red.center, read_count(png, stock_box), red_scale, slots, title)

    def free_red_recipe(self, png, bench, cancel=lambda: False):
        matches = self._matches(png, 'free_red_recipe', cancel)
        return len(matches) == 1 and (
            -155*bench.scale < matches[0].x-bench.title.x < -95*bench.scale
            and 90*bench.scale < matches[0].y-bench.title.y < 155*bench.scale)

    def lure_location(self, png, cancel=lambda: False):
        scene = self.matcher.observe(png, cancel)
        popups = [popup for popup in scene.popups if not popup.rows and len(popup.navigation) == 1
                  and self.matcher.titles_match(self.red_title, popup.title_png)]
        return popups[0] if len(popups) == 1 else None

    def queued_red_lure(self, png, bench, cancel=lambda: False):
        # Both masked views describe the same lure. Backgrounds behind the
        # queue change when the workbench is reopened at another world zoom.
        artwork = (*self._matches(png,'red_lure',cancel),
                   *self._matches(png,'queued_red_lure',cancel))
        matches = [match for match in artwork
                   if -170*bench.scale < match.center[0]-bench.title.x < -105*bench.scale
                   and 340*bench.scale < match.center[1]-bench.title.y < 400*bench.scale]
        # Two references may find the same object; distinct queue targets
        # remain ambiguous and cannot confirm a production attempt.
        return bool(matches) and all(
            np.linalg.norm(np.subtract(match.center,matches[0].center)) < 20*bench.scale
            for match in matches) and not any(
            np.linalg.norm(np.subtract(slot.center, matches[0].center)) < 80*bench.scale
            for slot in bench.empty_slots)

    @staticmethod
    def first_slot_empty(bench):
        return any(-190*bench.scale < slot.center[0]-bench.title.x < -90*bench.scale
                   for slot in bench.empty_slots)

    def workbench_anchor(self, png, cancel=lambda: False):
        frame = _decode(png)
        base = frame.shape[0]/1080
        # The machine follows world zoom independently of the fixed-size menu.
        sizes = np.unique(np.r_[np.geomspace(base*.45, base*2.2, 30), base])
        # Small world signs lose a little agreement across pixel phases. This
        # only proposes a reopen target: collection requires two fresh sign
        # observations, followed by the independently recognized producer menu.
        matches = self.matcher._search(frame,self.references['workbench_anchor'],
                                       sizes,.93,cancel,max_peaks=3)
        return matches[0] if len(matches) == 1 else None

    def cooldown(self, png, cancel=lambda: False):
        from hayday.farming_vision import FarmingVision
        return self.home(png, cancel) is not None and FarmingVision(cancel=cancel).growing(png) is not None

    @staticmethod
    def line_end(png, point):
        """Find a thin white fishing line connected to the current contact.

        Hough segments bridge ripples and reeds that interrupt a pixel component.
        Collinearity and contact proximity exclude rings and unrelated highlights.
        """
        frame = _decode(png)
        scale = frame.shape[0]/1080
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 195]), np.array([179, 25, 255]))
        h, w = mask.shape
        mask[:round(h*.18)] = 0
        mask[round(h*.93):] = 0
        mask[:, :round(w*.18)] = 0
        mask[:, round(w*.83):] = 0
        lines = cv2.HoughLinesP(mask, 1, np.pi/360, threshold=max(15, round(25*scale)),
            minLineLength=round(45*scale), maxLineGap=round(35*scale))
        if lines is None:
            return None
        point = np.array(point, dtype=float)
        candidates = []
        for line in lines[:, 0]:
            a, b = line[:2].astype(float), line[2:].astype(float)
            if np.linalg.norm(a-point) > np.linalg.norm(b-point):
                a, b = b, a
            distance = np.linalg.norm(b-point)
            if np.linalg.norm(a-point) <= 45*scale and 45*scale < distance < 650*scale:
                candidates.append((distance, b))
        if not candidates:
            return None
        _, end = max(candidates, key=lambda entry: entry[0])
        # A bent line is represented by several Hough segments. Stopping at
        # the first segment's end can hide a still-visible bobber and falsely
        # signal a bite. Follow nearby forward segments with matching direction;
        # crossing ripples and remote white controls cannot extend the chain.
        for _ in range(6):
            direction = end-point
            direction /= max(1., np.linalg.norm(direction))
            extensions = []
            for line in lines[:, 0]:
                a, b = line[:2].astype(float), line[2:].astype(float)
                if np.linalg.norm(a-point) > np.linalg.norm(b-point):
                    a, b = b, a
                segment = b-a
                length = np.linalg.norm(segment)
                distance = np.linalg.norm(b-point)
                fraction = np.clip(np.dot(end-a, segment)/max(1., length**2), 0., 1.)
                nearest = a+fraction*segment
                if (np.linalg.norm(nearest-end) <= 35*scale and length > 0
                        and np.dot(segment/length, direction) >= .85
                        and np.linalg.norm(end-point)+10*scale < distance < 650*scale):
                    extensions.append((distance, b))
            if not extensions:
                break
            _, end = max(extensions, key=lambda entry: entry[0])
        return tuple(map(int, end))

    @staticmethod
    def bobber(png, end):
        frame = _decode(png)
        scale = frame.shape[0]/1080
        x, y = end
        radius = round(40*scale)
        crop = frame[max(0,y-radius):y+radius, max(0,x-radius):x+radius]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0,95,80]), np.array([18,255,255]))
        return np.count_nonzero(red) > 50*scale*scale

    @staticmethod
    def ring_center(png, near):
        frame = _decode(png)
        scale = frame.shape[0]/1080
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([80,0,180]), np.array([105,65,255]))
                | cv2.inRange(hsv, np.array([0,65,100]), np.array([10,220,250]))
                | cv2.inRange(hsv, np.array([35,55,100]), np.array([80,220,250])))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if len(contour) < 15:
                continue
            (x,y), (a,b), angle = cv2.fitEllipse(contour)
            if (180*scale < b < 1200*scale and .46 < a/b < .56 and abs(angle-90) < 4
                    and np.hypot(x-near[0], y-near[1]) < 80*scale):
                candidates.append((abs(angle-90)+abs(a/b-.5)*50, (round(x),round(y))))
        return min(candidates)[1] if candidates else None

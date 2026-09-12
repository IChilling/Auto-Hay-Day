"""Harvest and replant through freshly observed selected-plot controls.

Unfinished gestures remain durable intent. Saved screenshots can relocate the
same selected soil; crop identity alone cannot resolve a previous gesture.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.resource_vision import ResourceVision, VisualTarget
from hayday.resources import ResourceChanged, ResourceResult, _save_json


class FarmingWorker:
    _PENDING = frozenset({'harvest_attempted', 'harvested_needs_replant', 'plant_attempted'})
    _STAGES = _PENDING | {'growing'}
    _FRESH_SECONDS = 5.0

    def __init__(self, client, capture, cancel_event, progress, state_path, *, vision=None,
                 resource_vision=None):
        from hayday.farming_vision import FarmingVision
        self.client = client
        self.capture = capture
        self.cancel_event = cancel_event
        self.progress = progress
        self.state_path = Path(state_path)
        self.vision = vision or FarmingVision(cancel=cancel_event.is_set)
        self.items = resource_vision or ResourceVision()
        self.serial = client.serial
        self._size = None
        self.state = {'version': 1, 'serial': self.serial, 'items': {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if (not isinstance(self.state, dict) or self.state.get('version') != 1
                    or not isinstance(self.state.get('items'), dict)):
                raise ValueError('Saved farming state is invalid; no farming input can be sent.')
            if 'serial' not in self.state and not self.state['items']:
                self.state['serial'] = self.serial
            if (self.state.get('serial') != self.serial
                    or any(not isinstance(key, str) or not isinstance(entry, dict)
                           or entry.get('stage') not in self._STAGES
                           for key, entry in self.state['items'].items())):
                raise ValueError('Saved farming intent is invalid or belongs to another device.')

    def _check(self):
        if self.cancel_event.is_set() or not self.serial or self.client.serial != self.serial:
            raise ResourceChanged('Farming was cancelled or its device selection changed.')

    def _frame(self):
        self._check()
        frame = self.capture()
        self._check()
        if self._size != (frame.width, frame.height):
            raise ResourceChanged('Device resolution changed during farming.')
        return frame

    def _pause(self, seconds=.35):
        self._check()
        self.cancel_event.wait(seconds)
        self._check()

    def _see(self, method, frame):
        self._check()
        try:
            result = getattr(self.vision, method)(frame.png)
        except RuntimeError:
            self._check()
            raise
        self._check()
        return result

    def _tap(self, target, frame):
        self._check()
        self.client.tap(*map(int, target), width=frame.width, height=frame.height)
        self._pause()
        return self._frame()

    def _record(self, key, stage, **details):
        entry = self.state['items'].setdefault(key, {})
        entry.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **details)
        _save_json(self.state_path, self.state)

    def _close(self, frame):
        self._check()
        ground = CameraNavigator._grass_start(frame, 0, 0)
        if ground:
            self._tap(ground[:2], frame)

    @staticmethod
    def _same_target(first, second, frame):
        if first is None or second is None:
            return False
        return (np.linalg.norm(np.subtract(first.center, second.center)) <= max(4, 8*frame.height/1080)
                and abs(first.width-second.width) <= max(3, first.width*.08)
                and abs(first.height-second.height) <= max(3, first.height*.08))

    @staticmethod
    def _translated_plot(before, after, point, *, require_visible=True):
        """Track a just-harvested plot only across a verified camera translation."""
        first = cv2.imdecode(np.frombuffer(before.png, np.uint8), 0)
        second = cv2.imdecode(np.frombuffer(after.png, np.uint8), 0)
        if first is None or second is None or first.shape != second.shape:
            return None
        mask = np.zeros_like(first)
        h, w = first.shape
        mask[int(h*.16):int(h*.83), int(w*.16):int(w*.84)] = 255
        orb = cv2.ORB_create(nfeatures=1200)
        ka, a = orb.detectAndCompute(first, mask)
        kb, b = orb.detectAndCompute(second, mask)
        if a is None or b is None:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
        if len(good) < 16:
            return None
        old = np.float32([ka[m.queryIdx].pt for m in good])
        new = np.float32([kb[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3)
        if (matrix is None or inliers is None or not np.isfinite(matrix).all()
                or inliers.sum() < max(15, len(good)*.5)):
            return None
        tracked = old[inliers.ravel().astype(bool)]
        # A local animation or a fixed UI patch cannot establish a farm-wide transform.
        if np.ptp(tracked[:, 0]) < w*.22 or np.ptp(tracked[:, 1]) < h*.18:
            return None
        if abs(np.hypot(matrix[0, 0], matrix[1, 0])-1) > .025 or abs(matrix[1, 0]) > .02:
            return None
        projected = matrix @ np.array([*point, 1])
        if require_visible and not (w*.13 < projected[0] < w*.88 and h*.15 < projected[1] < h*.88):
            return None
        return tuple(int(round(n)) for n in projected)

    @staticmethod
    def _stock_box(frame, item, *, separate=False):
        image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), 1)
        if image is None:
            return None
        h, w = image.shape[:2]
        x0, y0 = max(0, int(item.x-item.width*1.5)), max(0, int(item.y-item.height*.7))
        x1, y1 = min(w, int(item.x+item.width*.35)), min(h, int(item.y+item.height*.55))
        if x1 <= x0 or y1 <= y0:
            return None
        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        cream = cv2.inRange(hsv, (0, 0, 205), (55, 70, 255))
        if separate:
            # A cream barn behind the translucent picker can touch the stock
            # bubble. Break narrow connections before retrying digit reading.
            size = max(3, round(item.width*.09))
            cream = cv2.morphologyEx(cream, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        count, _, stats, _ = cv2.connectedComponentsWithStats(cream)
        candidates = []
        for x, y, bw, bh, area in stats[1:count]:
            if bw < item.width*.45 or bh < item.height*.2 or area < bw*bh*.55:
                continue
            cx, cy = x+x0+bw/2, y+y0+bh/2
            if cx < item.center[0] and cy < item.center[1]:
                candidates.append((np.hypot(cx-item.x, cy-item.y), (int(x+x0), int(y+y0), int(bw), int(bh))))
        return min(candidates, default=(0, None), key=lambda p: p[0])[1]

    def _seed(self, frame, icon, plot, *, plot_center=None):
        self._check()
        center = plot_center or (plot.center if plot is not None else None)
        if plot is None or center is None or center[0] <= 0:
            return None
        candidates = self.items.find_item(frame.png, icon, min_scale=.75, max_scale=3.2,
            region=(0, 0, min(frame.width, int(center[0])), int(frame.height*.84)),
            cancel=self.cancel_event.is_set)
        self._check()
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if any(not self._same_target(best, other, frame) and other.score >= best.score-.07
               for other in ranked[1:]):
            raise ResourceChanged('The requested seed has multiple plausible positions in the picker.')
        return best

    def _count(self, frame, seed):
        from hayday.quantities import read_count
        self._check()
        box = self._stock_box(frame, seed)
        value = read_count(frame.png, box) if box else None
        if value is None:
            box = self._stock_box(frame, seed, separate=True)
            value = read_count(frame.png, box) if box else None
        self._check()
        return value

    def _out_of_stock_seed(self, frame, icon, plot):
        """Recognize a zero-stock icon obscured by +, for diagnosis only.

        A partial icon must NEVER authorize planting. Its only use is to stop
        paging when the requested seed's readable stock is zero.
        """
        self._check()
        item = cv2.imdecode(np.frombuffer(icon, np.uint8), cv2.IMREAD_UNCHANGED)
        if item is None or item.ndim != 3 or item.shape[2] not in (3, 4):
            return False
        if item.shape[2] == 3:
            _, labels = cv2.connectedComponents(self.items._cream(item).astype(np.uint8))
            edges = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
            alpha = (~np.isin(labels, edges[edges != 0])).astype(np.uint8)*255
            item = np.dstack((item, cv2.erode(alpha, np.ones((3, 3), np.uint8))))
        ys, xs = np.nonzero(item[:, :, 3] >= 200)
        if len(xs) < 60:
            return False
        item = item[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
        item[:round(item.shape[0]*.5), :, 3] = 0
        image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), 1)
        if image is None:
            return False
        candidates = self.items._search(image[:int(frame.height*.84), :plot.center[0]],
            item, np.geomspace(.75, 3.2, 19), .93, self.cancel_event.is_set, 4)
        self._check()
        if not candidates or any(not self._same_target(candidates[0], other, frame)
                                 for other in candidates[1:]):
            return False
        return self._count(frame, candidates[0]) == 0

    def _fresh(self, started):
        self._check()
        if time.monotonic()-started > self._FRESH_SECONDS:
            raise ResourceChanged('The farming controls became stale during recognition; no gesture was sent.')

    def _await_seed_picker(self, frame, expected):
        """Wait for two stable picker frames after the one authorized plot tap."""
        previous = None
        for attempt in range(5):
            if self._see('seed_menu', frame):
                plot = self._see('empty_plot', frame)
                if (plot is not None
                        and np.linalg.norm(np.subtract(plot.center, expected))
                        <= max(24, plot.width*.8)):
                    if self._same_target(previous, plot, frame):
                        return frame, plot
                    previous = plot
                else:
                    previous = None
            else:
                previous = None
            if attempt < 4:
                self._pause(.2)
                frame = self._frame()
        return frame, None

    def _evidence(self, frame, operation, label):
        directory = self.state_path.parent / 'farming_evidence'
        directory.mkdir(parents=True, exist_ok=True)
        name = f'{operation}_{label}.png'
        (directory / name).write_bytes(frame.png)
        return {'file': name, 'sha256': hashlib.sha256(frame.png).hexdigest(),
                'width': frame.width, 'height': frame.height, 'captured_at': frame.captured_at}

    @staticmethod
    def _exposed_soil(before, after, harvest):
        """Find newly exposed brown soil at this crop after verified translation."""
        outline = harvest.highlight
        if outline is None:
            return None
        projected = FarmingWorker._translated_plot(before, after, outline.center)
        if projected is None:
            return None
        dx, dy = np.subtract(projected, outline.center)
        first = cv2.imdecode(np.frombuffer(before.png, np.uint8), 1)
        second = cv2.imdecode(np.frombuffer(after.png, np.uint8), 1)
        height, width = second.shape[:2]
        first = cv2.warpAffine(first, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
        margin = round(72*harvest.scale)
        x, y, w, h = outline.box
        left, top = max(0, x+dx-margin), max(0, y+dy-margin)
        right, bottom = min(width, x+dx+w+margin), min(height, y+dy+h+margin)
        # Hue excludes red tomatoes and yellow kernels that a broad RGB brown
        # test can misclassify as soil. Compare the same observed farm region.
        masks = [cv2.inRange(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV),
                            (8, 85, 55), (24, 235, 230)) for image in (first, second)]
        changed = ((masks[1] > 0) & (masks[0] == 0)).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN,
                                  np.ones((max(1, round(2*harvest.scale)),)*2, np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE,
                                  np.ones((max(3, round(5*harvest.scale)),)*2, np.uint8))
        _, labels, stats, centers = cv2.connectedComponentsWithStats(changed)
        choices = []
        for label, (rx, ry, rw, rh, area) in enumerate(stats[1:], 1):
            if (area < max(40, 120*harvest.scale**2) or min(rw, rh) < 10*harvest.scale
                    or rx == 0 or ry == 0 or rx+rw == changed.shape[1] or ry+rh == changed.shape[0]):
                continue
            ys, xs = np.nonzero((labels == label) & (masks[1] > 0))
            if not len(xs):
                continue
            cx, cy = centers[label]
            nearest = int(np.argmin((xs-cx)**2+(ys-cy)**2))
            # The input point must be an actual soil pixel, not just the center
            # of a bounding box that could contain neighboring foliage.
            point = int(left+xs[nearest]), int(top+ys[nearest])
            choices.append((int(area), point))
        choices.sort(reverse=True)
        if not choices or len(choices) > 1 and choices[1][0] >= choices[0][0]*.65:
            return None
        point = choices[0][1]
        radius = max(4, round(8*harvest.scale))
        return VisualTarget(point[0]-radius, point[1]-radius, radius*2, radius*2, 1.)

    def _await_harvested_soil(self, before, harvest, key):
        prior = None
        evidence_frames = []
        for attempt in range(6):
            self._pause(.75 if attempt == 0 else .4)
            frame = self._frame()
            evidence_frames = [*evidence_frames[-1:], frame]
            soil = None if self._see('harvest', frame) else self._exposed_soil(before, frame, harvest)
            self._check()
            if self._same_target(prior, soil, frame):
                break
            prior = soil
        else:
            soil = None
        operation = self.state['items'][key]['operation']
        evidence = [self._evidence(frame, operation, f'harvest_after_{i}')
                    for i, frame in enumerate(evidence_frames)]
        self._record(key, 'harvested_needs_replant' if soil else 'harvest_attempted',
                     harvest_after=evidence)
        return frame, soil

    def _resume_picker(self, frame, entry):
        """Resume only the same selected soil, freshly relocated from its evidence."""
        proof = entry.get('replant_evidence')
        if entry.get('stage') != 'harvested_needs_replant' or not isinstance(proof, dict):
            return None
        try:
            name = proof['file']
            if not isinstance(name, str) or Path(name).name != name:
                return None
            png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
            if hashlib.sha256(png).hexdigest() != proof['sha256']:
                return None
            saved = Screenshot(png, proof['width'], proof['height'], proof['captured_at'])
            box = proof['plot']
            if len(box) != 4 or any(type(value) is not int for value in box):
                return None
            center = box[0]+box[2]//2, box[1]+box[3]//2
            expected = self._translated_plot(saved, frame, center)
        except (KeyError, TypeError, ValueError, OSError, cv2.error):
            return None
        if expected is None or not self._see('seed_menu', frame):
            return None
        plot = self._see('empty_plot', frame)
        if plot and np.linalg.norm(np.subtract(plot.center, expected)) <= max(5, 8*frame.height/1080):
            return plot
        return None

    def work_if_recognized(self, frame, icon, key):
        self._size = frame.width, frame.height
        self._check()
        pending = self.state['items'].get(key, {}).get('stage')
        if pending in self._PENDING:
            resumed = self._resume_picker(frame, self.state['items'][key])
            if resumed:
                return self._plant_selected(frame, icon, key, resumed, True, resumed.center)
            # Another plot of the same crop cannot settle this plot's intent.
            return ResourceResult('unsupported',
                'Earlier crop work needs inspection before another harvest or planting attempt.',
                {'pending_stage': pending, 'state_file': str(self.state_path)})
        growing = self._see('growing', frame)
        if growing:
            fresh = self._frame()
            if not self._same_target(growing, self._see('growing', fresh), fresh):
                return ResourceResult('changed', 'The selected crop’s growth timer changed before confirmation.')
            self._record(key, 'growing')
            self._close(fresh)
            return ResourceResult('waiting', 'The selected plant is already in place and still growing; checking other requirements.',
                                  {'wait_seconds': 30, 'item': key, 'defer_item': True, 'retry_after_seconds': 30})
        harvest = self._see('harvest', frame)
        harvested = False
        plot = None
        tracked_plot = None
        if harvest:
            fresh = self._frame()
            started = time.monotonic()
            checked = self._see('harvest', fresh)
            if (not checked or not self._same_target(harvest.tool, checked.tool, fresh)
                    or not self._same_target(harvest.target, checked.target, fresh)
                    or not self._same_target(harvest.drag_target or harvest.target,
                                             checked.drag_target or checked.target, fresh)):
                return ResourceResult('changed', 'The harvest tool or selected plot moved before input.')
            self._fresh(started)
            self.progress('Harvesting the plot selected by the game’s resource link.')
            operation = uuid.uuid4().hex
            proof = self._evidence(fresh, operation, 'harvest_before')
            self._record(key, 'harvest_attempted', operation=operation, harvest_before=proof,
                         harvest_after=[], replant_evidence=None, replanted=False)
            self._fresh(started)
            self.client.swipe(*checked.tool.center, *(checked.drag_target or checked.target).center, width=fresh.width,
                              height=fresh.height, duration_ms=650)
            after, soil = self._await_harvested_soil(fresh, checked, key)
            if soil is None:
                return ResourceResult('unsupported', 'Harvest input was sent once, but newly exposed soil was not confirmed in two frames. Its intent remains pending.')
            position = soil.center
            frame = self._tap(position, after)
            tracked_plot = position
            frame, plot = self._await_seed_picker(frame, tracked_plot)
            harvested = True
        elif not self._see('seed_menu', frame):
            return None
        if harvested and plot is not None:
            proof = self._evidence(frame, self.state['items'][key]['operation'], 'replant_picker')
            self._record(key, 'harvested_needs_replant', replant_evidence={**proof, 'plot': list(plot.box)})
        return self._plant_selected(frame, icon, key, plot, harvested, tracked_plot)

    def _plant_selected(self, frame, icon, key, plot=None, harvested=False, tracked_plot=None):
        if harvested and plot is None:
            return ResourceResult('unsupported', 'Harvest was attempted, but its seed picker was not confirmed. Replanting remains pending.')
        if plot is None and not self._see('seed_menu', frame):
            return ResourceResult('unsupported', 'Harvest was attempted, but its seed picker was not confirmed. Replanting remains pending.')
        plot = plot or self._see('empty_plot', frame)
        # Search at most five observed pages, preserving this selected tile.
        for page in range(5):
            if plot is None or not self._see('seed_menu', frame):
                return ResourceResult('changed', 'The selected empty plot or seed controls are no longer visible.')
            seed = self._seed(frame, icon, plot, plot_center=tracked_plot)
            if seed:
                break
            if self._out_of_stock_seed(frame, icon, plot):
                return ResourceResult('unsupported',
                    'The requested seed has zero stock. Replanting needs seed stock; no purchase was made.',
                    {'needs_seed': True, 'harvested': harvested})
            next_page = self._see('page_next', frame)
            if next_page is None or page == 4:
                return ResourceResult('unsupported', 'The requested seed was not found within the observed seed-picker pages.')
            fresh = self._frame()
            started = time.monotonic()
            fresh_plot = self._see('empty_plot', fresh)
            checked_page = self._see('page_next', fresh)
            if (not self._same_target(plot, fresh_plot, fresh)
                    or not self._same_target(next_page, checked_page, fresh)
                    or not self._see('seed_menu', fresh)):
                return ResourceResult('changed', 'The selected plot or seed page controls moved before input.')
            self._fresh(started)
            frame = self._tap(checked_page.center, fresh)
            observed_plot = self._see('empty_plot', frame)
            if not self._same_target(plot, observed_plot, frame):
                return ResourceResult('changed', 'The selected empty plot changed while paging through seeds.')
            plot = observed_plot
        available = self._count(frame, seed)
        if available is None or available < 1:
            self._close(frame)
            return ResourceResult('unsupported', 'Seed stock could not be confirmed positive; the plot was not planted.')
        fresh = self._frame()
        started = time.monotonic()
        fresh_plot = self._see('empty_plot', fresh)
        if (not self._same_target(plot, fresh_plot, fresh)
                or not self._see('seed_menu', fresh)):
            return ResourceResult('changed', 'The seed picker or selected plot changed before planting.')
        fresh_seed = self._seed(fresh, icon, fresh_plot, plot_center=tracked_plot)
        if not self._same_target(seed, fresh_seed, fresh):
            return ResourceResult('changed', 'The requested seed moved before planting.')
        if self._count(fresh, fresh_seed) != available:
            return ResourceResult('changed', 'Seed inventory changed before planting.')
        self._fresh(started)
        self._record(key, 'plant_attempted', available_before=available, operation=uuid.uuid4().hex)
        self._fresh(started)
        destination = tracked_plot or fresh_plot.center
        self.client.swipe(*fresh_seed.center, *destination, width=fresh.width, height=fresh.height, duration_ms=550)
        prior_timer = None
        for _ in range(6):
            self._pause(.4)
            frame = self._frame()
            timer = self._see('growing', frame)
            if self._same_target(prior_timer, timer, frame):
                self._record(key, 'growing', replanted=True)
                self._close(frame)
                message = ('The selected crop was replanted after a harvest attempt.' if harvested
                           else 'The selected empty plot was planted.')
                return ResourceResult('waiting', message+' Its growth timer was confirmed; returning to verify order stock.', {'wait_seconds': 1})
            prior_timer = timer
        return ResourceResult('unsupported', 'Planting was attempted, but two matching growth frames were not confirmed; its intent remains pending.')

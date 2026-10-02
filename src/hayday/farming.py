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
    _STAGES = _PENDING | {'growing', 'plant_no_effect', 'harvest_no_effect', 'stock_check'}
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
            for entry in self.state['items'].values():
                plan = entry.get('planting_plan')
                if ('growing_plots' in entry and (type(entry['growing_plots']) is not int
                                                 or entry['growing_plots'] < 0)):
                    raise ValueError('Saved crop quantity is invalid.')
                demand = entry.get('demand')
                if demand is not None and (not isinstance(demand, dict)
                        or type(demand.get('required')) is not int or demand['required'] < 1):
                    raise ValueError('Saved crop requirement is invalid.')
                if plan is not None and (not isinstance(plan, dict)
                        or any(type(plan.get(name)) is not int or plan[name] < 0
                               for name in ('required', 'stock_before', 'planned', 'confirmed', 'shortfall'))
                        or plan['required'] < 1 or not 0 <= plan['confirmed'] <= plan['planned'] <= plan['stock_before']
                        or type(plan.get('seed_limited')) is not bool
                        or not isinstance(plan.get('operations'), list)
                        or len(plan['operations']) != plan['confirmed']
                        or any(not isinstance(op, str) or not op for op in plan['operations'])
                        or len(set(plan['operations'])) != len(plan['operations'])):
                    raise ValueError('Saved crop planting plan is invalid.')

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
        palette = self._see('seed_menu_region', frame)
        ground = CameraNavigator._grass_start(frame, 0, 0,
                                              exclude_regions=[palette.box] if palette else [])
        if ground:
            return self._tap(ground[:2], frame)
        return frame

    def _requirement(self, key, baseline):
        if not isinstance(baseline, dict) or type(baseline.get('required')) is not int or baseline['required'] < 1:
            return
        entry = self.state['items'].get(key, {})
        # Older single-plot records contain one confirmed growing crop.
        growing = entry.get('growing_plots', int(entry.get('stage') == 'growing'))
        self._record(key, entry.get('stage', 'stock_check'), growing_plots=growing,
                     demand={name: baseline.get(name) for name in ('available', 'required', 'captured_at')})

    @staticmethod
    def _plant_budget(available, required, growing):
        # One seed becomes two crops. Retain one seed beyond the recipe's stock
        # so completing production cannot remove the ability to grow more.
        shortfall = max(0, required+1-available-2*growing)
        return min(available, shortfall), shortfall

    def _plan(self, key, available):
        entry = self.state['items'][key]
        demand = entry.get('demand')
        if not demand:
            return None
        plan = entry.get('planting_plan')
        if (plan and plan['confirmed'] < plan['planned']
                and plan['required'] == demand['required']
                and available == plan['stock_before']-plan['confirmed']):
            return plan
        count, shortfall = self._plant_budget(available, demand['required'], entry.get('growing_plots', 0))
        plan = {'required': demand['required'], 'stock_before': available,
                'planned': count, 'confirmed': 0, 'shortfall': shortfall,
                'seed_limited': count < shortfall, 'operations': []}
        self._record(key, entry['stage'], planting_plan=plan)
        return plan

    def _crop_wait(self, key, reason):
        entry = self.state['items'][key]
        plan = entry.get('planting_plan') or {}
        return ResourceResult('waiting', reason, {
            'item': key, 'defer_item': True, 'retry_after_seconds': 60,
            'required': entry.get('demand', {}).get('required'),
            'growing_plots': entry.get('growing_plots', 0),
            'plots_remaining': max(0, plan.get('planned', 0)-plan.get('confirmed', 0)),
            'seed_limited': plan.get('seed_limited', False)})

    def _continue_planting(self, result, icon, key):
        """Each extra plot gets its own picker, stock check and durable gesture."""
        for _ in range(16):
            entry = self.state['items'].get(key, {})
            plan = entry.get('planting_plan')
            if not plan or entry['stage'] != 'growing' or result.status != 'waiting':
                return result
            if plan['confirmed'] >= plan['planned']:
                reason = ('Available seeds are planted; another harvest cycle is needed to reach the recipe quantity.'
                          if plan['seed_limited'] else
                          'The planned crop quantity is growing; inventory will be checked after harvesting.')
                return self._crop_wait(key, reason)
            frame = self._frame()
            if CameraNavigator._modal_visible(frame):
                return self._crop_wait(key, 'A popup obscures the remaining crop plots; the planting plan is saved.')
            if self._see('growing', frame) or self._see('seed_menu', frame):
                frame = self._close(frame)
            candidates = self._see('empty_plots', frame)
            if not candidates:
                return self._crop_wait(key, 'No additional empty plot was verified; the remaining crop quantity is saved.')
            fresh = self._frame()
            started = time.monotonic()
            # Snow or rewards can briefly cover the highest-scoring tile.
            # Any tile observed empty in both frames can continue this plan.
            current = max((target for target in self._see('empty_plots', fresh)
                           if any(self._same_target(candidate, target, fresh) for candidate in candidates)),
                          key=lambda target: target.score, default=None)
            if current is None:
                return self._crop_wait(key, 'The next empty plot changed; the remaining crop quantity is saved.')
            self._fresh(started)
            selected = self._tap(current.center, fresh)
            selected, plot = self._await_seed_picker(selected, current.center, before=fresh)
            if plot is None:
                return self._crop_wait(key, 'The next plot’s empty seed picker was not confirmed.')
            self.progress(f"Planting required crop plot {plan['confirmed']+1} of {plan['planned']}.")
            result = self._plant_selected(selected, icon, key, plot)
        if self.state['items'][key]['stage'] != 'growing' or result.status != 'waiting':
            return result
        return self._crop_wait(key, 'The crop planting pass is complete; its remaining plots are saved for the next visit.')

    @staticmethod
    def _same_target(first, second, frame):
        if first is None or second is None:
            return False
        return (np.linalg.norm(np.subtract(first.center, second.center)) <= max(4, 8*frame.height/1080)
                and abs(first.width-second.width) <= max(3, first.width*.08)
                and abs(first.height-second.height) <= max(3, first.height*.08))

    @staticmethod
    def _translated_plot(before, after, point, *, require_visible=True, allow_scale=False):
        """Track a just-harvested plot only across a verified camera translation."""
        matrix = FarmingWorker._plot_transform(before, after, allow_scale=allow_scale)
        if matrix is None:
            return None
        projected = matrix @ np.array([*point, 1])
        w, h = after.width, after.height
        if require_visible and not (w*.13 < projected[0] < w*.88 and h*.15 < projected[1] < h*.88):
            return None
        return tuple(int(round(n)) for n in projected)

    @staticmethod
    def _plot_transform(before, after, *, allow_scale=False):
        """Require distributed farm features before relating two camera views."""
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
        scale = np.hypot(matrix[0, 0], matrix[1, 0])
        if (not (.5 <= scale <= 2) if allow_scale else abs(scale-1) > .025):
            return None
        if abs(matrix[1, 0])/scale > .02:
            return None
        return matrix

    @staticmethod
    def _stock_box(frame, item, *, separate=False, upper_margin=.7):
        image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), 1)
        if image is None:
            return None
        h, w = image.shape[:2]
        x0, y0 = max(0, int(item.x-item.width*1.5)), max(0, int(item.y-item.height*upper_margin))
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

    def _await_seed_picker(self, frame, expected, *, before=None):
        """Wait for two stable picker frames after the one authorized plot tap."""
        previous = None
        previous_stamp = None
        for attempt in range(5):
            if self._see('seed_menu', frame):
                plot = self._see('empty_plot', frame)
                position = self._translated_plot(before, frame, expected) if before is not None else expected
                if (plot is not None and position is not None
                        and np.linalg.norm(np.subtract(plot.center, position))
                        <= max(24, plot.width*.8)):
                    if frame.captured_at != previous_stamp and self._same_target(previous, plot, frame):
                        return frame, plot
                    previous = plot
                else:
                    previous = None
            else:
                previous = None
            previous_stamp = frame.captured_at
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
    def _exposed_soil(before, after, harvest, *, allow_scale=False):
        """Find newly exposed brown soil at this crop after verified translation."""
        outline = harvest.highlight
        if outline is None:
            return None
        matrix = FarmingWorker._plot_transform(before, after, allow_scale=allow_scale)
        if matrix is None:
            return None
        projected = matrix @ np.array([*outline.center, 1])
        if not (after.width*.13 < projected[0] < after.width*.88
                and after.height*.15 < projected[1] < after.height*.88):
            return None
        scale = harvest.scale*np.hypot(matrix[0, 0], matrix[1, 0])
        first = cv2.imdecode(np.frombuffer(before.png, np.uint8), 1)
        second = cv2.imdecode(np.frombuffer(after.png, np.uint8), 1)
        height, width = second.shape[:2]
        first = cv2.warpAffine(first, matrix, (width, height))
        x, y, w, h = outline.box
        corners = np.array([[x, y, 1], [x+w, y, 1], [x+w, y+h, 1], [x, y+h, 1]]) @ matrix.T
        left, top = np.rint(corners.min(axis=0)).astype(int)
        right, bottom = np.rint(corners.max(axis=0)).astype(int)
        if left < 0 or top < 0 or right > width or bottom > height:
            return None
        w, h = right-left, bottom-top
        # Hue excludes red tomatoes and yellow kernels that a broad RGB brown
        # test can misclassify as soil. Compare the same observed farm region.
        masks = [cv2.inRange(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV),
                            (8, 85, 55), (24, 235, 230)) for image in (first, second)]
        changed = ((masks[1] > 0) & (masks[0] == 0)).astype(np.uint8)
        # The disappearing tool palette exposes neighboring soil as well.
        # Require the selected plant's own interior to become mostly soil;
        # palette edges or a moving white selection ring cannot prove harvest.
        core = np.s_[round(h*.18):round(h*.7), round(w*.2):round(w*.8)]
        if (not changed[core].size or changed[core].mean() < .2
                or (masks[1][core] > 0).mean() < .6):
            return None
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN,
                                  np.ones((max(1, round(2*scale)),)*2, np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE,
                                  np.ones((max(3, round(5*scale)),)*2, np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
        destination = matrix @ np.array([*(harvest.drag_target or harvest.target).center, 1])-np.array([left, top])
        choices = []
        for label, (_rx, _ry, rw, rh, area) in enumerate(stats[1:], 1):
            if area < max(40, 120*scale**2) or min(rw, rh) < 10*scale:
                continue
            ys, xs = np.nonzero((labels == label) & (masks[1] > 0))
            if not len(xs):
                continue
            cx, cy = destination
            nearest = int(np.argmin((xs-cx)**2+(ys-cy)**2))
            # The input point must be an actual soil pixel, not just the center
            # of a bounding box that could contain neighboring foliage.
            point = int(left+xs[nearest]), int(top+ys[nearest])
            choices.append((int(area), point))
        choices.sort(reverse=True)
        if not choices or len(choices) > 1 and choices[1][0] >= choices[0][0]*.65:
            return None
        point = choices[0][1]
        radius = max(4, round(8*scale))
        return VisualTarget(point[0]-radius, point[1]-radius, radius*2, radius*2, 1.)

    def _await_harvested_soil(self, before, harvest, key):
        prior = None
        prior_stamp = before.captured_at
        evidence_frames = []
        for attempt in range(6):
            self._pause(.75 if attempt == 0 else .4)
            frame = self._frame()
            evidence_frames = [*evidence_frames[-1:], frame]
            soil = None if self._see('harvest', frame) else self._exposed_soil(before, frame, harvest)
            self._check()
            if frame.captured_at != prior_stamp and self._same_target(prior, soil, frame):
                break
            prior = soil
            prior_stamp = frame.captured_at
        else:
            soil = None
        operation = self.state['items'][key]['operation']
        evidence = [self._evidence(frame, operation, f'harvest_after_{i}')
                    for i, frame in enumerate(evidence_frames)]
        self._record(key, 'harvested_needs_replant' if soil else 'harvest_attempted',
                     harvest_after=evidence)
        return frame, soil

    def _confirm_crop_harvest(self, key):
        entry = self.state['items'][key]
        receipt = entry.get('harvest_receipt') or {}
        if receipt.get('operation') == entry['operation'] and receipt.get('outcome') == 'harvested':
            return
        self._record(key, 'harvested_needs_replant', planting_plan=None,
            growing_plots=max(0, entry.get('growing_plots', 0)-1),
            harvest_receipt={'operation': entry['operation'], 'outcome': 'harvested',
                             'evidence': [entry['harvest_before'], *entry['harvest_after']]})

    def _saved_harvest_soil(self, key):
        """Validate both intact post-harvest observations before retiring its intent."""
        entry = self.state['items'][key]
        frames = self._harvest_frames(entry)
        if frames is None:
            return None
        original = self._see('harvest', frames[0])
        if original is None:
            return None
        soils = [None if self._see('harvest', after) else
                 self._exposed_soil(frames[0], after, original) for after in frames[1:]]
        if not self._same_target(*soils, frames[-1]):
            return None
        return frames, original

    def _resume_harvested_soil(self, frame, key):
        """Reopen the original harvested plot after a restart, without a drag."""
        entry = self.state['items'][key]
        saved = self._saved_harvest_soil(key)
        if saved is None:
            return None
        frames, original = saved
        self._confirm_crop_harvest(key)
        if CameraNavigator._modal_visible(frame):
            return None
        if any(self._see(method, frame) for method in ('seed_menu', 'harvest', 'growing')):
            frame = self._close(frame)
        previous = None
        previous_stamp = None
        for attempt in range(3):
            started = time.monotonic()
            if CameraNavigator._modal_visible(frame):
                return None
            soil = self._exposed_soil(frames[0], frame, original, allow_scale=True)
            if frame.captured_at != previous_stamp and self._same_target(previous, soil, frame):
                self._fresh(started)
                opened = self._tap(soil.center, frame)
                opened, plot = self._await_seed_picker(opened, soil.center, before=frame)
                if plot:
                    proof = self._evidence(opened, entry['operation'], 'replant_picker')
                    self._record(key, 'harvested_needs_replant',
                                 replant_evidence={**proof, 'plot': list(plot.box)})
                    return opened, plot
                return None
            previous, previous_stamp = soil, frame.captured_at
            if attempt < 2:
                frame = self._frame()
        return None

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

    def work_if_recognized(self, frame, icon, key, baseline=None):
        self._size = frame.width, frame.height
        self._check()
        pending = self.state['items'].get(key, {}).get('stage')
        if pending in self._PENDING:
            self._requirement(key, baseline)
            if pending == 'harvest_attempted' and self._unchanged_harvest(key):
                self._close(frame)
                return ResourceResult('waiting',
                    'The saved post-harvest frames show the same mature crop still selected; the ineffective drag is reconciled.',
                    {'item': key, 'wait_seconds': 1})
            if pending == 'plant_attempted':
                reconciled = self._unchanged_planting(frame, icon, key)
                if reconciled is not None:
                    return reconciled
            if (pending in {'harvest_attempted', 'harvested_needs_replant'}
                    and isinstance(baseline, dict) and type(baseline.get('available')) is int
                    and type(baseline.get('required')) is int and baseline['required'] > 0
                    and baseline['available'] >= 0 and self._saved_harvest_soil(key) is not None):
                self._confirm_crop_harvest(key)
                entry = self.state['items'][key]
                budget = self._plant_budget(baseline['available'], baseline['required'], entry.get('growing_plots', 0))
                if budget == (0, 0):
                    # The harvested tile no longer needs replanting. Preserve
                    # its receipt without forcing a return to that empty tile.
                    self._plan(key, baseline['available'])
                    self._record(key, 'stock_check', replanted=False)
                    self._close(frame)
                    return self._crop_wait(key, 'The saved harvest is confirmed; existing stock and planted crops cover the recipe and seed reserve.')
            resumed = self._resume_picker(frame, self.state['items'][key])
            if resumed:
                result = self._plant_selected(frame, icon, key, resumed, True, resumed.center)
                return self._continue_planting(result, icon, key)
            if pending in {'harvest_attempted', 'harvested_needs_replant'}:
                recovered = self._resume_harvested_soil(frame, key)
                if recovered:
                    frame, plot = recovered
                    result = self._plant_selected(frame, icon, key, plot, True, plot.center)
                    return self._continue_planting(result, icon, key)
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
            self._requirement(key, baseline)
            self._close(fresh)
            entry = self.state['items'][key]
            available = (baseline or {}).get('available')
            if entry.get('demand') and type(available) is int and available >= 0:
                self._plan(key, available)
                return self._continue_planting(ResourceResult('waiting', 'Crop growth confirmed.'), icon, key)
            return ResourceResult('waiting', 'The selected plant is already in place and still growing; checking other requirements.',
                                  {'wait_seconds': 30, 'item': key, 'defer_item': True, 'retry_after_seconds': 30})
        harvest = self._see('harvest', frame)
        harvested = False
        plot = None
        tracked_plot = None
        if harvest:
            self._requirement(key, baseline)
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
            self.client.drag_path([checked.tool.center, (checked.drag_target or checked.target).center],
                                  width=fresh.width, height=fresh.height, duration_ms=1000,
                                  cancel_event=self.cancel_event, min_waypoint_ms=100,
                                  max_step_px=max(1, round(20*fresh.height/1080)))
            after, soil = self._await_harvested_soil(fresh, checked, key)
            if soil is None:
                return ResourceResult('unsupported', 'Harvest input was sent once, but newly exposed soil was not confirmed in two frames. Its intent remains pending.')
            self._confirm_crop_harvest(key)
            position = soil.center
            frame = self._tap(position, after)
            frame, plot = self._await_seed_picker(frame, position, before=after)
            tracked_plot = plot.center if plot is not None else None
            harvested = True
        else:
            if not self._see('seed_menu', frame):
                return None
            # Fishing lures share the paging buttons and gold drag arrows.
            # Only a selected soil tile establishes a planting workflow.
            plot = self._see('empty_plot', frame)
            if plot is None:
                return None
            self._requirement(key, baseline)
        if harvested and plot is not None:
            proof = self._evidence(frame, self.state['items'][key]['operation'], 'replant_picker')
            self._record(key, 'harvested_needs_replant', replant_evidence={**proof, 'plot': list(plot.box)})
        result = self._plant_selected(frame, icon, key, plot, harvested, tracked_plot)
        return self._continue_planting(result, icon, key)

    def _harvest_frames(self, entry):
        """Read the original attempt's two immediate, intact observations."""
        operation = entry.get('operation')
        proofs = [entry.get('harvest_before'), *entry.get('harvest_after', [])]
        if not isinstance(operation, str) or Path(operation).name != operation or len(proofs) != 3:
            return None
        frames = []
        try:
            for proof in proofs:
                name = proof['file']
                if Path(name).name != name or not name.startswith(operation+'_harvest_'):
                    return None
                png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
                if hashlib.sha256(png).hexdigest() != proof['sha256']:
                    return None
                image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
                if image is None or image.shape[:2] != (proof['height'], proof['width']):
                    return None
                frames.append(Screenshot(png, proof['width'], proof['height'], proof['captured_at']))
            stamps = [datetime.fromisoformat(frame.captured_at) for frame in frames]
            # These observations belong to the original attempt, not a later
            # visit where manual harvesting/replanting could have regrown crops.
            if not (0 < (stamps[1]-stamps[0]).total_seconds()
                    < (stamps[2]-stamps[0]).total_seconds() <= 30):
                return None
        except (OSError, KeyError, TypeError, ValueError, cv2.error):
            return None
        return frames

    def _unchanged_harvest(self, key):
        """Resolve only a no-effect attempt proven by its immediate saved frames."""
        entry = self.state['items'][key]
        frames = self._harvest_frames(entry)
        if frames is None:
            return False
        try:
            original = self._see('harvest', frames[0])
            if original is None or original.highlight is None:
                return False
            x, y, w, h = original.highlight.box
            padding = round(16*original.scale)
            x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
            image = cv2.imdecode(np.frombuffer(frames[0].png, np.uint8), 1)
            before = image[y:y+h, x:x+w]
            for frame in frames[1:]:
                checked = self._see('harvest', frame)
                projected = self._translated_plot(frames[0], frame, original.highlight.center)
                if (checked is None or checked.highlight is None or projected is None
                        or not self._same_target(original.tool, checked.tool, frame)
                        or np.linalg.norm(np.subtract(projected, checked.highlight.center)) > 24*original.scale):
                    return False
                dx, dy = np.subtract(projected, original.highlight.center)
                current = cv2.imdecode(np.frombuffer(frame.png, np.uint8), 1)
                after = current[y+dy:y+dy+h, x+dx:x+dx+w]
                if before.shape != after.shape or not before.size:
                    return False
                difference = after.astype(float)-before.astype(float)
                # Passing clouds alter brightness without removing the crop.
                # Compensate only a bounded channel offset, while still requiring
                # the plant's spatial detail and both selected-plot controls.
                offset = np.median(difference, axis=(0, 1))
                correlation = np.corrcoef(before.mean(axis=2).ravel(), after.mean(axis=2).ravel())[0, 1]
                if (np.max(np.abs(offset)) > 45 or np.abs(difference-offset).mean() >= 24
                        or not np.isfinite(correlation) or correlation < .8):
                    return False
        except (OSError, KeyError, TypeError, ValueError, cv2.error):
            return False
        self._record(key, 'harvest_no_effect', harvest_receipt={
            'operation': entry['operation'], 'outcome': 'no_effect',
            'evidence': [entry['harvest_before'], *entry['harvest_after']]})
        return True

    def _unchanged_planting(self, frame, icon, key):
        """An unchanged seed count and empty picker can prove a drag had no effect."""
        entry = self.state['items'][key]
        available = entry.get('available_before')
        if type(available) is not int or available < 1 or not self._see('seed_menu', frame):
            return None
        plot = self._see('empty_plot', frame)
        seed = self._seed(frame, icon, plot) if plot else None
        if seed is None or self._count(frame, seed) != available:
            return None
        fresh = self._frame()
        checked_plot = self._see('empty_plot', fresh)
        if (fresh.captured_at == frame.captured_at or not self._see('seed_menu', fresh)
                or not self._same_target(plot, checked_plot, fresh)):
            return None
        checked_seed = self._seed(fresh, icon, checked_plot)
        if (not self._same_target(seed, checked_seed, fresh)
                or self._count(fresh, checked_seed) != available):
            return None
        operation = entry.get('operation')
        if not isinstance(operation, str) or Path(operation).name != operation:
            raise ResourceChanged('Invalid saved planting operation.')
        proof = [self._evidence(f, operation, f'plant_unchanged_{i}')
                 for i, f in enumerate((frame, fresh))]
        stage = 'harvested_needs_replant' if entry.get('replant_evidence') else 'plant_no_effect'
        self._record(key, stage, plant_receipt={'operation': operation, 'outcome': 'no_effect',
                     'before': available, 'after': available, 'evidence': proof})
        self._close(fresh)
        return ResourceResult('waiting',
            'The plot remains empty and seed stock is unchanged in two frames; the unsuccessful planting is reconciled.',
            {'item': key, 'wait_seconds': 1})

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
                if self.state['items'].get(key, {}).get('demand'):
                    self._plan(key, 0)
                    self._close(frame)
                    return self._crop_wait(key, 'No seed stock remains; waiting for planted crops before continuing the required quantity.')
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
            if available == 0 and self.state['items'].get(key, {}).get('demand'):
                self._plan(key, 0)
                return self._crop_wait(key, 'No seed stock remains; waiting for planted crops before continuing the required quantity.')
            return ResourceResult('unsupported', 'Seed stock could not be confirmed positive; the plot was not planted.')
        plan = self._plan(key, available) if self.state['items'].get(key, {}).get('demand') else None
        if plan and plan['planned'] == 0:
            self._record(key, 'stock_check', replanted=False)
            self._close(frame)
            return self._crop_wait(key, 'Existing stock and planted crops cover the recipe and seed reserve; checking inventory.')
        if plan and available != plan['stock_before']-plan['confirmed']:
            return ResourceResult('changed', 'Seed stock changed since this planting plan; its remaining plots are preserved.')
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
        operation = uuid.uuid4().hex
        proof = self._evidence(fresh, operation, 'plant_before')
        self._record(key, 'plant_attempted', available_before=available, operation=operation,
                     replanted=False,
                     replant_evidence=self.state['items'].get(key, {}).get('replant_evidence') if harvested else None,
                     plant_before={**proof, 'plot': list(fresh_plot.box), 'seed': list(fresh_seed.box)})
        self._fresh(started)
        destination = tracked_plot or fresh_plot.center
        self.client.drag_path([fresh_seed.center, destination], width=fresh.width, height=fresh.height,
                              duration_ms=1000, cancel_event=self.cancel_event,
                              min_waypoint_ms=100, max_step_px=max(1, round(20*fresh.height/1080)))
        prior_timer = None
        prior_stamp = fresh.captured_at
        evidence_frames = []
        for _ in range(6):
            self._pause(.4)
            frame = self._frame()
            evidence_frames = [*evidence_frames[-1:], frame]
            timer = self._see('growing', frame)
            if frame.captured_at != prior_stamp and self._same_target(prior_timer, timer, frame):
                details = {}
                if plan:
                    details = {'planting_plan': {**plan, 'confirmed': plan['confirmed']+1,
                               'operations': [*plan['operations'], operation]},
                               'growing_plots': self.state['items'][key].get('growing_plots', 0)+1,
                               'plant_confirmed': [self._evidence(f, operation, f'growth_{i}')
                                                   for i, f in enumerate(evidence_frames)]}
                self._record(key, 'growing', replanted=True, **details)
                self._close(frame)
                message = ('The selected crop was replanted after a harvest attempt.' if harvested
                           else 'The selected empty plot was planted.')
                return ResourceResult('waiting', message+' Its growth timer was confirmed; returning to verify order stock.', {'wait_seconds': 1})
            prior_timer = timer
            prior_stamp = frame.captured_at
        self._record(key, 'plant_attempted', plant_after=[
            self._evidence(f, operation, f'plant_after_{i}') for i, f in enumerate(evidence_frames)])
        return ResourceResult('unsupported', 'Planting was attempted, but two matching growth frames were not confirmed; its intent remains pending.')

"""Recognize the fruit-basket guide and preserve one attempted harvest.

Guide disappearance never proves collection. Only the caller's subsequent,
consistent inventory observations can retire a pending harvest intent.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.resources import ResourceChanged, ResourceResult, _save_json


@dataclass(frozen=True)
class FruitTarget:
    tool: VisualTarget
    target: VisualTarget | None
    score: float
    scale: float
    arrow: VisualTarget
    species: str | None = None
    fruit_features: tuple[VisualTarget, ...] = ()
    sweep_path: tuple[tuple[int, int], ...] = ()
    pen_polygon: tuple[tuple[int, int], ...] = ()
    action: str = 'harvest'
    available: int | None = None
    enabled: bool = True


class FruitVision:
    """Recognize the tool menu and independently locate the requested ripe fruit."""

    def __init__(self, reference_path: Path | None = None):
        source = Path(__file__).resolve().parents[2]/'images/fruit'
        self.reference_path = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/fruit')
        try:
            self.manifest = json.loads((self.reference_path/'manifest.json').read_text('utf-8'))
            if self.manifest.get('version') != 1:
                raise ValueError('Unsupported fruit references.')
            self.references = {name: _decode((self.reference_path/f'{name}.png').read_bytes(), True)
                               for name in ('basket', 'arrow')}
            for name, rgba in self.references.items():
                box = self.manifest['features'][name]['box']
                if (rgba.ndim != 3 or rgba.shape[2] != 4
                        or rgba.shape[:2] != (box[3]-box[1], box[2]-box[0])
                        or np.count_nonzero(rgba[:, :, 3]) < 100):
                    raise ValueError('Fruit references require original pixels and alpha masks.')
            canopy = self.manifest.get('canopy', {})
            self.canopy_references = tuple(
                (spec, _decode((self.reference_path/spec['file']).read_bytes(), True))
                for spec in canopy.get('references', ()))
            item = canopy.get('desired_item')
            self.cherry_item = (self.reference_path/item['file']).read_bytes() if item else None
            raspberry = self.manifest.get('raspberry', {})
            self.raspberry_references = tuple(
                (spec, _decode((self.reference_path/spec['file']).read_bytes(), True))
                for spec in raspberry.get('references', ()))
            item = raspberry.get('desired_item')
            self.raspberry_item = (self.reference_path/item['file']).read_bytes() if item else None
            self.raspberry_variants = tuple(
                (self.reference_path/spec['file']).read_bytes()
                for spec in raspberry.get('desired_variants', ()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Fruit guide references are missing or invalid.') from exc
        self._matcher = ResourceVision()
        self._sheep = None
        self._chicken = None

    @staticmethod
    def _relative(box, basket, source, scale):
        x0, y0, x1, y1 = box
        return VisualTarget(round(basket.x+(x0-source[0])*scale),
                            round(basket.y+(y0-source[1])*scale),
                            max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale)),
                            basket.score)

    @staticmethod
    def _check(cancel):
        if cancel():
            raise ResourceChanged('Fruit recognition cancelled.')

    def _desired_cherry(self, desired_icon, cancel):
        if desired_icon is None:
            return True  # Read-only guide diagnostics may omit the desired item.
        return self._desired_item(desired_icon, self.cherry_item, cancel)

    def _desired_item(self, desired_icon, reference, cancel, minimum_score=.93):
        if desired_icon is None or reference is None:
            return False
        try:
            found = self._matcher.find_item(desired_icon, reference,
                min_scale=.45, max_scale=2.2, cancel=cancel)
        except ValueError:
            return False
        self._check(cancel)
        return bool(found and found[0].score >= minimum_score)

    def _clusters(self, roi, reference, scales, cancel):
        """Validate small fruit patterns at native resolution without coarse pruning."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dimensions, found = set(), []
        for scale in scales:
            self._check(cancel)
            color, mask = self._matcher._scaled(reference, float(scale))
            height, width = mask.shape
            if ((width, height) in dimensions or height > gray.shape[0]
                    or width > gray.shape[1] or np.count_nonzero(mask) < 20):
                continue
            dimensions.add((width, height))
            response = cv2.matchTemplate(gray, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
                                         cv2.TM_CCOEFF_NORMED, mask=mask)
            response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
            for _ in range(8):
                _, correlation, _, (x, y) = cv2.minMaxLoc(response)
                if correlation < .825:
                    break
                patch = roi[y:y+height, x:x+width]
                # Passing clouds add neutral light and reduce contrast. Fit one
                # shared gain/offset across all RGB channels; independent channel
                # corrections would incorrectly turn other fruit into cherries.
                observed = patch[mask > 0].astype(np.float32).ravel()
                expected = color[mask > 0].astype(np.float32).ravel()
                centered = expected-expected.mean()
                gain = np.clip(np.dot(centered, observed-observed.mean())
                               /max(1., float(np.dot(centered, centered))), .55, 1.10)
                offset = np.clip(observed.mean()-gain*expected.mean(), -15., 120.)
                difference = np.abs(observed-(gain*expected+offset)).mean()
                score = .8*correlation+.2*max(0., 1.-float(difference)/100.)
                if score >= .86:
                    found.append(VisualTarget(x, y, width, height, score))
                rx, ry = max(3, width//2), max(3, height//2)
                response[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1] = -1
        return tuple(found)

    @staticmethod
    def _foliage_context(image, target):
        """A single detailed bunch still needs surrounding leaf texture."""
        margin = max(5, round(target.width*.7))
        left, top = max(0, target.x-margin), max(0, target.y-margin)
        right = min(image.shape[1], target.x+target.width+margin)
        bottom = min(image.shape[0], target.y+target.height+margin)
        patch = image[top:bottom, left:right]
        blue, green, red = (patch[:, :, channel].astype(np.int16) for channel in range(3))
        leaves = (green > 50) & (green > red*1.10) & (green > blue*1.20)
        leaves[target.y-top:target.y-top+target.height,
               target.x-left:target.x-left+target.width] = False
        if leaves.mean() < .14:
            return False
        values = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[leaves]
        return bool(len(values) >= 20 and np.percentile(values, 90)-np.percentile(values, 10) >= 20)

    def _canopy(self, image, basket, tool, source, scale, cancel, species='cherry'):
        """The UI scale bounds the search vicinity, never the size of the tree."""
        height, width = image.shape[:2]
        tx, ty = tool.center
        left, right = max(0, round(tx+75*scale)), min(width, round(tx+520*scale))
        top = max(0, round(ty-240*scale))
        bottom = min(height, round(ty+(240 if species == 'raspberry' else 120)*scale))
        if right <= left or bottom <= top:
            return None, ()
        roi = image[top:bottom, left:right]
        candidates = []
        references = self.raspberry_references if species == 'raspberry' else self.canopy_references
        for spec, reference in references:
            self._check(cancel)
            low, high = spec['world_scale_range']
            # Integer widths include tiny world clusters that lose their shape
            # when the ordinary full-frame matcher downsamples or prunes peaks.
            sizes = np.unique(np.r_[np.arange(max(8, round(low*scale*reference.shape[1])),
                round(high*scale*reference.shape[1])+1, .25)/reference.shape[1], scale])
            found = self._clusters(roi, reference, sizes, cancel)
            for match in found:
                target = VisualTarget(match.x+left, match.y+top, match.width, match.height, match.score)
                cx, cy = target.center
                # The slanted white edge belongs to the tool menu. Ignore fruit
                # behind it; locate visible clusters in the world separately.
                reference_y = source[1]+(cy-basket.y)/scale
                edge_x = 924+(reference_y-480)*49/224
                boundary = basket.x+(edge_x-source[0])*scale+10*scale
                if species == 'raspberry':
                    # The compact bush can overlap the translucent wedge. Its
                    # own leaf/berry pattern must lie beyond the basket arrow,
                    # rather than beyond the cherry tree's taller menu edge.
                    arrow_right = self.manifest['features']['arrow']['box'][2]
                    boundary = basket.x+(arrow_right-source[0]+8)*scale
                if cx-target.width/2 < boundary or target.width < 12*scale:
                    continue
                patch = image[target.y:target.y+target.height, target.x:target.x+target.width]
                blue, green, red = (patch[:, :, channel].astype(np.int16) for channel in range(3))
                fruit_pixels = (red > 80) & (red > green*1.6) & (blue > green*1.2) & (red > blue*1.5)
                # Raspberry bunches are separated by leaves even when ripe.
                if np.mean(fruit_pixels) < (.12 if species == 'raspberry' else .20):
                    continue
                candidates.append(target)
        self._check(cancel)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        distinct = []
        for candidate in candidates:
            if not any(np.linalg.norm(np.subtract(candidate.center, other.center))
                       < min(candidate.width, other.width)*.65 for other in distinct):
                distinct.append(candidate)
        supported = []
        for candidate in distinct:
            peers = tuple(other for other in distinct if other is not candidate
                and .72 <= other.width/candidate.width <= 1.4
                and max(candidate.width, other.width)*1.2
                    <= np.linalg.norm(np.subtract(candidate.center, other.center))
                    <= max(candidate.width, other.width)*5)
            if peers or (candidate.score >= .93 and self._foliage_context(image, candidate)):
                supported.append((candidate, peers))
        if not supported:
            return None, ()
        # Repeated fruit clusters are expected within a canopy. The nearest
        # supported cluster minimizes the drag without relying on score jitter.
        candidate, peers = min(supported,
            key=lambda pair: (np.linalg.norm(np.subtract(pair[0].center, tool.center)),
                              pair[0].center[0], pair[0].center[1]))
        # Revalidation may keep a previously selected cluster even when another
        # valid bunch becomes slightly closer as foliage sways or clouds pass.
        evidence = tuple(pair[0] for pair in supported)
        return candidate, evidence

    def basket(self, png: bytes, cancel: Callable[[], bool] | None = None,
               desired_icon: bytes | None = None) -> FruitTarget | None:
        cancel = cancel or (lambda: False)
        self._check(cancel)
        if desired_icon is not None:
            if self._chicken is None:
                from hayday.chickens import ChickenVision
                self._chicken = ChickenVision()
            chicken = self._chicken.basket(png, desired_icon, cancel)
            self._check(cancel)
            if chicken is not None:
                return chicken
            if self._sheep is None:
                from hayday.animals import SheepVision
                self._sheep = SheepVision()
            sheep = self._sheep.harvest(png, desired_icon, cancel)
            self._check(cancel)
            if sheep is not None:
                return sheep
            feeding = self._sheep.harvest(png, desired_icon, cancel, feeding=True)
            self._check(cancel)
            if feeding is not None:
                # Disabled gray shears mean there is no ready wool. Keep the
                # recognized sheep menu available to the separate feed action.
                return replace(feeding, target=None, sweep_path=())
        image = _decode(png)
        desired_cherry = self._desired_cherry(desired_icon, cancel)
        species = 'cherry' if desired_cherry else (
            'raspberry' if any(self._desired_item(desired_icon, reference, cancel, .92)
                for reference in (self.raspberry_item, *self.raspberry_variants)) else None)
        base = image.shape[0]/1080
        scales = np.unique(np.r_[np.geomspace(base*.55, base*1.5, 23), base])
        baskets = self._matcher._search(image, self.references['basket'], scales, .90, cancel, max_peaks=4)
        self._check(cancel)
        if not baskets:
            return None
        arrows = self._matcher._search(image, self.references['arrow'], scales, .90, cancel, max_peaks=4)
        self._check(cancel)
        source = self.manifest['features']['basket']['box']
        candidates = []
        for basket in baskets:
            scale = basket.width/self.references['basket'].shape[1]
            expected = self._relative(self.manifest['features']['arrow']['box'], basket, source, scale)
            arrow = next((candidate for candidate in arrows
                          if np.linalg.norm(np.subtract(candidate.center, expected.center)) <= 15*scale
                          and .87 <= candidate.width/expected.width <= 1.13), None)
            if arrow is None:
                continue
            highlights = []
            for box in self.manifest['highlight_boxes']:
                x, y, width, height = self._relative(box, basket, source, scale).box
                if x < 0 or y < 0 or x+width > image.shape[1] or y+height > image.shape[0]:
                    highlights.append(False)
                    continue
                patch = image[y:y+height, x:x+width]
                low, high = patch.min(axis=2), patch.max(axis=2)
                white = (low > 195) & (high-low < 65)
                highlights.append(np.count_nonzero(white) >= max(30, 130*scale*scale))
            if not all(highlights):
                continue
            tx, ty = self.manifest['tool_point']
            tool = self._relative((tx-6, ty-6, tx+6, ty+6), basket, source, scale)
            target, features = (self._canopy(image, basket, tool, source, scale, cancel, species)
                                if species else (None, ()))
            candidates.append(FruitTarget(tool, target, min(basket.score, arrow.score), scale, arrow,
                                          species if target else None, features))
        self._check(cancel)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0].score-candidates[1].score < .025):
            return None
        return candidates[0]

    def sheep_feed(self, png, icon, cancel, *, pen_hint=None):
        if self._sheep is None:
            from hayday.animals import SheepVision
            self._sheep = SheepVision()
        return self._sheep.harvest(png, icon, cancel, feeding=True, pen_hint=pen_hint)

    def sheep_enclosure(self, png, icon, cancel):
        if self._sheep is None:
            from hayday.animals import SheepVision
            self._sheep = SheepVision()
        return self._sheep.enclosure(png, icon, cancel)

    def chicken_feed(self, png, icon, cancel, **kwargs):
        if self._chicken is None:
            from hayday.chickens import ChickenVision
            self._chicken = ChickenVision()
        return self._chicken.feed(png, icon, cancel)

    def chicken_enclosure(self, png, icon, cancel):
        return None  # Chicken feeding uses its freshly matched coop and floor.


class FruitWorker:
    """One fresh basket drag followed by inventory verification, never a blind retry."""

    def __init__(self, client, capture, cancel_event, progress, state_path, *, vision=None):
        self.client = client
        self.capture = capture
        self.cancel_event = cancel_event
        self.progress = progress
        self.state_path = Path(state_path)
        self.vision = vision or FruitVision()
        self.serial = client.serial
        self._size = None
        self.storage = None
        self.state = {'version': 1, 'serial': self.serial, 'items': {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text('utf-8'))
            if (not isinstance(self.state, dict) or self.state.get('version') != 1
                    or self.state.get('serial') != self.serial
                    or not isinstance(self.state.get('items'), dict)
                    or any(not isinstance(entry, dict) or entry.get('stage') not in {'attempted', 'confirmed', 'no_effect'}
                           for entry in self.state['items'].values())):
                raise ValueError('Saved fruit harvest intent is invalid or belongs to another device.')

    def _check(self):
        if self.cancel_event.is_set() or not self.serial or self.client.serial != self.serial:
            raise ResourceChanged('Fruit work was cancelled or its selected device changed.')

    def _frame(self):
        self._check()
        frame = self.capture()
        self._check()
        if self._size and self._size != (frame.width, frame.height):
            raise ResourceChanged('Device resolution changed during fruit work.')
        return frame

    def _record(self, key, **details):
        entry = self.state['items'].setdefault(key, {})
        entry.update(updated_at=datetime.now(UTC).isoformat(), **details)
        _save_json(self.state_path, self.state)

    def _read(self, frame, icon):
        self._check()
        result = self.vision.basket(frame.png, cancel=self.cancel_event.is_set, desired_icon=icon)
        self._check()
        return result

    @staticmethod
    def _same(first, second, frame):
        if first is None or second is None:
            return False
        return (np.linalg.norm(np.subtract(first.center, second.center)) <= max(4, 8*frame.height/1080)
                and abs(first.width-second.width) <= max(3, first.width*.08)
                and abs(first.height-second.height) <= max(3, first.height*.08))

    @staticmethod
    def _same_pen(first, second):
        if first is None or second is None or len(first.pen_polygon) < 4 or len(second.pen_polygon) < 4:
            return False
        old, new = np.array(first.pen_polygon, np.float32), np.array(second.pen_polygon, np.float32)
        intersection, _ = cv2.intersectConvexConvex(old, new)
        return intersection/max(1., cv2.contourArea(old)+cv2.contourArea(new)-intersection) >= .86

    def _feed_sheep(self, frame, icon, key):
        return self._feed_animal(frame, icon, key, 'sheep')

    def _feed_animal(self, frame, icon, key, animal):
        """One feed sweep with durable intent and two observed stock decreases."""
        entry = self.state['items'].get(key, {})
        if animal == 'sheep' and entry.get('feed_stage') == 'attempted':
            return False
        read_feed = self.vision.sheep_feed if animal == 'sheep' else self.vision.chicken_feed
        read_enclosure = self.vision.sheep_enclosure if animal == 'sheep' else self.vision.chicken_enclosure
        pen_hint = None
        observed = read_feed(frame.png, icon, self.cancel_event.is_set)
        if entry.get('feed_stage') == 'attempted':
            # A disabled tool can still show the shared feed stock. Reconcile
            # a prior sweep using two readings before allowing another one.
            before = entry.get('feed_available_before')
            if observed is not None and type(before) is int and observed.available is not None and observed.available < before:
                fresh = self._frame()
                checked = read_feed(fresh.png, icon, self.cancel_event.is_set)
                if (checked is not None and checked.available == observed.available
                        and fresh.captured_at != frame.captured_at):
                    self._record(key, feed_stage='confirmed', feed_available_after=checked.available)
                    return True
            return False
        if observed is not None and observed.target is None and observed.available:
            # A selected animal's timer can hide the floor and fence. Close
            # that overlay, rediscover the enclosure, and open its trough menu.
            from hayday.camera import CameraNavigator
            ground = CameraNavigator._grass_start(frame, 0, 0)
            if ground is not None:
                self._check()
                self.client.tap(*ground[:2], width=frame.width, height=frame.height)
                self.cancel_event.wait(.3)
                clear = self._frame()
                first = read_enclosure(clear.png, icon, self.cancel_event.is_set)
                if first is not None:
                    fresh = self._frame()
                    started = time.monotonic()
                    second = read_enclosure(fresh.png, icon, self.cancel_event.is_set)
                    if (second is not None and self._same(first[0], second[0], fresh)
                            and time.monotonic()-started <= 4):
                        self._check()
                        self.client.tap(*second[0].center, width=fresh.width, height=fresh.height)
                        self.cancel_event.wait(.4)
                        frame = self._frame()
                        pen_hint = (fresh.png, second[1])
                        observed = read_feed(frame.png, icon, self.cancel_event.is_set,
                                                         pen_hint=pen_hint)
        if observed is None or observed.target is None or not observed.available or not observed.enabled:
            return False
        fresh = self._frame()
        started = time.monotonic()
        checked = read_feed(fresh.png, icon, self.cancel_event.is_set, pen_hint=pen_hint)
        if (not self._same_pen(observed, checked) or checked.target is None or not checked.enabled
                or not self._same(observed.tool, checked.tool, fresh)
                or observed.available != checked.available or not checked.sweep_path):
            return False
        operation = uuid.uuid4().hex
        evidence = self.state_path.parent/'fruit_evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        before_path = evidence/(operation+'_feed_before.png')
        before_path.write_bytes(fresh.png)
        if pen_hint is not None:
            (evidence/(operation+'_feed_pen.png')).write_bytes(pen_hint[0])
        self._record(key, stage=entry.get('stage', 'no_effect'), feed_stage='attempted',
                     feed_operation=operation, feed_available_before=checked.available,
                     feed_before_capture=before_path.name)
        self._check()
        if time.monotonic()-started > 4:
            self._record(key, feed_stage='not_sent')
            return False
        self.progress(f'Sweeping {animal} feed across the verified production pen.')
        self.client.drag_path((checked.tool.center, *checked.sweep_path), width=fresh.width,
                              height=fresh.height, duration_ms=4500, cancel_event=self.cancel_event)
        previous_stock, confirmations, previous_id = None, 0, None
        unchanged_ids = set()
        for _ in range(4):
            self.cancel_event.wait(.5)
            after = self._frame()
            current = read_feed(after.png, icon, self.cancel_event.is_set)
            if (animal == 'chicken' and current is not None and current.available == checked.available
                    and after.captured_at):
                unchanged_ids.add(after.captured_at)
            if current is None or current.available is None or current.available >= checked.available:
                confirmations = 0
                continue
            if after.captured_at == previous_id:
                continue
            confirmations = confirmations+1 if current.available == previous_stock else 1
            previous_stock, previous_id = current.available, after.captured_at
            if confirmations >= 2:
                after_path = evidence/(operation+'_feed_after.png')
                after_path.write_bytes(after.png)
                self._record(key, feed_stage='confirmed', feed_available_after=current.available,
                             feed_after_capture=after_path.name)
                return True
        if animal == 'chicken' and len(unchanged_ids) >= 2:
            self._record(key, feed_stage='no_effect', feed_available_after=checked.available)
        return False

    def _prepare_chickens(self, frame, icon, key, observed):
        from hayday.barn_storage import StorageReader
        from hayday.chicken_care import ChickenCare

        self.storage = self.storage or StorageReader()
        status = self.storage.read(frame.png, self.cancel_event.is_set)
        if status.full:
            return ResourceResult('waiting', 'Barn storage is full; animal collection is paused.',
                {'item':key, 'storage_blocked':True, 'defer_session':True})
        care = ChickenCare(self)
        excluded = []
        for _ in range(3):
            self._check()
            used = (observed.action == 'feed' or observed.target is None
                    or care.recently_collected(frame, key, observed))
            current_pen = (frame, care._center(observed.pen_polygon)) if observed.pen_polygon else None
            self._feed_animal(frame, icon, key, 'chicken')
            frame = self._frame()
            if current_pen:
                excluded.append(current_pen)
            fresh = self._read(frame, icon)
            if not used and fresh is not None and fresh.species == 'egg' and fresh.target:
                status = self.storage.read(frame.png, self.cancel_event.is_set)
                if status.full or status.free is not None and status.free < 6:
                    return ResourceResult('waiting',
                        'There is insufficient verified barn room for a chicken-pen sweep; collection is paused.',
                        {'item':key, 'storage_blocked':True, 'defer_session':True})
                return frame, fresh
            switched = care.next_pen(frame, icon, excluded)
            if switched is None:
                return ResourceResult('waiting',
                    'No additional ready chicken pen was verified in the bounded nearby search. Checking other requirements.',
                    {'item':key, 'defer_item':True, 'retry_after_seconds':600})
            frame, observed = switched
        return ResourceResult('waiting', 'The chicken-pen care pass reached its limit; checking other requirements.',
                              {'item':key, 'defer_item':True, 'retry_after_seconds':600})

    def work_if_recognized(self, frame, icon, key, baseline=None):
        self._check()
        self._size = frame.width, frame.height
        if self.state['items'].get(key, {}).get('stage') == 'attempted':
            return ResourceResult('waiting',
                'A harvest remains pending; checking inventory before any repeat gesture.',
                {'wait_seconds': 1, 'pending_harvest': True, 'item': key})
        observed = self._read(frame, icon)
        if observed is None and isinstance(self.vision, FruitVision):
            observed = self.vision.chicken_feed(frame.png, icon, self.cancel_event.is_set)
        if observed is None:
            return None
        if observed.species == 'egg':
            prepared = self._prepare_chickens(frame, icon, key, observed)
            if isinstance(prepared, ResourceResult):
                return prepared
            frame, observed = prepared
        if observed.species == 'wool' and observed.target is None:
            fed = self._feed_sheep(frame, icon, key)
            if not fed:
                from hayday.farming_vision import FarmingVision
                growth = FarmingVision()
                if growth.growing(frame.png):
                    fresh = self._frame()
                    if growth.growing(fresh.png):
                        return ResourceResult('waiting',
                            'Wool is still growing; checking other requirements.',
                            {'item': key, 'defer_item': True, 'retry_after_seconds': 600})
            return ResourceResult('waiting' if fed else 'unsupported',
                'Sheep feed was used and its stock decrease verified; checking other items while wool grows.'
                if fed else 'No ready wool was verified, and feeding could not be confirmed.',
                {'item': key, 'defer_item': True, 'retry_after_seconds': 600} if fed else {})
        for attempt in range(4):
            fresh = self._frame()
            started = time.monotonic()
            checked = self._read(fresh, icon)
            if (checked is not None and observed is not None and observed.species is not None
                    and checked.species == observed.species
                    and not self._same(observed.target, checked.target, fresh)):
                # Any cluster independently supported in both frames is valid;
                # a passing cloud may hide the previously nearest bunch.
                matches = [(old, new) for old in observed.fruit_features
                           for new in checked.fruit_features if self._same(old, new, fresh)]
                if matches:
                    old, new = min(matches, key=lambda pair:
                        np.linalg.norm(np.subtract(pair[1].center, checked.tool.center)))
                    observed, checked = replace(observed, target=old), replace(checked, target=new)
            stable = (checked is not None and observed is not None
                      and checked.species == observed.species
                      and self._same(observed.tool, checked.tool, fresh)
                      and self._same(observed.target, checked.target, fresh)
                      and self._same(observed.arrow, checked.arrow, fresh))
            if (checked is not None and observed is not None
                    and checked.species == observed.species and checked.species in {'wool', 'egg'}):
                # Animals walk and turn during observation. A whole-pen sweep
                # requires the same enclosure and tool, not an immobile sheep.
                stable = (checked.target is not None and observed.target is not None
                          and self._same(observed.tool, checked.tool, fresh)
                          and self._same(observed.arrow, checked.arrow, fresh)
                          and len(observed.pen_polygon) >= 4 and len(checked.pen_polygon) >= 4)
                if stable:
                    stable = self._same_pen(observed, checked)
            if stable and time.monotonic()-started <= 4:
                break
            observed = checked
            if attempt < 3:
                self.progress('Waiting for the collection tool and harvest target to settle.')
                self.cancel_event.wait(.35)
                self._check()
        else:
            if observed is not None and observed.target is None:
                return ResourceResult('unsupported',
                    'The collection tool is recognized, but no supported harvest target was verified.')
            return ResourceResult('changed',
                'The collection tool and harvest target did not settle; no drag was sent.')
        self._check()
        self.progress('Sweeping the shears across the freshly recognized sheep pen.'
                      if checked.species == 'wool' else
                      'Sweeping the basket across the freshly recognized chicken pen.'
                      if checked.species == 'egg' else
                      'Dragging the basket onto freshly recognized ripe fruit.')
        before = dict(baseline) if isinstance(baseline, dict) else {}
        operation = uuid.uuid4().hex
        evidence = self.state_path.parent / 'fruit_evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        before_path = evidence / (operation + '_before.png')
        before_path.write_bytes(fresh.png)
        self._record(key, stage='attempted', operation=operation,
                     inventory_before=before, inventory_confirmation=None,
                     before_sha256=hashlib.sha256(fresh.png).hexdigest(),
                     before_capture=before_path.name, species=checked.species,
                     pen_polygon=checked.pen_polygon, harvest_observed_at=fresh.captured_at)
        self._check()
        if time.monotonic()-started > 4:
            self._record(key, stage='no_effect')
            return ResourceResult('changed', 'Fruit input was deferred because its saved observation became stale.')
        if checked.sweep_path:
            self.client.drag_path((checked.tool.center, *checked.sweep_path),
                                  width=fresh.width, height=fresh.height, duration_ms=4500,
                                  cancel_event=self.cancel_event)
        else:
            self.client.swipe(*checked.tool.center, *checked.target.center,
                              width=fresh.width, height=fresh.height, duration_ms=1000)
        self.cancel_event.wait(.6)
        self._check()
        after = self._frame()
        after_path = evidence / (operation + '_after.png')
        after_path.write_bytes(after.png)
        self._record(key, after_capture=after_path.name,
                     after_sha256=hashlib.sha256(after.png).hexdigest())
        if checked.species == 'egg' and self.storage is not None:
            status = self.storage.read(after.png, self.cancel_event.is_set)
            if status.full:
                return ResourceResult('waiting', 'Barn storage filled during collection; further harvesting is paused.',
                    {'item':key, 'pending_harvest':True, 'storage_blocked':True, 'defer_session':True})
        if checked.species == 'wool':
            self._feed_sheep(after, icon, key)
        elif checked.species == 'egg':
            self._feed_animal(after, icon, key, 'chicken')
        return ResourceResult('waiting',
            'Harvesting was attempted. Returning to verify the inventory gain.',
            {'wait_seconds': 1, 'pending_harvest': True, 'item': key})

    def observe_inventory(self, key, status, available, required, observation_id):
        """Accept two consistent positive stock observations, including partial gains."""
        self._check()
        entry = self.state['items'].get(key)
        if not entry or entry.get('stage') != 'attempted':
            return
        before = entry.get('inventory_before', {})
        same_requirement = type(required) is int and required > 0 and before.get('required') == required
        increased = (same_requirement and type(available) is int
                     and type(before.get('available')) is int and available > before['available'])
        fulfilled = same_requirement and status == 'fulfilled' and before.get('status') == 'missing'
        if not observation_id or not (increased or fulfilled):
            if entry.get('inventory_confirmation'):
                self._record(key, inventory_confirmation=None)
            return
        evidence = [status, available, required]
        confirmation = entry.get('inventory_confirmation') or {}
        if confirmation.get('observation_id') == observation_id:
            return
        count = confirmation.get('count', 0)+1 if confirmation.get('evidence') == evidence else 1
        self._record(key, inventory_confirmation={'evidence': evidence, 'count': count,
                                                  'observation_id': observation_id})
        if count >= 2:
            self._record(key, stage='confirmed', inventory_confirmation=None,
                         confirmed_available=available, confirmed_required=required)

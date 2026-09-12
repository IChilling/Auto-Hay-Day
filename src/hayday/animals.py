"""Recognize sheep shears and woolly animals independently of farm layout."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode


class SheepVision:
    def __init__(self, reference_path: Path | None = None):
        source = Path(__file__).resolve().parents[2]/'images/animals'
        self.reference_path = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/animals')
        self.manifest = json.loads((self.reference_path/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported sheep references.')
        self.references = {name: _decode((self.reference_path/spec['file']).read_bytes(), True)
                           for name, spec in self.manifest['features'].items()}
        for rgba in self.references.values():
            if rgba.ndim != 3 or rgba.shape[2] != 4 or np.count_nonzero(rgba[:, :, 3]) < 100:
                raise ValueError('Sheep references require original pixels and alpha masks.')
        self.item = (self.reference_path/self.manifest['desired_item']).read_bytes()
        self.matcher = ResourceVision()

    @staticmethod
    def pen(image, animal, *, slope_bounds=(.25, .8), vertices=(4,), close_radius=0):
        """Find the enclosed soil diamond containing the independently seen sheep."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        soil = cv2.inRange(hsv, (12, 85, 75), (32, 255, 245))
        if close_radius:
            soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE,
                                   np.ones((close_radius*2+1, close_radius*2+1), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(soil)
        candidates = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not (animal.width**2*3 < area < animal.width**2*40
                    and x <= animal.center[0] <= x+width
                    and y <= animal.center[1] <= y+height):
                continue
            ys, xs = np.where(labels == index)
            hull = cv2.convexHull(np.column_stack((xs, ys)))
            polygon = cv2.approxPolyDP(hull, .045*cv2.arcLength(hull, True), True)
            # The wooden trough sits against the rear fence, above the soil
            # in the isometric view. Its center can be just outside the floor
            # outline even when its feet are inside the enclosure.
            if (len(polygon) not in vertices
                    or cv2.pointPolygonTest(polygon, animal.center, True) < -animal.width*.25):
                continue
            points = polygon[:, 0]
            edges = np.roll(points, -1, axis=0)-points
            slopes = np.abs(edges[:, 1]/np.maximum(1, np.abs(edges[:, 0])))
            if not (np.all((slopes > slope_bounds[0]) & (slopes < slope_bounds[1]))
                    and .50 < area/cv2.contourArea(polygon) < 1.01):
                continue
            # A soil-colored building or road is insufficient: white pasture
            # fencing must support the observed diamond's perimeter as well.
            ring = np.zeros(image.shape[:2], np.uint8)
            cv2.polylines(ring, [polygon], True, 255, max(3, round(animal.width*.45)))
            pixels = image[ring > 0]
            white = (pixels.min(axis=1) > 175) & (np.ptp(pixels, axis=1) < 65)
            if white.mean() >= .045:
                candidates.append(tuple(tuple(map(int, p)) for p in points))
        return candidates[0] if len(candidates) == 1 else ()

    @staticmethod
    def sweep(polygon, animal_width):
        """Serpentine rows inside the pen, with spacing smaller than one animal."""
        points = np.array(polygon, np.int32)
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        raster = np.zeros((maximum[1]-minimum[1]+1, maximum[0]-minimum[0]+1), np.uint8)
        cv2.fillConvexPoly(raster, points-minimum, 255)
        margin = max(2, round(animal_width*.10))
        raster = cv2.erode(raster, np.ones((margin*2+1, margin*2+1), np.uint8))
        path = []
        for row in range(margin, raster.shape[0]-margin, max(5, round(animal_width*.30))):
            columns = np.flatnonzero(raster[row])
            if len(columns) < margin*2:
                continue
            ends = [int(columns[0]+minimum[0]), int(columns[-1]+minimum[0])]
            if len(path)//2 % 2:
                ends.reverse()
            path.extend((column, int(row+minimum[1])) for column in ends)
        return tuple(path)

    def harvest(self, png, desired_icon, cancel, *, feeding=False, pen_hint=None):
        from hayday.fruit import FruitTarget

        if desired_icon is None or cancel():
            return None
        identity = self.matcher.find_item(desired_icon, self.item, min_scale=.45,
                                         max_scale=2.2, threshold=.93, cancel=cancel)
        if not identity:
            return None
        image = _decode(png)
        base = image.shape[0]/self.manifest['reference_height']
        scales = np.unique(np.r_[np.geomspace(base*.65, base*1.4, 17), base])
        names = ('shears', 'arrow', 'feed') + (('shears_disabled', 'arrow_disabled') if feeding else ())
        matches = {name: self.matcher._search(image, self.references[name], scales, .92,
                                              cancel, max_peaks=3)
                   for name in names}
        source = self.manifest['features']['shears']['box']
        guides = []
        for shears in (*matches['shears'], *matches.get('shears_disabled', ())):
            scale = shears.width/self.references['shears'].shape[1]
            evidence = []
            for name in ('arrow', 'feed'):
                x, y, right, bottom = self.manifest['features'][name]['box']
                expected = (shears.x+((x+right)/2-source[0])*scale,
                            shears.y+((y+bottom)/2-source[1])*scale)
                choices = (*matches[name], *matches.get('arrow_disabled', ())) if name == 'arrow' else matches[name]
                found = next((m for m in choices
                    if np.linalg.norm(np.subtract(m.center, expected)) <= 14*scale
                    and .88 <= m.width/((right-x)*scale) <= 1.12), None)
                if found is None:
                    break
                evidence.append(found)
            if len(evidence) == 2:
                tx, ty = self.manifest['tool_point']
                tool = VisualTarget(round(shears.x+(tx-source[0])*scale)-6,
                                    round(shears.y+(ty-source[1])*scale)-6, 12, 12, shears.score)
                guides.append((shears, tool, evidence[0], scale))
        if len(guides) != 1 or cancel():
            return None
        shears, tool, arrow, scale = guides[0]
        # World animals have their own zoom. Search the whole frame; neither
        # the farm's layout nor the tool menu predicts the pasture's position.
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        animals = (() if feeding else self.matcher._search(
            image, self.references['ready'], sizes, .91, cancel, max_peaks=8))
        troughs = self.matcher._search(image, self.references['pen_trough'], sizes, .91,
                                       cancel, max_peaks=6)
        supported = []
        for trough in troughs:
            polygon = self.pen(image, trough)
            if not polygon and pen_hint is not None:
                old = _decode(pen_hint[0])
                previous = pen_hint[1]
                if old.shape == image.shape and len(previous) == 4:
                    old_troughs = self.matcher._search(old, self.references['pen_trough'],
                        sizes, .91, cancel, max_peaks=6)
                    stable = any(np.linalg.norm(np.subtract(m.center, trough.center)) <= 5*base
                                 and abs(m.width-trough.width) <= 3*base for m in old_troughs)
                    mask = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillConvexPoly(mask, np.array(previous, np.int32), 255)
                    pixels = np.max(np.abs(image.astype(np.int16)-old.astype(np.int16)), axis=2)[mask > 0]
                    if (stable and len(pixels) and np.mean(pixels < 30) >= .55
                            and cv2.pointPolygonTest(np.array(previous, np.int32),
                                                    trough.center, True) >= -trough.width*.25):
                        # A timer may cover the lower fence. A just-observed
                        # outline remains usable only while the world trough
                        # and most visible floor pixels independently agree.
                        polygon = previous
            if not polygon:
                continue
            contour = np.array(polygon, np.int32)
            center = np.mean(polygon, axis=0)
            # The location link's active tool menu must point into this pen.
            # A matching animal elsewhere cannot redirect a harvest.
            if not (tool.center[0] < center[0] < tool.center[0]+650*scale
                    and arrow.center[1] < center[1] < arrow.center[1]+420*scale):
                continue
            if feeding:
                # Feeding sweeps the enclosure even when sheep turn or overlap.
                # The wool identity, sheep feed guide, wooden trough, and white
                # fence already establish its species; individual poses do not.
                supported.append((trough, polygon))
            else:
                supported.extend((animal, polygon) for animal in animals
                                 if cv2.pointPolygonTest(contour, animal.center, True) >= -animal.width*.35
                                 and .65 <= trough.width/animal.width <= 1.65)
        animals = tuple(animal for animal, _ in supported)
        if cancel():
            return None
        target = min(animals, key=lambda m: np.linalg.norm(np.subtract(m.center, tool.center)),
                     default=None)
        polygon = next((p for a, p in supported if a is target), ())
        path = self.sweep(polygon, target.width*(.55 if feeding else 1)) if polygon else ()
        if not path:
            target = None
        stock = None
        if feeding:
            from hayday.quantities import read_count
            x, y, width, height = self.manifest['feed_count_box']
            stock = read_count(png, (round(shears.x+(x-source[0])*scale),
                                    round(shears.y+(y-source[1])*scale),
                                    round(width*scale), round(height*scale)))
            tx, ty = self.manifest['feed_point']
            tool = VisualTarget(round(shears.x+(tx-source[0])*scale)-6,
                                round(shears.y+(ty-source[1])*scale)-6, 12, 12, shears.score)
        return FruitTarget(tool, target, min(shears.score, arrow.score), scale, arrow,
                           'wool', animals, path, polygon, 'feed' if feeding else 'harvest', stock)

    def enclosure(self, png, desired_icon, cancel):
        """Find one production-sheep trough in a clear farm view to reopen its menu."""
        if not self.matcher.find_item(desired_icon, self.item, min_scale=.45,
                                      max_scale=2.2, threshold=.93, cancel=cancel):
            return None
        image = _decode(png)
        base = image.shape[0]/self.manifest['reference_height']
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        troughs = self.matcher._search(image, self.references['pen_trough'], sizes, .91,
                                      cancel, max_peaks=6)
        found = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clear_soil = cv2.inRange(hsv, (18, 100, 150), (30, 245, 245))
        for trough in troughs:
            polygon = self.pen(image, trough)
            if not polygon:
                continue
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.array(polygon, np.int32), 255)
            margin = max(3, round(trough.width*.2))
            mask = cv2.erode(mask, np.ones((margin*2+1, margin*2+1), np.uint8))
            mask = cv2.bitwise_and(mask, clear_soil)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, radius, _, point = cv2.minMaxLoc(distance)
            if radius >= max(5, trough.width*.08):
                # Trough artwork projects into the pen behind it. Tap clear
                # soil inside the actual enclosure to avoid its neighbor.
                found.append((VisualTarget(point[0]-5, point[1]-5, 10, 10, trough.score), polygon))
        return found[0] if len(found) == 1 and not cancel() else None

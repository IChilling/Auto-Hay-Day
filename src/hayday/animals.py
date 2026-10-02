"""Recognize enabled shears and fenced production pens independently of animal poses."""

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
    def pen(image, animal, *, slope_bounds=(.25, .8), vertices=(4,), close_radius=0, fence='white'):
        """Find fenced soil around an independently recognized animal or trough."""
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
            # The four-corner approximation can cut inside the actual soil
            # hull as animals move along its edge. Count only this component's
            # pixels inside the diamond; outside soil cannot inflate occupancy
            # above 100 percent and make a valid fenced pen disappear.
            interior = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(interior, points-np.array([x, y]), 1)
            occupancy = np.count_nonzero((labels[y:y+height, x:x+width] == index)
                                         & (interior > 0))/max(1, np.count_nonzero(interior))
            if not (np.all((slopes > slope_bounds[0]) & (slopes < slope_bounds[1]))
                    and occupancy > .50):
                continue
            # Soil alone is insufficient: the requested fence type must also
            # support the observed diamond's perimeter.
            ring = np.zeros(image.shape[:2], np.uint8)
            cv2.polylines(ring, [polygon], True, 255, max(3, round(animal.width*.45)))
            pixels = image[ring > 0]
            if fence == 'wood':
                hsv_ring = hsv[ring > 0]
                supported = ((hsv_ring[:, 0] < 19) & (hsv_ring[:, 1] > 105)
                             & (hsv_ring[:, 2] > 65)).mean() >= .20
            elif fence == 'white':
                white = (pixels.min(axis=1) > 175) & (np.ptp(pixels, axis=1) < 65)
                supported = white.mean() >= .045
            else:
                supported = False
            if supported:
                candidates.append(tuple(tuple(map(int, p)) for p in points))
        return candidates[0] if len(candidates) == 1 else ()

    @staticmethod
    def fenced_pen(image, trough):
        """Read the complete white fence even when wool covers its soil corners."""
        white = ((image.min(axis=2) > 175) & (np.ptp(image, axis=2) < 65)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        soil = cv2.inRange(hsv, (12, 85, 75), (32, 255, 245))
        candidates = []
        for label, (x, y, width, height, area) in enumerate(stats[1:count], 1):
            if (not 1.5 < width/max(1, height) < 2.6
                    or not 3 < width/trough.width < 7
                    or area < trough.width**2
                    or not x < trough.center[0] < x+width
                    or not y < trough.center[1] < y+height):
                continue
            ys, xs = np.nonzero(labels == label)
            hull = cv2.convexHull(np.column_stack((xs, ys)))
            polygon = cv2.approxPolyDP(hull, .045*cv2.arcLength(hull, True), True)
            if len(polygon) != 4 or cv2.pointPolygonTest(polygon, trough.center, False) <= 0:
                continue
            points = polygon[:, 0]
            edges = np.roll(points, -1, axis=0)-points
            slopes = np.abs(edges[:, 1]/np.maximum(1, np.abs(edges[:, 0])))
            if not np.all((slopes > .25) & (slopes < .8)):
                continue
            interior = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(interior, polygon, 255)
            if (soil[interior > 0] > 0).mean() < .25:
                continue
            # All four sides need white fence pixels, rather than a large
            # white animal or a neighboring building supplying the hull.
            supported = True
            for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
                edge = np.zeros_like(interior)
                cv2.line(edge, tuple(first), tuple(second), 255, max(3, round(trough.width*.25)))
                if white[edge > 0].mean() < .15:
                    supported = False
                    break
            if supported:
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
            if not feeding and not self._shears_enabled(image, shears):
                continue
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
                guides.append((shears, tool, evidence[0], scale, evidence[1]))
        if len(guides) != 1 or cancel():
            return None
        shears, tool, arrow, scale, feed = guides[0]
        # World pens have their own zoom. Search the whole frame; neither
        # the farm's layout nor the tool menu predicts the pasture's position.
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        troughs = self.matcher._search(image, self.references['pen_trough'], sizes, .91,
                                       cancel, max_peaks=6)
        supported = []
        for trough in troughs:
            polygon = self.fenced_pen(image, trough) or self.pen(image, trough)
            if pen_hint is not None:
                old = _decode(pen_hint[0])
                previous = pen_hint[1]
                if old.shape == image.shape and len(previous) == 4:
                    old_troughs = self.matcher._search(old, self.references['pen_trough'],
                        sizes, .91, cancel, max_peaks=6)
                    stable = any(np.linalg.norm(np.subtract(m.center, trough.center)) <= 5*base
                                 and abs(m.width-trough.width) <= 3*base for m in old_troughs)
                    mask = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillConvexPoly(mask, np.array(previous, np.int32), 255)
                    from hayday.farming_vision import FarmingVision
                    timer = FarmingVision(cancel=cancel).growing(png)
                    original_area = np.count_nonzero(mask)
                    if timer:
                        x, y, w, h = timer.box
                        mask[max(0, y-round(50*base)):min(image.shape[0], y+h+round(30*base)),
                             max(0, x-round(20*base)):min(image.shape[1], x+w+round(250*base))] = 0
                    pixels = np.max(np.abs(image.astype(np.int16)-old.astype(np.int16)), axis=2)[mask > 0]
                    if (stable and len(pixels) >= original_area*.25 and np.mean(pixels < 30) >= .55
                            and cv2.pointPolygonTest(np.array(previous, np.int32),
                                                    trough.center, True) >= -trough.width*.25):
                        # A timer may cover the lower fence. A just-observed
                        # outline remains usable only while the world trough
                        # and most visible floor pixels independently agree.
                        polygon = previous
            if not polygon:
                continue
            center = np.mean(polygon, axis=0)
            # The location link's active tool menu must point into this pen.
            # A matching animal elsewhere cannot redirect a harvest.
            if not (tool.center[0] < center[0] < tool.center[0]+650*scale
                    and arrow.center[1] < center[1] < arrow.center[1]+420*scale):
                continue
            # The wool identity, feed guide, wooden trough and white fence
            # identify production sheep. Enabled blue shears establish readiness;
            # costumes, movement and overlapping animals do not change that.
            supported.append((trough, polygon))
        animals = tuple(animal for animal, _ in supported)
        if cancel():
            return None
        target = min(animals, key=lambda m: np.linalg.norm(np.subtract(m.center, tool.center)),
                     default=None)
        polygon = next((p for a, p in supported if a is target), ())
        path = self.sweep(polygon, target.width*.55) if polygon else ()
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
        enabled = True
        if feeding:
            patch = image[feed.y:feed.y+feed.height, feed.x:feed.x+feed.width]
            mask = cv2.resize(self.references['feed'][:, :, 3], (feed.width, feed.height)) > 200
            saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
            enabled = bool(np.mean(saturation[mask] > 60) > .18)
        guide = FruitTarget(tool, target, min(shears.score, arrow.score), scale, arrow,
                            'wool', animals, path, polygon, 'feed' if feeding else 'harvest', stock,
                            enabled=enabled)
        if not feeding and target is not None:
            from hayday.animal_groups import grouped_harvest
            pens = list(self.enclosures(png, cancel))
            if pen_hint is not None and polygon == pen_hint[1]:
                from hayday.cows import CowVision
                pens = [p for p in pens if not CowVision.same_pen(p[2], polygon)]
                pens.append((target, target, polygon))
            seeds = [pen for pen in pens if pen[2] == polygon]
            if len(seeds) == 1:
                return grouped_harvest(guide, pens, seeds[0], .55)
        return guide

    def _shears_enabled(self, image, shears):
        reference = self.references['shears']
        hsv = cv2.cvtColor(reference[:, :, :3], cv2.COLOR_BGR2HSV)
        handle = ((reference[:, :, 3] > 0) & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 120)
                  & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90)).astype(np.uint8)
        mask = cv2.resize(handle, (shears.width, shears.height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        patch = image[shears.y:shears.y+shears.height, shears.x:shears.x+shears.width]
        if patch.shape[:2] != mask.shape or np.count_nonzero(mask) < 30:
            return False
        pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[mask]
        return bool(((pixels[:, 0] >= 90) & (pixels[:, 0] <= 120)
                     & (pixels[:, 1] > 70) & (pixels[:, 2] > 70)).mean() >= .60)

    def enclosure(self, png, desired_icon, cancel):
        """Find one production-sheep trough in a clear farm view to reopen its menu."""
        if not self.matcher.find_item(desired_icon, self.item, min_scale=.45,
                                      max_scale=2.2, threshold=.93, cancel=cancel):
            return None
        found = self.enclosures(png, cancel)
        return (found[0][1], found[0][2]) if len(found) == 1 else None

    def enclosures(self, png, cancel):
        """All complete production-sheep floors, without inspecting animal poses."""
        image = _decode(png)
        base = image.shape[0]/self.manifest['reference_height']
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        troughs = self.matcher._search(image, self.references['pen_trough'], sizes, .91,
                                      cancel, max_peaks=6)
        found = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clear_soil = cv2.inRange(hsv, (18, 100, 150), (30, 245, 245))
        for trough in troughs:
            polygon = self.fenced_pen(image, trough) or self.pen(image, trough)
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
                found.append((trough, VisualTarget(point[0]-5, point[1]-5, 10, 10, trough.score), polygon))
        return tuple(found) if not cancel() else ()

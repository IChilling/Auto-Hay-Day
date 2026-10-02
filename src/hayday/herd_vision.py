"""Read animal feed controls and complete pens without identifying individuals."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from hayday.animals import SheepVision
from hayday.chickens import ChickenVision
from hayday.cows import CowVision
from hayday.quantities import read_count
from hayday.resource_vision import ResourceVision, VisualTarget, _decode, _png

ANIMALS = ('sheep', 'cow', 'chicken', 'pig', 'goat')


@dataclass(frozen=True)
class FeedMenu:
    animal: str
    tool: VisualTarget
    available: int | None
    enabled: bool
    icon: bytes = field(repr=False)
    count_box: tuple[int, int, int, int]
    collection_tool: VisualTarget | None = None
    collection_enabled: bool | None = None


class HerdVision:
    def __init__(self):
        self.cow = CowVision()
        self.sheep = SheepVision()
        self.chicken = ChickenVision()
        self.matcher = ResourceVision()
        source = Path(__file__).resolve().parents[2]/'images/animals/herd'
        root = source if source.is_dir() else Path(__file__).parent/'assets/animals/herd'
        self.spec = json.loads((root/'manifest.json').read_text('utf-8'))
        self.refs = {name: _decode((root/entry['file']).read_bytes(), True)
                     for name, entry in self.spec['features'].items()}

    def _controls(self, animal):
        if animal == 'sheep':
            vision = self.sheep
            box = vision.manifest['features']['feed']['box']
            tool_box = vision.manifest['features']['shears']['box']
            count = vision.manifest['feed_count_box']
            return vision.references['feed'], vision.references['shears'], (
                tool_box[0]-box[0], tool_box[1]-box[1]), (
                count[0]-box[0], count[1]-box[1], *count[2:])
        if animal == 'cow':
            vision = self.cow
            box = vision.manifest['features']['feed_inactive']['box']
            tool_box = vision.manifest['features']['bucket']['box']
            count = vision.manifest['feed_count_box']
            return vision.references['feed'], vision.references['bucket'], (
                tool_box[0]-box[0], tool_box[1]-box[1]), (
                count[0]-box[0], count[1]-box[1], *count[2:])
        if animal == 'chicken':
            vision = self.chicken
            # The basket position varies between pens; its presence plus a
            # complete coop is checked separately by the caller.
            return vision.references['feed'], vision.guide_refs['basket'], None, (-126, -51, 135, 80)
        feed = animal+'_feed_inactive' if animal == 'goat' else animal+'_feed'
        box = self.spec['features'][feed]['box']
        tool_box = self.spec['features'][animal+'_tool']['box']
        return self.refs[feed], self.refs[animal+'_tool'], (
            tool_box[0]-box[0], tool_box[1]-box[1]), tuple(self.spec['feed_count'][animal])

    @staticmethod
    def _grey(reference):
        grey = cv2.cvtColor(reference[:, :, :3], cv2.COLOR_BGR2GRAY)
        return np.dstack((cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), reference[:, :, 3]))

    def menu(self, png, animal, cancel):
        image = _decode(png)
        feed_ref, tool_ref, offset, count_box = self._controls(animal)
        grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        base = image.shape[0]/1080
        scales = np.unique(np.r_[np.linspace(.8, 1.2, 17)*base, base])
        feeds = self.matcher._search(grey, self._grey(feed_ref), scales, .935, cancel, max_peaks=3)
        tools = self.matcher._search(grey, self._grey(tool_ref), scales, .935, cancel, max_peaks=3)
        supported = []
        for feed in feeds:
            scale = feed.width/feed_ref.shape[1]
            if offset is None:
                valid = [tool for tool in tools if feed.x-450*scale < tool.x < feed.x-100*scale
                         and feed.y-50*scale < tool.y < feed.y+200*scale]
            else:
                expected = np.add((feed.x, feed.y), np.multiply(offset, scale))
                valid = [tool for tool in tools if np.linalg.norm(np.subtract((tool.x, tool.y), expected)) < 20*scale]
            if len(valid) != 1:
                continue
            patch = image[feed.y:feed.y+feed.height, feed.x:feed.x+feed.width]
            mask = cv2.resize(feed_ref[:, :, 3], (feed.width, feed.height)) > 200
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # Grey feed is unavailable even when its stock count is positive.
            enabled = bool(np.mean(hsv[:, :, 1][mask] > 60) > .18)
            x, y, w, h = count_box
            count = (round(feed.x+x*scale), round(feed.y+y*scale), round(w*scale), round(h*scale))
            available = read_count(png, count)
            icon = self.refs['goat_feed'] if animal == 'goat' else feed_ref
            tool = valid[0]
            if animal == 'cow':
                ready = bool(self.cow._bucket_enabled(image, tool))
            elif animal == 'sheep':
                ready = self.sheep._shears_enabled(image, tool)
            elif animal == 'chicken':
                ready = self.chicken._basket_enabled(image, tool)
            else:
                ready = None
            supported.append(FeedMenu(animal, feed, available, enabled, _png(icon), count, tool, ready))
        return supported[0] if len(supported) == 1 and not cancel() else None

    def pens(self, png, animal, cancel, *, complete=True):
        if animal in {'cow', 'chicken'}:
            result = getattr(self, animal).enclosures(png, cancel)
        else:
            image = _decode(png)
            base = image.shape[0]/1080
            scales = np.unique(np.r_[np.geomspace(.55, 2.3, 32)*base, base])
            references = [self.refs[animal+'_trough']]
            if animal == 'sheep':
                references.append(self.sheep.references['pen_trough'])
            troughs = CowVision._distinct([target for reference in references
                for target in self.matcher._search(image, reference, scales, .91, cancel, max_peaks=6)])
            result = []
            for trough in troughs:
                polygon = self.sheep.fenced_pen(image, trough) if animal == 'sheep' else ()
                polygon = polygon or SheepVision.pen(image, trough, fence='wood' if animal == 'pig' else 'white')
                if not polygon:
                    continue
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillConvexPoly(mask, np.array(polygon, np.int32), 255)
                soil = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (18, 100, 100), (32, 255, 245))
                distance = cv2.distanceTransform(mask & soil, cv2.DIST_L2, 5)
                _, radius, _, point = cv2.minMaxLoc(distance)
                if radius > 5 and not any(CowVision.same_pen(polygon, old[2]) for old in result):
                    result.append((trough, VisualTarget(point[0]-5, point[1]-5, 10, 10, trough.score), polygon))
        if not complete:
            return tuple(result)
        height, width = _decode(png).shape[:2]
        return tuple(pen for pen in result if min(x for x, y in pen[2]) > width*.12
                     and max(x for x, y in pen[2]) < width*.89
                     and min(y for x, y in pen[2]) > height*.18
                     and max(y for x, y in pen[2]) < height*.86)

    def confirm_pen(self, before, after, animal, pen, cancel):
        """Keep a clear outline only with camera, landmark and floor agreement."""
        from hayday.farming import FarmingWorker
        landmark, _, polygon = pen
        center = tuple(np.mean(polygon, axis=0))
        projected = FarmingWorker._translated_plot(before, after, center)
        if projected is None:
            return None
        delta = np.subtract(projected, center)
        points = np.rint(np.array(polygon)+delta).astype(np.int32)
        image, old = _decode(after.png), _decode(before.png)
        height, width = image.shape[:2]
        if (points[:, 0].min() < width*.12 or points[:, 0].max() > width*.89
                or points[:, 1].min() < height*.18 or points[:, 1].max() > height*.86):
            return None
        base = height/1080
        # A selected, already-fed animal exposes its growth timer across the
        # same pen. Compare only independently visible world pixels; the
        # timer and its title cannot count as changed ground or as evidence.
        from hayday.farming_vision import FarmingVision
        timer = FarmingVision(cancel=cancel).growing(after.png)
        visible = np.full((height, width), 255, np.uint8)
        if timer is not None:
            x, y, w, h = timer.box
            visible[y:min(height, y+h+round(25*base)),
                    max(0, x-round(10*base)):min(width, x+w+round(10*base))] = 0
            center = x+w/2
            visible[max(0, y-round(45*base)):y,
                    round(center-w*.11):round(center+w*.11)] = 0
        if animal == 'cow':
            matches = self.cow._search(image, 'trough', cancel, world=True)
        elif animal == 'chicken':
            matches = self.chicken.coops(image, cancel)
        else:
            references = [self.refs[animal+'_trough']]
            if animal == 'sheep':
                references.append(self.sheep.references['pen_trough'])
            matches = CowVision._distinct([target for reference in references
                for target in self.matcher._search(image, reference,
                    np.unique(np.r_[np.geomspace(.55, 2.3, 32)*base, base]), .91, cancel, max_peaks=6)])
        expected = np.add(landmark.center, delta)
        matches = [m for m in matches if np.linalg.norm(np.subtract(m.center, expected)) < 8*base
                   and abs(m.width-landmark.width) <= 3*base]
        if len(matches) != 1:
            # Menu illumination can change the canonical trough score. The
            # just-recognized landmark must still match its original pixels
            # at the independently registered location, within eight pixels.
            x, y, w, h = landmark.box
            reference = old[y:y+h, x:x+w]
            ex, ey = np.rint(np.add((x, y), delta)).astype(int)
            landmark_mask = visible[ey:ey+h, ex:ex+w]
            if landmark_mask.shape != (h, w) or np.mean(landmark_mask > 0) < .30:
                return None
            margin = max(1, round(8*base))
            x0, y0 = max(0, ex-margin), max(0, ey-margin)
            region = image[y0:min(height, ey+h+margin), x0:min(width, ex+w+margin)]
            if region.shape[0] < h or region.shape[1] < w:
                return None
            response = cv2.matchTemplate(region, reference, cv2.TM_CCOEFF_NORMED, mask=landmark_mask)
            response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
            if cv2.minMaxLoc(response)[1] < .95:
                return None
        aligned = cv2.warpAffine(old, np.float32([[1, 0, delta[0]], [0, 1, delta[1]]]), (width, height))
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        if timer:
            # Include the visible fence immediately around the soil outline.
            # This is visual evidence only; the sweep stays inside the soil.
            size = max(3, round(landmark.width*.4)) | 1
            mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
        area = np.count_nonzero(mask)
        mask &= visible
        if np.count_nonzero(mask) < area*.25:
            return None
        differences = np.max(np.abs(image.astype(np.int16)-aligned.astype(np.int16)), axis=2)[mask > 0]
        if not len(differences) or np.mean(differences < 35) < (.80 if timer else .55):
            return None
        return tuple(tuple(map(int, point)) for point in points)

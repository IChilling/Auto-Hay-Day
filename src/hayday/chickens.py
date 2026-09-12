"""Locate laying hens in their observed enclosure, using the egg resource guide."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hayday.animals import SheepVision
from hayday.resource_vision import ResourceVision, VisualTarget, _decode, _png


class ChickenVision:
    def __init__(self, reference_path=None):
        source = Path(__file__).resolve().parents[2]/'images/animals/chicken'
        self.root = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/animals/chicken')
        self.manifest = json.loads((self.root/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported chicken references.')
        self.item = (self.root/self.manifest['desired_item']).read_bytes()
        self.references = {name: _decode((self.root/spec['file']).read_bytes(), True)
                           for name, spec in self.manifest['features'].items()}
        self.matcher = ResourceVision()
        fruit = self.root.parents[1]/'fruit'
        self.guide_spec = json.loads((fruit/'manifest.json').read_text('utf-8'))
        self.guide_refs = {name: _decode((fruit/f'{name}.png').read_bytes(), True)
                           for name in ('basket', 'arrow')}

    def basket(self, png, icon, cancel):
        from hayday.fruit import FruitTarget, FruitVision

        if icon is None or not self.matcher.find_item(
                icon, self.item, min_scale=.45, max_scale=2.2, threshold=.93, cancel=cancel):
            return None
        image = _decode(png)
        base = image.shape[0]/1080
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*1.5, 23), base])
        baskets = self.matcher._search(image, self.guide_refs['basket'], sizes, .93, cancel, max_peaks=4)
        arrows = self.matcher._search(image, self.guide_refs['arrow'], sizes, .93, cancel, max_peaks=4)
        guides = []
        source = self.guide_spec['features']['basket']['box']
        for basket in baskets:
            scale = basket.width/self.guide_refs['basket'].shape[1]
            expected = FruitVision._relative(self.guide_spec['features']['arrow']['box'], basket, source, scale)
            matches = [a for a in arrows if np.linalg.norm(np.subtract(a.center, expected.center)) <= 15*scale
                       and .87 <= a.width/expected.width <= 1.13]
            if len(matches) != 1:
                continue
            tx, ty = self.guide_spec['tool_point']
            tool = FruitVision._relative((tx-6, ty-6, tx+6, ty+6), basket, source, scale)
            guide = FruitTarget(tool, None, min(basket.score, matches[0].score), scale, matches[0])
            # A chicken menu includes a paintbrush below the basket, changing
            # its translucent outline. Chicken feed replaces that unstable
            # outline as the third independent guide feature.
            found = self.harvest(png, icon, guide, cancel)
            if found is not None:
                guides.append(found)
        return guides[0] if len(guides) == 1 else None

    def harvest(self, png, icon, guide, cancel):
        if icon is None or not self.matcher.find_item(
                icon, self.item, min_scale=.45, max_scale=2.2, threshold=.93, cancel=cancel):
            return None
        image = _decode(png)
        base = image.shape[0]/self.manifest['reference_height']
        sizes = np.unique(np.r_[np.geomspace(base*.65, base*1.5, 19), base])
        feeds = [f for f, enabled in self._feeds(image, sizes, cancel)]
        # The chicken feed artwork independently distinguishes the egg basket
        # from the identical basket used for trees and other animal menus.
        feeds = [f for f in feeds if guide.tool.center[0]+50*guide.scale < f.center[0] < guide.tool.center[0]+350*guide.scale
                 and guide.tool.center[1]-250*guide.scale < f.center[1] < guide.tool.center[1]-20*guide.scale]
        if len(feeds) != 1:
            return None
        coops = self.coops(image, cancel)
        # This offset belongs to the selected-object UI, not the farm layout.
        # If the selected coop is hidden, do not switch the active guide to a
        # neighboring coop simply because that neighbor is still recognizable.
        expected_coop = np.add(feeds[0].center, (110*guide.scale, 245*guide.scale))
        nearby = [c for c in coops if np.linalg.norm(np.subtract(c.center, expected_coop)) < 80*guide.scale]
        selected = min(nearby, key=lambda c: np.linalg.norm(np.subtract(c.center, guide.tool.center)), default=None)
        if selected is None:
            return self._target(guide, [])
        supported = [(probe, polygon) for coop, probe, polygon in self.enclosures(png, cancel)
                     if selected is not None and np.linalg.norm(np.subtract(coop.center, selected.center)) < 10*guide.scale
                     and guide.tool.center[0] < np.mean(polygon, axis=0)[0] < guide.tool.center[0]+650*guide.scale
                     and guide.arrow.center[1] < np.mean(polygon, axis=0)[1] < guide.arrow.center[1]+420*guide.scale]
        if supported:
            observed = self._target(guide, supported)
            if self.ready_neighbors(image, observed.pen_polygon,
                                    selected.width/self.references['coop'].shape[1], cancel):
                return observed
            # Retain the selected enclosure so care can feed it and exclude it
            # while searching another pen; no harvest path is authorized.
            return replace(observed, target=None, sweep_path=(), fruit_features=())
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        hens = []
        for name, reference in self.references.items():
            if not name.startswith('ready'):
                continue
            for artwork in (reference, cv2.flip(reference, 1)):
                hens.extend(self.matcher._search(image, artwork, sizes, .93, cancel, max_peaks=12))
        supported = []
        for hen in hens:
            if selected is not None and not (
                    selected.x-100*guide.scale < hen.center[0] < selected.x+205*guide.scale
                    and selected.y+70*guide.scale < hen.center[1] < selected.y+215*guide.scale):
                continue
            # Gray/white livestock can resemble a masked hen in grayscale.
            # Its comb must also retain independently observed saturated red.
            patch = image[hen.y:hen.y+hen.height, hen.x:hen.x+hen.width]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 120)
            if np.count_nonzero(red[:max(1, hen.height//2)]) < max(6, hen.width*hen.height*.02):
                continue
            # The coop cuts off the back corner of its soil diamond, making
            # that visible edge steeper than the unobstructed sheep pasture.
            polygon = SheepVision.pen(image, hen, slope_bounds=(0, 1.3), vertices=(4, 5, 6))
            if not polygon:
                continue
            center = np.mean(polygon, axis=0)
            if (guide.tool.center[0] < center[0] < guide.tool.center[0]+650*guide.scale
                    and guide.arrow.center[1] < center[1] < guide.arrow.center[1]+420*guide.scale):
                supported.append((hen, polygon))
        return self._target(guide, supported)

    @staticmethod
    def _target(guide, supported):
        if not supported:
            return replace(guide, species='egg', target=None, fruit_features=())
        hen, polygon = min(supported, key=lambda entry: np.linalg.norm(np.subtract(
            np.mean(entry[1], axis=0), guide.tool.center)))
        return replace(guide, species='egg', target=hen,
                       fruit_features=tuple(h for h, p in supported if p == polygon),
                       pen_polygon=polygon, sweep_path=SheepVision.sweep(polygon, hen.width))

    def ready_neighbors(self, image, polygon, scale, cancel):
        """Require a laying-hen pose inside the pen before an egg sweep."""
        points = np.array(polygon, np.int32)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        top = max(0, top-round(45*scale))
        region = image[top:bottom+1, left:right+1]
        sizes = np.unique(np.r_[np.linspace(scale*.82, scale*1.18, 9), scale])
        for name in ('ready', 'ready_facing'):
            for reference in (self.references[name], cv2.flip(self.references[name], 1)):
                for target in self.matcher._search(region, reference, sizes, .93, cancel, max_peaks=5):
                    point = (target.center[0]+int(left), target.center[1]+int(top))
                    if cv2.pointPolygonTest(points, point, True) >= -target.width*.2:
                        return True
        return False

    def coops(self, image, cancel):
        base = image.shape[0]/self.manifest['reference_height']
        sizes = np.unique(np.r_[np.geomspace(base*.55, base*2.3, 32), base])
        return self.matcher._search(image, self.references['coop'], sizes, .94, cancel, max_peaks=12)

    def enclosures(self, png, cancel):
        """Return all visible coops with independently enclosed soil floors."""
        image = _decode(png)
        factor = image.shape[0]/self.manifest['reference_height']
        if abs(factor-1) > .01:
            normalized = cv2.resize(image, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
            def scaled(target):
                return VisualTarget(*(round(v*factor) for v in target.box), target.score)
            return tuple((scaled(coop), scaled(ground), tuple(tuple(round(v*factor) for v in point) for point in polygon))
                         for coop, ground, polygon in self.enclosures(_png(normalized), cancel))
        coops = self.coops(image, cancel)
        found = []
        for coop in coops:
            scale = coop.width/self.references['coop'].shape[1]
            px, py = self.manifest['pen_probe']['center']
            width = round(self.manifest['pen_probe']['width']*scale)
            probe = VisualTarget(round(coop.x+px*scale)-width//2,
                                 round(coop.y+py*scale)-width//2, width, width, coop.score)
            polygon = ()
            for radius in (0, max(1, round(scale)), max(1, round(2*scale))):
                candidate = SheepVision.pen(image, probe, slope_bounds=(-.01, 1.3),
                                            vertices=(4, 5, 6), close_radius=radius)
                # Closing gaps must not join the soil of neighboring pens.
                if candidate and np.ptp(np.array(candidate)[:, 0]) < coop.width*3.25:
                    polygon = candidate
                    break
            if not polygon:
                continue
            # Choose visible clear floor, avoiding the house, fence and hens.
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.array(polygon, np.int32), 255)
            soil = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (15, 100, 100), (32, 255, 245))
            mask &= soil
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, radius, _, point = cv2.minMaxLoc(distance)
            if radius >= max(4, width*.1):
                ground = VisualTarget(point[0]-width//2, point[1]-width//2, width, width, coop.score)
                found.append((coop, ground, polygon))
        return tuple(found)

    def _feeds(self, image, sizes, cancel):
        result = []
        for name in ('feed', 'feed_inactive'):
            for target in self.matcher._search(image, self.references[name], sizes, .94, cancel, max_peaks=4):
                if not any(np.linalg.norm(np.subtract(target.center, prior.center)) < 20 for prior, _ in result):
                    result.append((target, name == 'feed'))
        return result

    def feed(self, png, icon, cancel):
        from hayday.fruit import FruitTarget
        from hayday.quantities import read_count

        if icon is None or not self.matcher.find_item(icon, self.item, min_scale=.45,
                max_scale=2.2, threshold=.93, cancel=cancel):
            return None
        image = _decode(png)
        base = image.shape[0]/1080
        sizes = np.unique(np.r_[np.geomspace(base*.65, base*1.5, 19), base])
        feeds = self._feeds(image, sizes, cancel)
        if len(feeds) != 1:
            return None
        feed, enabled = feeds[0]
        scale = feed.width/self.references['feed'].shape[1]
        stock = read_count(png, (round(feed.x-126*scale), round(feed.y-51*scale),
                                 round(135*scale), round(80*scale)))
        expected_coop = np.add(feed.center, (110*scale, 245*scale))
        supported = [(ground, poly) for coop, ground, poly in self.enclosures(png, cancel)
                     if np.linalg.norm(np.subtract(coop.center, expected_coop)) < 80*scale
                     and feed.center[0]-80*scale < np.mean(poly, axis=0)[0] < feed.center[0]+480*scale
                     and feed.center[1]+150*scale < np.mean(poly, axis=0)[1] < feed.center[1]+600*scale]
        guide = FruitTarget(feed, None, feed.score, scale, feed, 'egg', action='feed', available=stock, enabled=enabled)
        return self._target(guide, supported)

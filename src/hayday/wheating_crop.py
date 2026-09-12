"""Wheat sweeps extend the existing selected-plot controls without changing orders."""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker
from hayday.farming_vision import FarmingVision, HarvestTarget
from hayday.resource_vision import VisualTarget, _decode
from hayday.resources import ResourceChanged, ResourceResult
from hayday.wheating_motion import WheatMotion
from hayday.wheating_numbers import read_number


class WheatFarmingVision(FarmingVision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control_locations = {}
        root = Path(__file__).parent/'assets/wheating'
        self._outline_reference = _decode((root/'empty_outline.png').read_bytes(), True)
        self._soil_texture = _decode((root/'soil_texture.png').read_bytes(), True)

    def _hsv(self, png):
        image = self._frame(png)
        if 'hsv' not in self._cache:
            self._cache['hsv'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return self._cache['hsv']

    def _possible_seed_button(self, image):
        # A cheap thumbnail rejection avoids 23 full-image scale searches on
        # every clear farm frame. Passing this check never authorizes an action.
        ratio = min(1., 480/image.shape[1])
        small = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        base = image.shape[0]/1080
        for scale in np.r_[base, np.geomspace(base*.55, base*1.5, 11)]:
            if self.cancel():
                return False
            color, mask = self._matcher._scaled(self.references['page_previous'], float(scale)*ratio)
            h, w = mask.shape
            if h > gray.shape[0] or w > gray.shape[1] or np.count_nonzero(mask) < 12:
                continue
            response = cv2.matchTemplate(gray, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
                                         cv2.TM_CCOEFF_NORMED, mask=mask)
            response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
            _, correlation, _, (x, y) = cv2.minMaxLoc(response)
            pixels = small[y:y+h, x:x+w][mask > 0]
            difference = np.abs(pixels.astype(np.float32)-color[mask > 0]).mean()
            if .8*correlation+.2*max(0., 1.-difference/100) >= .74:
                return True
        return False

    def _seed_controls(self, png):
        self._frame(png)
        if 'seed_controls' not in self._cache:
            self._cache['seed_controls'] = (super()._seed_controls(png)
                if self._matches(png, 'page_previous', .92) else None)
        return self._cache['seed_controls']

    def _matches(self, png, name, threshold=.91):
        image = self._frame(png)
        key = name, threshold
        if key in self._cache:
            return self._cache[key]
        reference = self.references[name]
        local = []
        for old in self._control_locations.get(name, ()):
            margin = max(40, round(old.width*.4))
            x, y = max(0, old.x-margin), max(0, old.y-margin)
            right, bottom = min(image.shape[1], old.x+old.width+margin), min(image.shape[0], old.y+old.height+margin)
            scale = old.width/reference.shape[1]
            hits = self._matcher._search(image[y:bottom, x:right], reference,
                np.array([.97, 1., 1.03])*scale, threshold, self.cancel, max_peaks=4)
            for hit in hits:
                target = replace(hit, x=hit.x+x, y=hit.y+y)
                if not any(np.linalg.norm(np.subtract(target.center, prior.center)) < target.width*.4 for prior in local):
                    local.append(target)
        if not local:
            if name == 'page_previous' and not self._possible_seed_button(image):
                self._cache[key] = ()
                return ()
            # Farming controls normally retain their UI scale through farm
            # zooming. Try that scale first; unfamiliar layouts retain the
            # existing broad search as a fallback.
            local = list(self._matcher._search(image, reference, np.array([image.shape[0]/1080]),
                min(.99, threshold+.02), self.cancel, max_peaks=4))
        if local:
            result = tuple(sorted(local, key=lambda target: -target.score))
            self._cache[key] = result
        else:
            result = super()._matches(png, name, threshold)
        if result:
            self._control_locations[name] = result
        return result

    def harvest(self, png):
        self._frame(png)
        if 'wheat_harvest' not in self._cache:
            self._cache['wheat_harvest'] = self._harvest(png)
        return self._cache['wheat_harvest']

    def _harvest(self, png):
        observed = super().harvest(png)
        if observed:
            return self._wheat_target(png, observed)
        # Wheat's short, moving tips do not always form the tall closed outline
        # used by the general crop worker. Require both tool features plus white
        # tips immediately adjacent to yellow foliage in their shared UI region.
        image = self._frame(png)
        hsv = self._hsv(png)
        geometry = self.manifest['harvest_geometry']
        source = geometry['sickle_box']
        found = []
        for sickle in self._matches(png, 'sickle', .90):
            scale = sickle.width/self.references['sickle'].shape[1]
            expected = self._relative(self.manifest['features']['guide_arrow']['box'], sickle, scale, source)
            arrow = next((a for a in self._matches(png, 'guide_arrow', .90)
                          if np.linalg.norm(np.subtract(a.center, expected.center)) < 12*scale), None)
            if arrow is None:
                continue
            roi = self._relative(geometry['white_highlight_box'], sickle, scale, source)
            left, top = max(0, roi.x), max(0, roi.y)
            right, bottom = min(image.shape[1], roi.x+roi.width+round(55*scale)), min(image.shape[0], roi.y+roi.height+round(30*scale))
            patch = hsv[top:bottom, left:right]
            white = cv2.inRange(patch, (0, 0, 210), (179, 50, 255))
            yellow = cv2.inRange(patch, (18, 145, 150), (36, 255, 255))
            tips = white & cv2.dilate(yellow, np.ones((7, 7), np.uint8))
            ys, xs = np.nonzero(tips)
            if len(xs) < 45*scale**2 or (yellow > 0).mean() < .12:
                continue
            x, y, w, h = cv2.boundingRect(np.column_stack((xs, ys)).astype(np.int32))
            if not (35*scale <= w <= 170*scale and 8*scale <= h <= 155*scale):
                continue
            highlight = VisualTarget(left+x, top+y, w, h, min(sickle.score, arrow.score))
            tx, ty = geometry['tool_point']
            tool = self._relative((tx-6, ty-6, tx+6, ty+6), sickle, scale, source)
            cx, cy = highlight.center
            destination = VisualTarget(cx-6, cy+round(8*scale), 12, 12, highlight.score)
            found.append(HarvestTarget(tool, destination, highlight.score, scale, destination, highlight))
        return self._wheat_target(png, found[0]) if len(found) == 1 else None

    def _wheat_target(self, png, harvest):
        highlight = harvest.highlight
        if highlight is None:
            return None
        hsv = self._hsv(png)
        x, y, w, h = highlight.box
        bottom = min(len(hsv), y+h+round(25*harvest.scale))
        yellow = cv2.inRange(hsv[y:bottom, x:x+w], (18, 155, 150), (36, 255, 255))
        yellow = cv2.erode(yellow, np.ones((3, 3), np.uint8))
        yy, xx = np.nonzero(yellow)
        if len(xx) < 80*harvest.scale**2:
            return None
        # Generic geometry for tall crops can end below wheat's entire tile.
        # End inside observed foliage, including when the white outline wraps
        # both the short stalks and the ground diamond.
        preferred = (w/2, max(25*harvest.scale, h*.6))
        nearest = int(np.argmin((xx-preferred[0])**2+(yy-preferred[1])**2))
        target = VisualTarget(int(x+xx[nearest])-8, int(y+yy[nearest])-8, 16, 16, harvest.score)
        return replace(harvest, target=target, drag_target=target)

    def empty_plot(self, png):
        self._frame(png)
        if 'wheat_empty_plot' not in self._cache:
            plot = self._empty_plot(png)
            self._cache['wheat_empty_plot'] = plot, self.full_outline
        plot, self.full_outline = self._cache['wheat_empty_plot']
        return plot

    def _empty_plot(self, png):
        self.full_outline = False
        if not self.seed_menu(png):
            return None
        image = self._frame(png)
        hsv = self._hsv(png)
        reference = self._outline_reference
        base = image.shape[0]/1080
        white = cv2.inRange(hsv, (0, 0, 205), (179, 55, 255))
        reference_white = cv2.inRange(cv2.cvtColor(reference[:, :, :3], cv2.COLOR_BGR2HSV), (0, 0, 205), (179, 55, 255))
        shape_matches = []
        for scale in (base,):
            shape = cv2.resize(reference_white, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            response = cv2.matchTemplate(white, shape, cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                if score < .70:
                    break
                plot = VisualTarget(x+round(6*scale), y+round(6*scale), round(117*scale), round(65*scale), score)
                pixels = self.tile_pixels(hsv, plot.center, plot.width/2, plot.height/2)
                if pixels is not None and ((pixels[:, 0] >= 8) & (pixels[:, 0] <= 16) & (pixels[:, 1] >= 30)).mean() > .7:
                    shape_matches.append(plot)
                response[max(0, y-30):y+31, max(0, x-60):x+61] = -1
        if len(shape_matches) == 1:
            self.full_outline = True
            return shape_matches[0]
        outlined = self._matcher._search(image, reference, np.array([base]), .89, self.cancel)
        if not outlined:
            outlined = self._matcher._search(image, reference, np.geomspace(base*.4, base*1.4, 22), .89, self.cancel)
        full = []
        for match in outlined:
            scale = match.width/reference.shape[1]
            plot = VisualTarget(match.x+round(6*scale), match.y+round(6*scale), round(117*scale), round(65*scale), match.score)
            pixels = self.tile_pixels(hsv, plot.center, plot.width/2, plot.height/2)
            if pixels is not None:
                soil = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 26) & (pixels[:, 1] >= 35)
                if soil.mean() > .75:
                    full.append(plot)
        if len(full) == 1:
            self.full_outline = True
            return full[0]
        white = cv2.inRange(hsv, (0, 0, 205), (179, 55, 255))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base = image.shape[0]/1080
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (45*base < w < 230*base and 1.45 < w/h < 2.3
                    and image.shape[0]*.2 < y < image.shape[0]*.83
                    and image.shape[1]*.18 < x < image.shape[1]*.83):
                continue
            hull = cv2.convexHull(contour)
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [hull-(x, y)], -1, 255, cv2.FILLED)
            inner = cv2.erode(filled, np.ones((5, 5), np.uint8)) > 0
            soil = cv2.inRange(hsv[y:y+h, x:x+w], (8, 65, 45), (26, 245, 245)) > 0
            if inner.sum() > 100*base**2 and .40 < cv2.contourArea(hull)/(w*h) < .7 and float(soil[inner].mean()) > .80:
                perimeter = np.zeros((h, w), np.uint8)
                cv2.drawContours(perimeter, [hull-(x, y)], -1, 255, 1)
                supported = cv2.dilate(white[y:y+h, x:x+w], np.ones((5, 5), np.uint8)) > 0
                complete = float(supported[perimeter > 0].mean()) >= .92
                found.append((VisualTarget(x, y, w, h, float(soil[inner].mean())), complete))
        if len(found) == 1:
            self.full_outline = found[0][1]
            return found[0][0]
        return super().empty_plot(png)

    @staticmethod
    def tile_pixels(image, point, dx, dy):
        cx, cy = map(int, point)
        dx, dy = max(3, round(dx*.65)), max(3, round(dy*.65))
        if cx-dx < 0 or cy-dy < 0 or cx+dx >= image.shape[1] or cy+dy >= image.shape[0]:
            return None
        yy, xx = np.indices((2*dy+1, 2*dx+1))
        mask = abs(xx-dx)/dx+abs(yy-dy)/dy < 1
        return image[cy-dy:cy+dy+1, cx-dx:cx+dx+1][mask]

    def empty_tiles(self, frame, plot, limit):
        """Follow the selected tile's observed lattice through contiguous soil."""
        if not getattr(self, 'full_outline', False):
            return []
        hsv = self._hsv(frame.png)
        # White selection art extends beyond the plot itself. Derive grid pitch
        # from repeated furrow textures instead of treating that border as soil.
        texture = self._soil_texture
        scale = plot.width/117
        matches = self._matcher._search(self._frame(frame.png), texture, np.array([scale]), .91, self.cancel, 16)
        offsets = []
        for first in matches:
            for second in matches:
                x, y = abs(first.center[0]-second.center[0]), abs(first.center[1]-second.center[1])
                if plot.width*.4 < x < plot.width*.51 and plot.height*.34 < y < plot.height*.52:
                    offsets.append((x, y))
        dx, dy = np.median(offsets, axis=0) if len(offsets) >= 4 else ((plot.width-10)/2, (plot.height-10)/2)
        if min(dx, dy) < 6:
            return [plot.center]
        seen, queue, points = set(), deque([(0, 0)]), []
        while queue and len(points) < min(limit, 98):
            i, j = queue.popleft()
            if (i, j) in seen:
                continue
            seen.add((i, j))
            center = (round(plot.center[0]+(i-j)*dx), round(plot.center[1]+(i+j)*dy))
            if not (frame.width*.12 < center[0] < frame.width*.87 and frame.height*.2 < center[1] < frame.height*.84):
                continue
            pixels = self.tile_pixels(hsv, center, dx, dy)
            if pixels is None:
                continue
            soil = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 16) & (pixels[:, 1] >= 65) & (pixels[:, 2] < 245)
            if float(soil.mean()) < .93:
                continue
            points.append(center)
            queue.extend(((i+1, j), (i, j+1), (i-1, j), (i, j-1)))
        return points


class WheatCropWorker(FarmingWorker):
    def __init__(self, *args, fast_capture=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_capture = fast_capture
        self._replant_grid = None
        self.planted_points = []
        self.plant_before = None
        self.harvest_plan = None
        self.harvest_plot_centers = False
        self.soil_frame = None
        self.remaining_seed_stock = None
        self.field_size = 0
        self._wheat_seed_location = None
        self.growth_ready_at = None
        self.growth_timer = None
        self.resumed_mature = False
        self._prepared_harvest = None
        self._prepared_seed_picker = None
        self._motion = WheatMotion()
        # Set by WheatFields for the selected whole-field route.  The
        # defaults keep direct worker callers on the established behavior.
        self.harvest_route_mode = None
        self.harvest_observed_points = 0
        self.harvest_expected_points = 0
        self.harvest_sample_spacing = None
        self.harvest_waypoint_hold = None
        self._harvest_shift = None

    def _bind_size(self, frame):
        size = frame.width, frame.height
        if self._size is None:
            self._size = size
        elif self._size != size:
            raise ResourceChanged('Device resolution changed during farming.')

    @property
    def growth_duration(self):
        from hayday.wheating_growth import WheatGrowthTimer
        return self.growth_timer.duration if self.growth_timer else WheatGrowthTimer.FALLBACK_SECONDS

    def _quick_frame(self):
        if self.fast_capture is None:
            return self._frame()
        self._check()
        frame = self.fast_capture()
        self._check()
        if self._size != (frame.width, frame.height):
            raise ResourceChanged('Device resolution changed during farming.')
        return frame

    @staticmethod
    def _translated_plot(before, after, point, *, require_visible=True):
        # A picker and animated buildings can consume the generic matcher's
        # feature budget and tilt its affine fit by several pixels. Use broad
        # farm landmarks and verify the displacement as a pure translation.
        first = cv2.imdecode(np.frombuffer(before.png, np.uint8), 0)
        second = cv2.imdecode(np.frombuffer(after.png, np.uint8), 0)
        if first is None or second is None or first.shape != second.shape:
            return None
        h, w = first.shape
        mask = np.zeros_like(first)
        mask[int(h*.16):int(h*.83), int(w*.16):int(w*.84)] = 255
        orb = cv2.ORB_create(nfeatures=4000)
        ka, a = orb.detectAndCompute(first, mask)
        kb, b = orb.detectAndCompute(second, mask)
        if a is None or b is None:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72*p[1].distance]
        if len(good) < 40:
            return None
        old = np.float32([ka[m.queryIdx].pt for m in good])
        new = np.float32([kb[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3)
        if (matrix is None or inliers is None or not np.isfinite(matrix).all()
                or inliers.sum() < 30):
            return None
        if abs(np.hypot(matrix[0, 0], matrix[1, 0])-1) > .025 or abs(matrix[1, 0]) > .02:
            return None
        offsets = new-old
        # The affine RANSAC subset can tilt around moving smoke/picker artwork
        # and omit valid landmarks. It only seeds the displacement estimate;
        # the stricter translation consensus below is the acceptance test.
        shift = np.median(offsets[inliers.ravel().astype(bool)], axis=0)
        supported = np.linalg.norm(offsets-shift, axis=1) <= 3
        if supported.sum() < max(30, len(good)*.60):
            return None
        tracked = old[supported]
        if np.ptp(tracked[:, 0]) < w*.25 or np.ptp(tracked[:, 1]) < h*.20:
            return None
        projected = np.asarray(point)+np.median(offsets[supported], axis=0)
        if require_visible and not (w*.13 < projected[0] < w*.88 and h*.15 < projected[1] < h*.88):
            return None
        return tuple(int(round(n)) for n in projected)

    def _close(self, frame):
        self._check()
        self._bind_size(frame)
        # An already clear farm needs no input. Repeated ground taps during a
        # camera return animation can accidentally select nearby buildings.
        if self.vision.growing(frame.png) is None and not self.vision.seed_menu(frame.png):
            return frame
        super()._close(frame)
        previous = None
        for _ in range(8):
            self._pause(.2)
            after = self._frame()
            view = CameraNavigator._view(after)
            if previous is not None and CameraNavigator._same_view(previous, view):
                return after
            previous = view
        raise RuntimeError('The camera did not settle after closing the wheat controls.')

    def _planned_harvest(self, frame, harvest):
        if self.harvest_plan is None:
            return self.harvest_path(frame, harvest) or [harvest.target.center]
        before, points = self.harvest_plan
        origin = points[0]
        moved = self._translated_plot(before, frame, origin, require_visible=False)
        if moved is None:
            return []
        shift = np.subtract(moved, origin)
        path = [tuple(map(int, np.add(point, shift))) for point in points]
        if any(not (frame.width*.1 < x < frame.width*.9 and frame.height*.15 < y < frame.height*.89) for x, y in path):
            return []
        return [harvest.target.center, *path]

    @staticmethod
    def _sweep_duration(points):
        distance = sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))
        return max(300, min(4000, round(distance/3)))

    def _residual_grid_points(self, frame):
        """Return known crop centers that still look ripe after the sweep.

        This is diagnostic only.  It samples the already verified grid in one
        HSV pass and never schedules another gesture, so normal cycles do not
        pay for a second field scan.
        """
        if not (self.harvest_plan and self.harvest_plot_centers):
            return []
        before, points = self.harvest_plan
        if not points:
            return []
        shift = self._harvest_shift
        if shift is None:
            moved = self._translated_plot(before, frame, points[0], require_visible=False)
            if moved is None:
                return []
            shift = np.subtract(moved, points[0])
        image = cv2.imdecode(np.frombuffer(frame.png, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        residual = []
        for point in points:
            x, y = map(int, np.add(point, shift))
            if not (12 <= x < frame.width-12 and 8 <= y < frame.height-8):
                continue
            patch = hsv[y-8:y+9, x-12:x+13]
            wheat = cv2.inRange(patch, (18, 140, 160), (31, 255, 255))
            if float((wheat > 0).mean()) >= .4:
                residual.append((x, y))
        return residual

    def _resume_picker(self, frame, entry):
        resumed = super()._resume_picker(frame, entry)
        if resumed:
            return resumed
        proof = entry.get('replant_evidence')
        if entry.get('stage') != 'harvested_needs_replant' or not isinstance(proof, dict):
            return None
        try:
            name = proof['file']
            if not isinstance(name, str) or '/' in name or '\\' in name:
                return None
            png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
            if hashlib.sha256(png).hexdigest() != proof['sha256']:
                return None
            saved = Screenshot(png, proof['width'], proof['height'], proof['captured_at'])
            x, y, width, height = proof['plot']
            expected = self._translated_plot(saved, frame, (x+width//2, y+height//2))
            current = self.vision.empty_plot(frame.png)
            if expected is None or current is None:
                return None
            # Neighboring wheat can grow over part of this saved outline while
            # stopped. A smaller outline at the same field can resume planting;
            # the tolerance is well below the distance to an adjacent tile.
            if (width*.6 <= current.width <= width*1.05
                    and np.linalg.norm(np.subtract(expected, current.center)) < min(width, current.width)*.28):
                return current
        except (KeyError, TypeError, ValueError, OSError, cv2.error):
            pass
        return None

    @staticmethod
    def _stock_box(frame, item, *, separate=False):
        # The bubble can visually join a cream building behind the transparent
        # picker. Its position relative to the matched wheat artwork remains
        # stable; verify cream coverage in that local bubble before reading it.
        left = round(item.x-item.width*.74)
        top = round(item.y-item.height*.07)
        width, height = round(item.width*.97), round(item.height*.58)
        if 0 <= left < left+width <= frame.width and 0 <= top < top+height <= frame.height:
            hsv = cv2.cvtColor(_decode(frame.png)[top:top+height, left:left+width], cv2.COLOR_BGR2HSV)
            cream = cv2.inRange(hsv, (0, 0, 205), (55, 70, 255))
            if (cream > 0).mean() > .58:
                return left, top, width, height
        return FarmingWorker._stock_box(frame, item, separate=separate)

    def _seed(self, frame, icon, plot, *, plot_center=None):
        self._check()
        center = plot_center or (plot.center if plot else None)
        if plot is None or center is None or not self.vision.seed_menu(frame.png):
            return None
        # The picker can straddle the selected tile after the camera centers it.
        # Match the isolated wheat artwork and its stock bubble, rather than
        # cutting the search off at the tile's center.
        wheat = (Path(__file__).parent/'assets/wheating/wheat_inventory.png').read_bytes()
        image = self.vision._frame(frame.png)
        reference = _decode(wheat, True)
        # Match the same tight alpha bounds as find_item. Previously the local
        # cache mixed padded and trimmed sizes, forcing a broad search again.
        yy, xx = np.nonzero(reference[:, :, 3] >= 200)
        reference = reference[yy.min():yy.max()+1, xx.min():xx.max()+1]
        region = (0, 0, min(frame.width, round(center[0]+frame.width*.12)), round(frame.height*.84))
        targets = []
        old = self._wheat_seed_location
        if old:
            x, y = max(0, old.x-60), max(0, old.y-60)
            right, bottom = min(region[2], old.x+old.width+60), min(region[3], old.y+old.height+60)
            if x < right and y < bottom:
                scale = old.width/reference.shape[1]
                targets = [replace(t, x=t.x+x, y=t.y+y) for t in self.vision._matcher._search(
                    image[y:bottom, x:right], reference, np.array([.97, 1., 1.03])*scale, .92,
                    self.cancel_event.is_set, 4)]
        if not targets:
            # Wheat's first-page artwork stays immediately above the selected
            # soil. Search this small area at native resolution before the
            # wide, downsampled fallback, which can lose the fine wheat edges.
            base = frame.height/1080
            x, y = max(0, round(center[0]-300*base)), max(0, round(center[1]-300*base))
            right, bottom = min(region[2], round(center[0]+80*base)), min(region[3], round(center[1]-65*base))
            if x < right and y < bottom:
                targets = [replace(t, x=t.x+x, y=t.y+y) for t in self.vision._matcher._search(
                    image[y:bottom, x:right], reference, np.array([1.23, 1.26, 1.29])*base, .94,
                    self.cancel_event.is_set, 4)]
        if not targets:
            targets = self.vision._matcher._search(image[:region[3], :region[2]], reference,
                np.array([1.26, 1.])*frame.height/1080, .94, self.cancel_event.is_set, 4)
        if not targets:
            targets = self.items.find_item(frame.png, wheat, min_scale=.5, max_scale=2.,
                region=region, threshold=.90, cancel=self.cancel_event.is_set)
        targets = [t for t in targets if self._stock_box(frame, t) is not None]
        self._check()
        if len(targets) == 1:
            self._wheat_seed_location = targets[0]
            return targets[0]
        return None

    @staticmethod
    def _exposed_soil(before, after, harvest):
        if harvest.highlight is None:
            return None
        point = WheatCropWorker._translated_plot(before, after, harvest.highlight.center)
        recovered_scale = point is None
        if point is None:
            # Restart/reconnect can change zoom. Only the recovery fallback
            # accepts a scale transform, still proven by distributed landmarks.
            from hayday.wheating_restart_camera import farm_transform
            matrix = farm_transform(before, after)
            if matrix is None:
                return None
            point = tuple(map(int, np.rint(matrix @ np.array([*harvest.highlight.center, 1]))))
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        else:
            shift = np.subtract(point, harvest.highlight.center)
            matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            scale = 1.
        original = cv2.cvtColor(_decode(before.png), cv2.COLOR_BGR2HSV)
        current = cv2.cvtColor(_decode(after.png), cv2.COLOR_BGR2HSV)
        wheat = cv2.inRange(original, (18, 155, 150), (36, 255, 255))
        wheat = cv2.warpAffine(wheat, matrix, (after.width, after.height))
        soil = cv2.inRange(current, (8, 65, 45), (18, 245, 245))
        changed = soil & wheat
        # Only foliage pixels from this selected wheat can establish exposed
        # soil. Disappearing translucent menus must not qualify brown buildings.
        roi = np.zeros_like(changed)
        radius = round(150*harvest.scale*scale)
        x, y = point
        roi[max(0, y-radius//3):min(after.height, y+radius), max(0, x-radius):min(after.width, x+radius)] = 255
        changed &= roi
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if recovered_scale:
            # A shifted stalk edge can expose a few brown pixels even while
            # wheat is still standing. Recovery requires a broad soil interior.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 17))
            changed &= cv2.erode(soil, kernel)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(changed)
        choices = []
        for i in range(1, count):
            if stats[i, 4] < 100*harvest.scale**2:
                continue
            ys, xs = np.nonzero((labels == i) & (soil > 0) & (wheat > 0))
            if not len(xs):
                continue
            cx, cy = centers[i]
            nearest = np.argmin((xs-cx)**2+(ys-cy)**2)
            choices.append((int(stats[i, 4]), (int(xs[nearest]), int(ys[nearest]))))
        if not choices:
            return None
        _, (x, y) = max(choices)
        return VisualTarget(x-8, y-8, 16, 16, 1.)

    @staticmethod
    def harvest_path(frame, harvest):
        """A continuous sweep stays inside the selected wheat's yellow foliage."""
        if harvest.highlight is None:
            return []
        hsv = cv2.cvtColor(_decode(frame.png), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 155, 150), (36, 255, 255))
        base = frame.height/1080
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(3, round(5*base)),)*2, np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        hx, hy = harvest.highlight.center
        choices = [(i, stat) for i, stat in enumerate(stats[1:count], 1)
                   if stat[4] > 500*base**2 and stat[0] <= hx < stat[0]+stat[2]
                   and stat[1] <= hy < stat[1]+stat[3]]
        if len(choices) != 1:
            return []
        label, (x, y, w, h, _) = choices[0]
        component = (labels == label).astype(np.uint8)
        component[:round(frame.height*.20)] = 0
        component[round(frame.height*.84):] = 0
        component[:, :round(frame.width*.12)] = 0
        component[:, round(frame.width*.88):] = 0
        interior = cv2.erode(component, np.ones((3, 3), np.uint8))
        allowed = cv2.dilate(component, np.ones((5, 5), np.uint8))
        step = max(12, round(22*base))
        candidates = []
        for row, yy in enumerate(range(int(y+step//2), int(y+h), step)):
            xs = list(range(int(x+step//2), int(x+w), step))
            if row % 2:
                xs.reverse()
            candidates.extend((xx, yy) for xx in xs if interior[yy, xx])
        start = harvest.target.center
        points = [start]
        while candidates and len(points) < 32:
            candidates.sort(key=lambda p: np.linalg.norm(np.subtract(points[-1], p)))
            selected = None
            for index, point in enumerate(candidates):
                distance = np.linalg.norm(np.subtract(points[-1], point))
                line = np.rint(np.linspace(points[-1], point, max(2, round(distance/3)))).astype(int)
                if allowed[line[:, 1], line[:, 0]].mean() >= .98:
                    selected = index
                    break
            if selected is None:
                break
            points.append(candidates.pop(selected))
        return points if len(points) > 3 else []

    def _saved_frame(self, proof):
        if not isinstance(proof, dict):
            return None
        name = proof.get('file')
        if not isinstance(name, str) or not name or '/' in name or '\\' in name:
            return None
        try:
            png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
            image = _decode(png)
            if (hashlib.sha256(png).hexdigest() != proof.get('sha256')
                    or image.shape[:2] != (proof.get('height'), proof.get('width'))
                    or (proof.get('width'), proof.get('height')) != self._size):
                return None
            return Screenshot(png, proof['width'], proof['height'], proof.get('captured_at', ''))
        except (OSError, ValueError, cv2.error):
            return None

    def _planted_state(self, before, after, points, plot):
        from hayday.game_scene import farm_scene_vision
        if not farm_scene_vision().ready(after.png) or CameraNavigator._modal_visible(after):
            return None
        moved = self._translated_plot(before, after, points[0])
        if moved is None:
            return None
        shift = np.subtract(moved, points[0])
        hsv = cv2.cvtColor(self.vision._frame(after.png), cv2.COLOR_BGR2HSV)
        mature = True
        for point in points:
            center = np.add(point, shift)
            if not (after.width*.12 < center[0] < after.width*.88
                    and after.height*.18 < center[1] < after.height*.84):
                return None
            pixels = self.vision.tile_pixels(hsv, center, (plot.width-10)/2, (plot.height-10)/2)
            if pixels is None:
                return None
            yellow = (pixels[:, 0] >= 18) & (pixels[:, 0] <= 31) & (pixels[:, 1] >= 140) & (pixels[:, 2] >= 160)
            ripe = float(yellow.mean()) >= .55
            if ripe:
                # Neighboring stalks can cover most of a missed tile's ground.
                # Its own crop body, just above the seed position, must exist.
                x, y = map(int, center-(0, round(16*after.height/1080)))
                body = hsv[y-8:y+9, x-12:x+13]
                ripe = float((cv2.inRange(body, (18, 140, 160), (31, 255, 255)) > 0).mean()) >= .4
            green = (pixels[:, 0] >= 28) & (pixels[:, 0] <= 90) & (pixels[:, 1] > 70) & (pixels[:, 2] > 70)
            soil = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 18) & (pixels[:, 1] >= 65) & (pixels[:, 2] < 245)
            if not ripe and not (float(green.mean()) >= .035 and float(soil.mean()) >= .45):
                return None
            mature = mature and ripe
        return 'mature' if mature else 'growing'

    def _resume_planted(self, frame, icon, key, entry):
        """Confirm an interrupted planting from current plants; never replay it."""
        before = self._saved_frame(entry.get('plant_before'))
        points = entry.get('points')
        if (before is None or not isinstance(points, list) or not 1 <= len(points) <= 98
                or any(not isinstance(p, list) or len(p) != 2 or any(type(n) is not int for n in p)
                       or not 0 <= p[0] < frame.width or not 0 <= p[1] < frame.height for p in points)
                or len({tuple(p) for p in points}) != len(points)):
            return ResourceResult('unsupported', 'The interrupted planting evidence is invalid; no gesture was replayed.')
        plot = self.vision.empty_plot(before.png)
        seed = self._seed(before, icon, plot) if plot else None
        if (seed is None or type(entry.get('available_before')) is not int
                or self._count(before, seed) != entry['available_before']
                or entry['available_before'] < len(points)):
            return ResourceResult('unsupported', 'The interrupted planting does not verify wheat seeds; no gesture was replayed.')
        previous, previous_state = None, None
        fresh = frame
        for observation in range(8):
            second = self._planted_state(before, fresh, points, plot)
            if (second is not None and second == previous_state
                    and previous is not None and fresh.captured_at
                    and fresh.captured_at != previous.captured_at):
                frame = previous
                break
            # Clouds and crop animation can briefly obscure a single tile.
            # Keep the same per-plot proof and require two consecutive clear
            # captures rather than restarting the game on one unclear frame.
            previous, previous_state = fresh, second
            if observation < 7:
                self._pause(.25)
                fresh = self._frame()
        else:
            return ResourceResult('unsupported',
                'Some attempted wheat plots are not visibly planted in two fresh matching field observations; no gesture was replayed.')
        self.plant_before = before
        self.planted_points = [tuple(p) for p in points]
        self.field_size = len(points)
        self.remaining_seed_stock = None  # Inventory may have changed while stopped.
        self.resumed_mature = second == 'mature'
        remaining = self.growth_duration
        try:
            remaining = max(0., min(remaining, datetime.fromisoformat(entry['updated_at']).timestamp()
                                   +self.growth_duration+30-time.time()))
        except (KeyError, TypeError, ValueError):
            pass
        self.growth_ready_at = time.monotonic()+(0 if self.resumed_mature else remaining)
        proof = [self._evidence(f, entry['operation'], f'plant_reconciled_{i}') for i, f in enumerate((frame, fresh))]
        self._record(key, 'growing', replanted=True, planted_count=len(points), plant_after=proof, recovered=True)
        return ResourceResult('waiting', f'Verified wheat on all {len(points)} previously attempted plots without repeating planting.')

    def _resume_replanted_field(self, frame, icon, key, entry):
        """Retire an old replant intent when the entire saved field is ripe again."""
        picker = self._saved_frame(entry.get('replant_evidence'))
        soil = self.soil_frame
        if picker is None or soil is None:
            return None
        plot = self.vision.empty_plot(picker.png)
        if plot is None or self._seed(picker, icon, plot) is None:
            return None
        from hayday.wheating_transition import saved_grid

        grid = saved_grid(self, entry)
        if grid:
            soil, points, _ = grid
        else:
            origin = self._translated_plot(picker, soil, plot.center, require_visible=False)
            if origin is None:
                return None
            soil_plot = replace(plot, x=origin[0]-plot.width//2, y=origin[1]-plot.height//2)
            points = self.vision.empty_tiles(soil, soil_plot, 98)
        if not points:
            return None
        previous = None
        for _ in range(8):
            if self._planted_state(soil, frame, points, plot) == 'mature':
                if previous and frame.captured_at and frame.captured_at != previous.captured_at:
                    break
                previous = frame
            else:
                previous = None
            # Smoke and drifting clouds can briefly wash out a planted tile.
            # Wait for two clear observations instead of weakening its proof.
            self._pause(.35)
            frame = self._frame()
        else:
            return None
        self.plant_before = soil
        self.planted_points = points
        self.field_size = len(points)
        self.remaining_seed_stock = None
        self.resumed_mature = True
        self.growth_ready_at = time.monotonic()
        proofs = [self._evidence(f, entry['operation'], f'replanted_field_{i}') for i, f in enumerate((previous, frame))]
        self._record(key, 'growing', replanted=True, externally_replanted=True,
                     plant_after=proofs, recovered=True)
        return ResourceResult('waiting', f'Verified ripe wheat on all {len(points)} saved plots; the old replant intent is complete.')

    def work_if_recognized(self, frame, icon, key):
        self._check()
        self._bind_size(frame)
        prepared, self._prepared_harvest = self._prepared_harvest, None
        entry = self.state['items'].get(key, {})
        if entry.get('stage') == 'plant_attempted' and entry.get('points'):
            return self._resume_planted(frame, icon, key, entry)
        if entry.get('stage') == 'harvested_needs_replant' and entry.get('replant_evidence'):
            # Restore the unobscured field, since the open picker hides tiles.
            # The superclass separately verifies the saved picker before input.
            proofs = entry.get('harvest_after', [])
            if not proofs or not isinstance(proofs[-1], dict):
                return ResourceResult('unsupported', 'The saved clear wheat field evidence is missing.')
            proof = proofs[-1]
            name = proof.get('file', '')
            if not isinstance(name, str) or not name or '/' in name or '\\' in name:
                return ResourceResult('unsupported', 'The saved clear wheat field evidence is invalid.')
            try:
                png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
                decoded = _decode(png)
                valid = (hashlib.sha256(png).hexdigest() == proof.get('sha256')
                         and decoded.shape[:2] == (proof.get('height'), proof.get('width'))
                         and (proof.get('width'), proof.get('height')) == self._size)
            except (OSError, ValueError, cv2.error):
                valid = False
            if not valid:
                return ResourceResult('unsupported', 'The saved clear wheat field evidence changed.')
            self.soil_frame = Screenshot(png, proof['width'], proof['height'], proof.get('captured_at', ''))
            reconciled = self._resume_replanted_field(frame, icon, key, entry)
            if reconciled:
                return reconciled
            if not self.vision.seed_menu(frame.png):
                from hayday.wheating_crop_resume import resume_bare_field
                resumed = resume_bare_field(self, frame, icon, key, entry)
                if resumed:
                    return resumed
        if entry.get('stage') in {'harvest_attempted', 'harvested_needs_replant'} and entry.get('harvest_before') and not entry.get('replant_evidence'):
            proof = entry['harvest_before']
            name = proof.get('file', '')
            if not name or '/' in name or '\\' in name:
                return ResourceResult('unsupported', 'The saved wheat harvest evidence is invalid.')
            png = (self.state_path.parent/'farming_evidence'/name).read_bytes()
            if hashlib.sha256(png).hexdigest() != proof.get('sha256'):
                return ResourceResult('unsupported', 'The saved wheat harvest evidence changed.')
            before = Screenshot(png, proof['width'], proof['height'], proof['captured_at'])
            harvest = self.vision.harvest(before.png)
            if harvest is None:
                return ResourceResult('unsupported', 'The earlier wheat harvest controls cannot be reconciled.')
            return self._replant_after_harvest(before, harvest, icon, key, entry['operation'])
        if self.state['items'].get(key, {}).get('stage') in self._PENDING:
            return super().work_if_recognized(frame, icon, key)
        harvest = self.vision.harvest(frame.png)
        if harvest is None:
            return super().work_if_recognized(frame, icon, key)
        from hayday.wheating_selection import same_harvest
        if prepared is not None and prepared.current(self, frame) and prepared.harvest is harvest:
            # The field selector already checked two distinct captures. Consume
            # its most recent controls/route once, without a redundant screenshot.
            fresh, checked = frame, harvest
            fresh_path, started = list(prepared.path), prepared.observed_at
        else:
            fresh = self._frame()
            started = time.monotonic()
            checked = self.vision.harvest(fresh.png)
            fresh_path = self._planned_harvest(fresh, checked) if checked else []
        if not same_harvest(harvest, checked, fresh) or not fresh_path:
            return ResourceResult('changed', 'The selected wheat moved before its harvest sweep.')
        if self.harvest_plan and self.harvest_plot_centers:
            fresh_path = [fresh_path[0], *self._motion.plots(fresh_path[1:], 'harvest')]
        elif self.harvest_plan:
            fresh_path = self._motion.sweep(fresh_path)
        timing = self._motion.timing(len(fresh_path))
        operation = uuid.uuid4().hex
        self._fresh(started)
        proof = self._evidence(fresh, operation, 'harvest_before')
        self._record(key, 'harvest_attempted', operation=operation, harvest_before=proof,
                     harvest_after=[], replanted=False, replant_evidence=None,
                     harvest_points=[list(p) for p in fresh_path], time_factors=list(timing),
                     harvest_route={
                         'mode': self.harvest_route_mode or (
                             'known_grid' if self.harvest_plan and self.harvest_plot_centers
                             else 'foliage' if self.harvest_plan else 'single'),
                         'observed_points': int(self.harvest_observed_points),
                         'expected_points': int(self.harvest_expected_points),
                         'route_points': max(0, len(fresh_path)-1),
                     })
        self._check()
        self._fresh(started)
        # A verified plot route already emits a touch inside every crop, just
        # like seeding. Foliage sweeps need extra samples between their row ends
        # so fast movement cannot jump over an untracked plot.
        duration = self._sweep_duration(fresh_path)
        if self.harvest_plan and not self.harvest_plot_centers:
            # The horizontal sweep revisits each tile on adjacent scan lines.
            # Shorten its travel time while retaining every 32-pixel sample;
            # plot-center harvesting and planting keep their existing timing.
            duration = max(250, round(duration*.5))
        spacing = self.harvest_sample_spacing
        if spacing is None and not (self.harvest_plan and self.harvest_plot_centers):
            spacing = max(1, round(32*fresh.height/1080))
        waypoint_hold = self.harvest_waypoint_hold
        if waypoint_hold is None:
            waypoint_hold = 40 if self.harvest_plan and self.harvest_plot_centers else 0
        self.client.drag_path([checked.tool.center, *fresh_path], width=fresh.width, height=fresh.height,
            duration_ms=duration, cancel_event=self.cancel_event,
            time_factors=timing,
            # Reusing evdev removes the old per-open dwell. Explicitly give
            # each crop center two game frames, including the final crop,
            # without imposing a hold on every horizontal interpolation point.
            min_waypoint_ms=waypoint_hold,
            max_step_px=None if self.harvest_plan and self.harvest_plot_centers else spacing)
        return self._replant_after_harvest(fresh, checked, icon, key, operation)

    def _await_harvested_soil(self, before, harvest, key):
        from hayday.wheating_transition import tracked_soil

        prior = None
        evidence_frames = []
        self._replant_grid = None
        self._harvest_shift = None
        tracked = bool(self.harvest_plan and self.harvest_plot_centers)
        for _ in range(12):
            self._pause(.05)
            frame = self._quick_frame()
            evidence_frames = [*evidence_frames[-1:], frame]
            if tracked:
                soil = tracked_soil(self, frame)
            else:
                soil = None if self._see('harvest', frame) else self._exposed_soil(before, frame, harvest)
            self._check()
            if tracked and soil is not None:
                # This only opens a known soil tile. The two independent
                # picker/seed observations still gate the planting gesture.
                break
            distinct = (len(evidence_frames) == 2 and bool(frame.captured_at)
                        and frame.captured_at != evidence_frames[0].captured_at)
            if distinct and self._same_target(prior, soil, frame):
                break
            prior = soil
        else:
            soil = None
        operation = self.state['items'][key]['operation']
        evidence = [self._evidence(f, operation, f'harvest_after_{i}') for i, f in enumerate(evidence_frames)]
        self._record(key, 'harvested_needs_replant' if soil else 'harvest_attempted', harvest_after=evidence)
        return frame, soil

    def _replant_after_harvest(self, before, harvest, icon, key, operation):
        harvested_at = time.monotonic()
        after, soil = self._await_harvested_soil(before, harvest, key)
        if soil is None:
            return ResourceResult('unsupported', 'The wheat harvest sweep is unconfirmed; its intent was retained.')
        # Known grid coordinates survive flying rewards. Only unfamiliar
        # foliage routes need an unobscured frame to trace their planting grid.
        if self._replant_grid is None:
            # Foliage routes have no saved grid to prove every tile clear. Let
            # rewards age from harvest completion, including observation time.
            self._pause(max(0., 3.-(time.monotonic()-harvested_at)))
            cleared = self._quick_frame()
            point = self._translated_plot(after, cleared, soil.center)
            if point is None:
                return ResourceResult('unsupported', 'The harvested field could not be aligned after its rewards cleared.')
            soil = replace(soil, x=point[0]-soil.width//2, y=point[1]-soil.height//2)
            after = cleared
        residual = self._residual_grid_points(after)
        proof = self._evidence(after, operation, 'harvest_clear')
        grid_proof = ({**self._evidence(self._replant_grid[0], operation, 'replant_grid'),
                       'points': [list(p) for p in self._replant_grid[1]],
                       'pitch': list(self._replant_grid[2])} if self._replant_grid else None)
        self._record(key, 'harvested_needs_replant', harvest_after=[proof], replant_grid=grid_proof,
                     harvest_validation={
                         'expected_points': int(self.harvest_expected_points or
                                                (len(self.harvest_plan[1]) if self.harvest_plan else 0)),
                         'residual_points': [list(point) for point in residual],
                         'residual_count': len(residual),
                     })
        self.soil_frame = after
        self._picker_before = self._replant_grid[0] if self._replant_grid else after
        expected = self._replant_origin if self._replant_grid else soil.center
        self._check()
        self.client.tap(*soil.center, width=after.width, height=after.height)
        self._pause(.2)
        opened = self._quick_frame()
        opened, plot = self._await_seed_picker(opened, expected)
        if plot is None:
            return ResourceResult('unsupported', 'The harvested field seed picker did not settle.')
        proof = self._evidence(opened, operation, 'replant_picker')
        self._record(key, 'harvested_needs_replant', replant_evidence={**proof, 'plot': list(plot.box)})
        return self._plant_selected(opened, icon, key, plot, True, plot.center)

    def _await_seed_picker(self, frame, expected):
        self._prepared_seed_picker = None
        before = getattr(self, '_picker_before', None)
        if before is None:
            return super()._await_seed_picker(frame, expected)
        prior = None
        prior_frame = None
        for _ in range(8):
            point = self._translated_plot(before, frame, expected)
            plot = self.vision.empty_plot(frame.png)
            if point and plot and np.linalg.norm(np.subtract(point, plot.center)) < max(24, plot.width*.8):
                if (prior_frame is not None and frame.captured_at
                        and prior_frame.captured_at != frame.captured_at
                        and self._same_target(prior, plot, frame)):
                    self._picker_before = None
                    self._prepared_seed_picker = prior_frame, prior, frame, time.monotonic()
                    return frame, plot
                prior = plot
                prior_frame = frame
            else:
                prior = None
                prior_frame = None
            self._pause(.03)
            frame = self._quick_frame()
        self._picker_before = None
        return frame, None

    def _count(self, frame, seed):
        for separate in (False, True):
            box = self._stock_box(frame, seed, separate=separate)
            if box:
                count = read_number(frame.png, box, image=self.vision._frame(frame.png))
                if count is not None:
                    return count
        return super()._count(frame, seed)

    def _plant_selected(self, frame, icon, key, plot=None, harvested=False, tracked_plot=None):
        prepared, self._prepared_seed_picker = self._prepared_seed_picker, None
        if prepared and (prepared[2] is not frame or time.monotonic()-prepared[3] > 1.):
            prepared = None
        self.planted_points = []
        self.plant_before = None
        self.resumed_mature = False
        self.growth_ready_at = None
        plot = plot or self.vision.empty_plot(frame.png)
        if plot is None:
            return super()._plant_selected(frame, icon, key, plot, harvested, tracked_plot)
        # These are already separate, settled observations of the same picker.
        # Check stock in each instead of adding another screenshot after them.
        seed_frame, seed_plot = prepared[:2] if prepared else (frame, plot)
        seed = self._seed(seed_frame, icon, seed_plot)
        available = self._count(seed_frame, seed) if seed else None
        retrying_stock = available is None and prepared is not None
        if retrying_stock:
            # The outline can settle one frame before a flying harvest reward
            # clears the wheat icon. Use the newer observation, then require a
            # fresh matching seed/stock below; never plant from one reading.
            seed = self._seed(frame, icon, plot)
            available = self._count(frame, seed) if seed else None
            prepared = None
        if available is None:
            return ResourceResult('unsupported', 'Wheat seed stock could not be verified in the selected picker.')
        if available < 1:
            return ResourceResult('waiting', 'Waiting for wheat seed stock; ripe wheat will be checked first.', {'needs_seed': True})
        from hayday.wheating_transition import planting_grid, saved_grid

        soil_frame = self.soil_frame or frame
        grid = saved_grid(self, self.state['items'].get(key, {})) if harvested else None
        if harvested and self.state['items'].get(key, {}).get('replant_grid') is not None and grid is None:
            return ResourceResult('unsupported', 'The saved planting grid could not be verified.')
        if grid:
            soil_points = planting_grid(self, frame, plot, grid)
            origin = plot.center
        else:
            origin = self._translated_plot(frame, soil_frame, plot.center, require_visible=False) if soil_frame is not frame else plot.center
            if origin is None:
                return ResourceResult('changed', 'The whole field could not be aligned with its seed picker.')
            soil_plot = replace(plot, x=origin[0]-plot.width//2, y=origin[1]-plot.height//2)
            soil_points = self.vision.empty_tiles(soil_frame, soil_plot, 98)
        if not soil_points:
            return ResourceResult('unsupported', 'The full selected plot outline is needed to trace one planting sweep.')
        self.field_size = len(soil_points)
        if len(soil_points) > available:
            self.progress(f'Wheating: using {available} seeds in one sweep across the {len(soil_points)}-plot field; the next harvest will supply the rest.')
            soil_points = soil_points[:available]
        shift = np.subtract(plot.center, origin)
        points = [tuple(map(int, np.add(p, shift))) for p in soil_points]
        if len(points) < 2:
            self.plant_before = frame
            planted_at = time.monotonic()
            result = super()._plant_selected(frame, icon, key, plot, harvested, tracked_plot)
            if self.state['items'].get(key, {}).get('replanted'):
                self.planted_points = [plot.center]
                self.remaining_seed_stock = available-1
                self.growth_ready_at = planted_at+self.growth_duration
                if self.growth_timer and self.growth_timer.seconds is None:
                    timer_frame = self.growth_timer.measure(self, self._quick_frame(), frame,
                                                          plot.center, planted_at)
                    self._close(timer_frame)
                return ResourceResult('waiting', 'Wheat replanted; growth confirmed.', {'planted': 1})
            return result
        fresh = frame if prepared else self._quick_frame()
        if retrying_stock and (not fresh.captured_at or fresh.captured_at == frame.captured_at):
            return ResourceResult('changed', 'A fresh wheat seed stock confirmation is needed before planting.')
        started = prepared[3] if prepared else time.monotonic()
        checked = self.vision.empty_plot(fresh.png)
        fresh_seed = self._seed(fresh, icon, checked) if checked else None
        if (not self._same_target(plot, checked, fresh) or not self._same_target(seed, fresh_seed, fresh)
                or self._count(fresh, fresh_seed) != available):
            return ResourceResult('changed', 'The wheat seed stock or empty field changed before its sweep.')
        # Floating particles can briefly cover individual tiles. Sweep only the
        # soil verified in both observations; rediscover the remainder afterward.
        if soil_frame is frame and grid is None:
            current_points = self.vision.empty_tiles(fresh, checked, available)
            points = [p for p in points if p in current_points]
        if not points or points[0] != plot.center:
            return ResourceResult('changed', 'The selected soil became obscured before planting.')
        ordered = self._motion.plots(points, 'seed')
        timing = self._motion.timing(len(ordered))
        operation = uuid.uuid4().hex
        self._fresh(started)
        proof = self._evidence(fresh, operation, 'plant_before')
        self._record(key, 'plant_attempted', operation=operation, available_before=available,
                     plant_before=proof, points=[list(p) for p in ordered], time_factors=list(timing))
        self._check()
        self._fresh(started)
        self.client.drag_path([fresh_seed.center, *ordered], width=fresh.width, height=fresh.height,
                              duration_ms=self._sweep_duration(ordered), cancel_event=self.cancel_event,
                              min_waypoint_ms=50, time_factors=timing)
        planted_at = time.monotonic()
        self.growth_ready_at = planted_at+self.growth_duration
        # Releasing over a planted tile can open its growth timer across several
        # other plots and recenter the camera. Let it settle before choosing
        # grass, so closing the timer cannot select a moving building instead.
        previous_view = None
        for _ in range(8):
            self._pause(.25)
            settled = self._frame()
            current_view = CameraNavigator._view(settled)
            if previous_view is not None and CameraNavigator._same_view(previous_view, current_view):
                if self.growth_timer and self.growth_timer.seconds is None:
                    settled = self.growth_timer.measure(self, settled, fresh, ordered[-1], planted_at)
                self._close(settled)
                break
            previous_view = current_view
        else:
            return ResourceResult('unsupported', 'The camera did not settle after planting; its intent remains pending.')
        previous = False
        for _ in range(8):
            self._pause(.35)
            after = self._frame()
            translated = self._translated_plot(fresh, after, plot.center)
            confirmed = False
            if translated:
                shift = np.subtract(translated, plot.center)
                hsv = cv2.cvtColor(_decode(after.png), cv2.COLOR_BGR2HSV)
                confirmed = True
                for point in ordered:
                    pixels = self.vision.tile_pixels(hsv, np.add(point, shift), (plot.width-6)/2, (plot.height-6)/2)
                    if pixels is None:
                        confirmed = False
                        break
                    green = (pixels[:, 0] >= 28) & (pixels[:, 0] <= 90) & (pixels[:, 1] > 70) & (pixels[:, 2] > 70)
                    if float(green.mean()) < .035:
                        confirmed = False
                        break
            if confirmed and previous:
                self._record(key, 'growing', replanted=True, planted_count=len(ordered))
                self.planted_points, self.plant_before = ordered, fresh
                self.remaining_seed_stock = available-len(ordered)
                self._close(after)
                return ResourceResult('waiting', f'Planted wheat on {len(ordered)} observed empty plots.', {'planted': len(ordered)})
            previous = confirmed
        return ResourceResult('unsupported', 'The wheat sweep was sent once but not every plot confirmed growth; its intent remains pending.')

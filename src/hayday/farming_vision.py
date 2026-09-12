"""Recognize observed farming controls; this module never sends input."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode


@dataclass(frozen=True)
class HarvestTarget:
    tool: VisualTarget
    target: VisualTarget
    score: float
    scale: float
    drag_target: VisualTarget | None = None
    highlight: VisualTarget | None = None

    @property
    def tool_center(self) -> tuple[int, int]:
        return self.tool.center


class FarmingVision:
    """Match generic UI geometry without recognizing or guessing crop species.

    A harvest target requires the sickle, its gold drag arrow, and a white
    selection highlight in agreement. The short drag ends at selected foliage.
    Replanting uses newly exposed soil observed by the worker after harvesting,
    rather than assuming the foliage and its soil share a fixed offset.
    """

    def __init__(self, reference_path: Path | None = None, *, cancel: Callable[[], bool] | None = None):
        self.cancel = cancel or (lambda: False)
        source = Path(__file__).resolve().parents[2] / "images/farming"
        self.reference_path = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent / "assets/farming"
        )
        try:
            self.manifest = json.loads((self.reference_path / "manifest.json").read_text("utf-8"))
            self.references = {name: _decode((self.reference_path / f"{name}.png").read_bytes(), True)
                for name in ("sickle", "guide_arrow", "guide_tip", "empty_plot", "page_previous", "page_next", "growth_bar")}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Farming UI references are missing or invalid.") from exc
        for reference in self.references.values():
            if reference.ndim != 3 or reference.shape[2] != 4 or np.count_nonzero(reference[:, :, 3]) < 40:
                raise ValueError("Farming references require original pixels with an alpha inclusion mask.")
        # Share the resource reader's mask-aware, coarse-to-fine visual matcher.
        self._matcher = ResourceVision()
        self._last_png: bytes | None = None
        self._image: np.ndarray | None = None
        self._cache: dict[tuple[str, float], tuple[VisualTarget, ...]] = {}

    def _frame(self, png: bytes) -> np.ndarray:
        if self.cancel():
            raise RuntimeError("Farming vision cancelled.")
        if png != self._last_png:
            self._image = _decode(png)
            self._last_png = png
            self._cache.clear()
        return self._image

    def _matches(self, png: bytes, name: str, threshold: float = .91) -> tuple[VisualTarget, ...]:
        image = self._frame(png)
        key = name, threshold
        if key not in self._cache:
            base = image.shape[0]/1080
            scales = np.unique(np.r_[np.geomspace(base*.55, base*1.5, 23), base])
            self._cache[key] = self._matcher._search(
                image, self.references[name], scales, threshold, self.cancel, max_peaks=4,
            )
        return self._cache[key]

    @staticmethod
    def _unambiguous(matches: tuple[VisualTarget, ...]) -> VisualTarget | None:
        if not matches or (len(matches) > 1 and matches[0].score-matches[1].score < .025):
            return None
        return matches[0]

    @staticmethod
    def _relative(box, anchor: VisualTarget, scale: float, source_box) -> VisualTarget:
        x0, y0, x1, y1 = box
        return VisualTarget(round(anchor.x+(x0-source_box[0])*scale),
            round(anchor.y+(y0-source_box[1])*scale), max(1, round((x1-x0)*scale)),
            max(1, round((y1-y0)*scale)), anchor.score)

    def harvest(self, png: bytes) -> HarvestTarget | None:
        image = self._frame(png)
        sickles = self._matches(png, "sickle", .90)
        if not sickles:
            return None
        arrows = self._matches(png, "guide_arrow", .90)
        geometry = self.manifest["harvest_geometry"]
        source = geometry["sickle_box"]
        candidates = []
        for sickle in sickles:
            scale = sickle.width / self.references["sickle"].shape[1]
            arrow_box = self.manifest["features"]["guide_arrow"]["box"]
            expected = self._relative(arrow_box, sickle, scale, source)
            agreed = next((arrow for arrow in arrows if
                abs(arrow.center[0]-expected.center[0]) <= 12*scale
                and abs(arrow.center[1]-expected.center[1]) <= 12*scale
                and .86 <= arrow.width/max(1, expected.width) <= 1.14), None)
            if agreed is None:
                continue
            highlight = self._relative(geometry["white_highlight_box"], sickle, scale, source)
            hx, hy, hw, hh = highlight.box
            target = self._relative(geometry["target_box"], sickle, scale, source)
            if min(hx, hy, target.x, target.y) < 0 or max(hx+hw, target.x+target.width) > image.shape[1] or max(hy+hh, target.y+target.height) > image.shape[0]:
                continue
            crop = image[hy:hy+hh, hx:hx+hw]
            # Bright low-saturation crop outline is separate evidence from the
            # gold arrow; ordinary ripe yellow crops are insufficient.
            low, high = crop.min(axis=2), crop.max(axis=2)
            white = (low > 195) & (high-low < 65)
            if np.count_nonzero(white) < max(20, round(150*scale*scale)):
                continue
            tx, ty = geometry["tool_point"]
            tool = self._relative((tx-6, ty-6, tx+6, ty+6), sickle, scale, source)
            outlined = self._selected_foliage(image, target, scale)
            if outlined is None:
                continue
            highlight, drag_target, ground = outlined
            candidates.append(HarvestTarget(tool, ground, min(sickle.score, agreed.score),
                                            scale, drag_target, highlight))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0].score-candidates[1].score < .025):
            return None
        return candidates[0]

    @staticmethod
    def _selected_foliage(image, expected, scale):
        """Locate the actual white crop outline; the guide only bounds its search.

        Tool artwork has a fixed UI size, while each crop has different foliage.
        Remove long guide beams, then join nearby outline fragments. Ordinary
        crop colors never count as selection evidence.
        """
        x, y, w, h = expected.box
        left, top = max(0, x-round(30*scale)), max(0, y-round(110*scale))
        right = min(image.shape[1], x+w+round(90*scale))
        bottom = min(image.shape[0], y+h+round(20*scale))
        patch = image[top:bottom, left:right]
        white = ((patch.min(axis=2) > 225) & (np.ptp(patch, axis=2) < 24)).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        for label, (px, py, pw, ph, _) in enumerate(stats[1:], 1):
            if (ph > max(30*scale, pw*3) or pw > max(90*scale, ph*4)
                    or px == 0 or py == 0 or px+pw == white.shape[1] or py+ph == white.shape[0]):
                white[labels == label] = 0
        joined = cv2.dilate(white, np.ones((max(3, round(13*scale)),)*2, np.uint8))
        count, labels, _, _ = cv2.connectedComponentsWithStats(joined)
        groups = []
        for label in range(1, count):
            ys, xs = np.nonzero((labels == label) & (white > 0))
            if len(xs) < max(35, 120*scale*scale):
                continue
            bw, bh = int(np.ptp(xs))+1, int(np.ptp(ys))+1
            if not (30*scale < bw < 180*scale and 20*scale < bh < 180*scale):
                continue
            groups.append((len(xs), xs, ys, bw, bh))
        groups.sort(key=lambda entry: entry[0], reverse=True)
        if not groups or len(groups) > 1 and groups[1][0] >= groups[0][0]*.65:
            return None
        _, xs, ys, bw, bh = groups[0]
        cx, cy = left+round(float(np.median(xs))), top+round(float(np.percentile(ys, 65)))
        highlight = VisualTarget(left+int(xs.min()), top+int(ys.min()), bw, bh, expected.score)
        radius = max(4, round(8*scale))
        drag = VisualTarget(cx-radius, cy-radius, radius*2, radius*2, expected.score)
        # Approximate footprint for pre-gesture stability checks only. The
        # post-harvest soil tap must come from before/after image evidence.
        ground_y = highlight.y+highlight.height+round(11*scale)
        ground = VisualTarget(cx-round(w/2), ground_y-round(h/2), w, h, expected.score)
        return highlight, drag, ground

    def _plot_evidence(self, image: np.ndarray, target: VisualTarget) -> bool:
        x, y, w, h = target.box
        patch = image[max(0, y):min(image.shape[0], y+h),
                      max(0, x):min(image.shape[1], x+w)]
        if patch.shape[:2] != (h, w) or patch.size == 0:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        soil = ((hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 45)
                & (hsv[:, :, 2] > 45) & (hsv[:, :, 2] < 235))
        rim = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 185)
        return float(soil.mean()) >= .45 and float(rim.mean()) >= .035

    def _seed_controls(self, png: bytes) -> tuple[VisualTarget, VisualTarget] | None:
        previous = self._matches(png, "page_previous", .92)
        following = self._matches(png, "page_next", .92)
        arrows = self._matches(png, "guide_tip", .89)
        candidates = []
        for left in previous:
            for right in following:
                scale = (left.width+right.width)/(123+124)
                if (abs(right.center[1]-left.center[1]) > 15*scale
                        or abs(right.center[0]-left.center[0]-337*scale) > 28*scale
                        or not .85 <= left.width/max(1, right.width) <= 1.15):
                    continue
                visible = [arrow for arrow in arrows if
                    left.x-30*scale < arrow.center[0] < right.center[0]+350*scale
                    and left.y-600*scale < arrow.center[1] < left.y]
                if visible:
                    candidates.append((min(left.score, right.score), left, right))
        candidates.sort(reverse=True, key=lambda value: value[0])
        if not candidates or (len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .025):
            return None
        return candidates[0][1], candidates[0][2]

    def empty_plot(self, png: bytes) -> VisualTarget | None:
        image = self._frame(png)
        strict = tuple(target for target in self._matches(png, "empty_plot", .93)
                       if self._plot_evidence(image, target))
        if strict:
            # Multiple strong plots stay ambiguous. Control-relative geometry
            # may recover a weak changed-layout match, but must not overrule
            # conflicting high-confidence visual evidence.
            return self._unambiguous(strict)
        controls = self._seed_controls(png)
        if controls is None:
            return None
        _, right = controls
        scale = right.width/124
        relaxed = tuple(target for target in self._matches(png, "empty_plot", .75)
            if self._plot_evidence(image, target)
            and right.center[1]-420*scale <= target.center[1] < right.center[1]
            and right.center[0]-80*scale < target.center[0] < right.center[0]+500*scale)
        return self._unambiguous(relaxed)

    def seed_menu(self, png: bytes) -> bool:
        return self._seed_controls(png) is not None

    def page_next(self, png: bytes) -> VisualTarget | None:
        controls = self._seed_controls(png)
        return controls[1] if controls else None

    def growing(self, png: bytes) -> VisualTarget | None:
        """Find the blue pill and its white frame, independent of label/fill.

        Both completed and remaining segments are blue. Matching their union
        avoids tying growth detection to a particular countdown, plant name,
        text color, or progress percentage in the original Corn capture.
        """
        image = self._frame(png)
        height, width = image.shape[:2]
        base = height/1080
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (95, 80, 40), (125, 255, 255))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
        white = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 200)
        candidates = []
        for label, (x, y, w, h, area) in enumerate(stats[1:count], 1):
            if self.cancel():
                raise RuntimeError('Farming vision cancelled.')
            if not (24*base <= h <= 80*base and 9 <= w/h <= 13
                    and area >= w*h*.88):
                continue
            pill = labels[y:y+h, x:x+w] == label
            cap = max(2, round(h*.35))
            corners = (pill[:cap, :cap], pill[-cap:, :cap],
                       pill[:cap, -cap:], pill[-cap:, -cap:])
            if (any(not .25 <= float(corner.mean()) <= .86 for corner in corners)
                    or float(pill[h//3:2*h//3].mean()) < .95):
                continue
            scale = h/49
            # Read independent portions of the same frame, outside the label,
            # timer, progress fill, and the neighboring speed-up button.
            regions = (
                (-24, 8, -7, h/scale-8),
                (10, h/scale+3, 90, h/scale+20),
                (w/scale-90, h/scale+3, w/scale-10, h/scale+20),
                (20, -11, 90, -3),
                (w/scale-90, -11, w/scale-20, -3),
            )
            support = []
            for left, top, right, bottom in regions:
                left, right = round(x+left*scale), round(x+right*scale)
                top, bottom = round(y+top*scale), round(y+bottom*scale)
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    break
                support.append(float(white[top:bottom, left:right].mean()))
            if len(support) == len(regions) and min(support) >= .85:
                candidates.append(VisualTarget(round(x-34*scale), round(y-18*scale),
                    round(w+56*scale), round(h+58*scale), min(support)))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return self._unambiguous(tuple(candidates))

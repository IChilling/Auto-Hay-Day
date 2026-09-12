"""Find corner X and double-arrow skip controls without inferring ad completion.

The caller must separately establish ad completion and reobserve before input.
This module has no device client and cannot click, skip, or close anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AdExitCandidate:
    bounds: tuple[int, int, int, int]
    score: float
    corner: str
    kind: str = "close"

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.bounds
        return x+width//2, y+height//2


class AdExitDetector:
    """Recognize neutral, high-contrast X or skip controls in either upper corner.

    Both diagonals need four balanced arms and an isolated, uniform background.
    Nearby lettering, colorful game buttons, plus/minus icons and irregular
    photographic edges are rejected. An isolated typographic X can be visually
    identical to a close icon; caller-owned ad state is therefore mandatory.
    A separate skip rule requires two filled right-pointing triangles and a
    terminal bar, allowing the colored overlay and adjacent store label used
    by some ad providers without weakening the X checks.
    """

    @staticmethod
    def _shape(mask: np.ndarray) -> float | None:
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        if len(xs) < 12 or not .68 <= w/h <= 1.48:
            return None
        fill = len(xs)/(w*h)
        if not .12 <= fill <= .65:
            return None
        x = (xs-(w-1)/2)/max(1, (w-1)/2)
        y = (ys-(h-1)/2)/max(1, (h-1)/2)
        diagonal = np.minimum(np.abs(x-y), np.abs(x+y))
        # Plus signs and branching letters put substantial ink off diagonals.
        if float(np.quantile(diagonal, .90)) > .38 or float((diagonal > .42).mean()) > .08:
            return None
        outer = (np.abs(x) > .35) & (np.abs(y) > .35)
        arms = [int(np.count_nonzero(outer & (x*sx > 0) & (y*sy > 0)))
                for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        if min(arms) < max(2, len(xs)*.055) or max(arms) > min(arms)*2.4:
            return None
        # Each diagonal must reach both opposite edges, including its corners.
        for sign in (-1, 1):
            along = np.abs(x-sign*y) <= .32
            if np.count_nonzero(along & (x < -.65)) < 2 or np.count_nonzero(along & (x > .65)) < 2:
                return None
        axis_ink = (np.maximum(np.abs(x), np.abs(y)) > .6) & (np.minimum(np.abs(x), np.abs(y)) < .2)
        if axis_ink.mean() > .055:
            return None
        symmetry = min(arms)/max(arms)
        alignment = 1-min(1., float(np.mean(diagonal))/.38)
        return .6*alignment+.4*symmetry

    @staticmethod
    def _isolated(binary: np.ndarray, labels: np.ndarray, identity: int,
                  x: int, y: int, w: int, h: int) -> bool:
        pad = max(4, round(max(w, h)*1.0))
        left, top = max(0, x-pad), max(0, y-pad)
        right, bottom = min(binary.shape[1], x+w+pad), min(binary.shape[0], y+h+pad)
        region = binary[top:bottom, left:right].copy()
        region[labels[top:bottom, left:right] == identity] = 0
        # A separate ring around the X is allowed; neighboring text is not.
        count, components, stats, centers = cv2.connectedComponentsWithStats(region)
        cx, cy = x+(w-1)/2-left, y+(h-1)/2-top
        for i in range(1, count):
            bx, by, bw, bh, area = stats[i]
            if (bw > w*1.25 and bh > h*1.25 and .72 <= bw/bh <= 1.38
                    and abs(centers[i][0]-cx) <= w*.25 and abs(centers[i][1]-cy) <= h*.25
                    and area < bw*bh*.6):
                ys, xs = np.nonzero(components == i)
                radius = np.sqrt(((xs-cx)/(bw/2))**2+((ys-cy)/(bh/2))**2)
                if np.quantile(np.abs(radius-1), .8) < .24:
                    region[components == i] = 0
        # Ignore material far above/below a line of text; aligned neighbors near
        # an X make EXIT/EXTRA/etc. unsafe to treat as a standalone control.
        band = region[max(0, y-top-h//3):min(region.shape[0], y-top+h+h//3)]
        return int(np.count_nonzero(band)) <= max(4, round(w*h*.12))

    @staticmethod
    def _appearance(image: np.ndarray, mask: np.ndarray) -> float | None:
        foreground = image[mask > 0].astype(np.float32)
        expanded = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        background = image[expanded == 0].astype(np.float32)
        if len(background) < max(6, mask.size*.08):
            return None
        fg = np.median(foreground, axis=0)
        bg = np.median(background, axis=0)
        if np.ptp(fg) > 48 or np.ptp(bg) > 55:
            return None
        contrast = abs(float(fg.mean()-bg.mean()))
        if contrast < 48:
            return None
        noise = np.quantile(np.abs(background-bg).mean(axis=1), .85)
        if noise > 36:
            return None
        return .75*min(1, contrast/160)+.25*(1-min(1, float(noise)/36))

    @staticmethod
    def _same(first: AdExitCandidate, second: AdExitCandidate) -> bool:
        a, b = first.bounds, second.bounds
        return (first.kind == second.kind
                and abs(first.center[0]-second.center[0]) < min(a[2], b[2])*.65
                and abs(first.center[1]-second.center[1]) < min(a[3], b[3])*.65)

    @staticmethod
    def _skip_shape(mask: np.ndarray) -> float | None:
        """Require two filled rightward triangles followed by a vertical bar.

        Geometric masks adapt to size, stroke thickness, and gaps. A play
        triangle, chevrons, and fast-forward without the stop bar are excluded.
        """
        h, w = mask.shape
        ink = mask > 0
        if not 1.55 <= w/h <= 2.8 or not .30 <= ink.mean() <= .78:
            return None
        best = 0.
        for bar_fraction in (.06, .10, .16, .22):
            bar = max(1, round(h*bar_fraction))
            # A real terminal bar reaches the top and bottom, not just a tip.
            if ink[:, -bar:].mean() < .83:
                continue
            for gap_fraction in (0., .05, .12, .20):
                gap = round(h*gap_fraction)
                available = w-bar-gap*2
                for ratio in (.85, 1., 1.15):
                    left = round(available*ratio/(1+ratio))
                    start, end = left+gap, w-bar-gap-1
                    if min(left, end-start+1) < h*.55:
                        continue
                    expected = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(expected, np.array([(0, 0), (0, h-1),
                        (left-1, (h-1)//2)], np.int32), 1)
                    cv2.fillConvexPoly(expected, np.array([(start, 0), (start, h-1),
                        (end, (h-1)//2)], np.int32), 1)
                    expected[:, -bar:] = 1
                    expected = expected > 0
                    intersection = np.count_nonzero(ink & expected)
                    precision = intersection/max(1, np.count_nonzero(ink))
                    recall = intersection/max(1, np.count_nonzero(expected))
                    iou = intersection/max(1, np.count_nonzero(ink | expected))
                    if precision >= .84 and recall >= .80:
                        best = max(best, iou)
        return best if best >= .74 else None

    def _skip_candidates(self, color, origin, corner, minimum, maximum, cancelled):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        candidates = []
        for threshold in (40, 72, 104, 136, 168, 200, 224):
            for inverse in (False, True):
                if cancelled():
                    return []
                binary = ((gray < threshold) if inverse else (gray > threshold)).astype(np.uint8)*255
                count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
                pieces = [(i, tuple(map(int, stats[i]))) for i in range(1, count)
                          if minimum <= stats[i][3] <= maximum
                          and 1 <= stats[i][2] <= maximum*2.8 and stats[i][4] >= 6]
                for identity, (x, y, w, h, _) in pieces:
                    if cancelled():
                        return []
                    # The arrows/bar may touch or form two/three components.
                    neighbors = sorted(((j, box) for j, box in pieces if j != identity
                        and x+w <= box[0] <= x+round(h*2.8)
                        and abs(box[1]-y) <= max(1, h*.18)
                        and .75 <= box[3]/h <= 1.25), key=lambda p: p[1][0])[:2]
                    for length in range(len(neighbors)+1):
                        group = [(identity, (x, y, w, h, 0)), *neighbors[:length]]
                        left, top = min(b[0] for _, b in group), min(b[1] for _, b in group)
                        right = max(b[0]+b[2] for _, b in group)
                        bottom = max(b[1]+b[3] for _, b in group)
                        if (left == 0 or top == 0 or right >= color.shape[1]
                                or bottom >= color.shape[0] or not 1.55 <= (right-left)/(bottom-top) <= 2.8):
                            continue
                        mask = np.isin(labels[top:bottom, left:right], [i for i, _ in group])
                        shape = self._skip_shape(mask)
                        if shape is None:
                            continue
                        patch = color[top:bottom, left:right].astype(np.float32)
                        fg, bg = np.median(patch[mask], axis=0), np.median(patch[~mask], axis=0)
                        # Unlike X controls, this SDK glyph can sit beside a
                        # store label on a translucent, colored video overlay.
                        contrast = abs(float(fg.mean()-bg.mean()))
                        if (np.ptp(fg) > 40 or contrast < 45
                                or np.quantile(np.abs(patch[mask]-fg).mean(axis=1), .85) > 28):
                            continue
                        score = .85*shape+.15*min(1., contrast/160)
                        candidates.append(AdExitCandidate((origin+left, top, right-left, bottom-top),
                                                          float(min(1., score)), corner, "skip"))
        return candidates

    def detect(self, png: bytes, cancel: Callable[[], bool] | None = None) -> AdExitCandidate | None:
        cancelled = cancel or (lambda: False)
        if cancelled():
            return None
        try:
            image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        except (cv2.error, TypeError, ValueError):
            return None
        if (image is None or min(image.shape[:2]) < 80 or max(image.shape[:2]) > 16384
                or image.shape[0]*image.shape[1] > 64_000_000):
            return None
        height, width = image.shape[:2]
        roi_width, roi_height = round(width*.25), round(height*.18)
        minimum = max(7, round(min(width, height)*.007))
        maximum = max(20, min(100, round(min(width, height)*.10)))
        candidates = []
        for corner, origin in (("top_left", 0), ("top_right", width-roi_width)):
            color = image[:roi_height, origin:origin+roi_width]
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            for threshold in (40, 72, 104, 136, 168, 200, 224):
                for inverse in (False, True):
                    if cancelled():
                        return None
                    binary = ((gray < threshold) if inverse else (gray > threshold)).astype(np.uint8)*255
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
                    for identity in range(1, count):
                        if identity % 64 == 0 and cancelled():
                            return None
                        x, y, w, h, area = stats[identity]
                        if (min(w, h) < minimum or max(w, h) > maximum or area < 12
                                or x == 0 or y == 0 or x+w >= roi_width or y+h >= roi_height):
                            continue
                        mask = (labels[y:y+h, x:x+w] == identity).astype(np.uint8)*255
                        shape = self._shape(mask)
                        if shape is None:
                            continue
                        appearance = self._appearance(color[y:y+h, x:x+w], mask)
                        if appearance is None or not self._isolated(binary, labels, identity, x, y, w, h):
                            continue
                        score = .7*shape+.3*appearance
                        if score >= .73:
                            candidates.append(AdExitCandidate(tuple(map(int, (origin+x, y, w, h))), min(1., score), corner))
            candidates.extend(self._skip_candidates(color, origin, corner, minimum, maximum, cancelled))
        if cancelled():
            return None
        distinct = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if not any(self._same(candidate, previous) for previous in distinct):
                distinct.append(candidate)
        # Never choose between two plausible exits from pixels alone.
        return distinct[0] if len(distinct) == 1 else None

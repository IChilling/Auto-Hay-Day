"""Recognize shop skins by their shared counter rim and wooden platform."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.wheating_shop_detail import detail_image, plausible_lighting


class ShopBuildingVision:
    def __init__(self, cancel=lambda: False):
        self.cancel = cancel
        root = Path(__file__).parent/'assets/wheating/shop_building'
        self.refs = {name: _decode((root/f'{name}.png').read_bytes(), True)
                     for name in ('shop', 'counter', 'deck')}
        self.matcher = ResourceVision()
        self._detail_refs = {name: detail_image(ref) for name, ref in self.refs.items()}
        self._previous = None
        self._size = None

    def _part(self, image, target, name, refs=None):
        """Independently validate each feature, allowing only a few raster pixels of drift."""
        rgba = (self.refs if refs is None else refs)[name]
        color, mask = self.matcher._scaled(rgba, target.width/rgba.shape[1])
        ys, xs = np.nonzero(mask)
        x, y, right, bottom = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        color, mask = color[y:bottom, x:right], mask[y:bottom, x:right]
        h, w = mask.shape
        pad = max(2, round(target.width*.015))
        left, top = max(0, target.x+x-pad), max(0, target.y+y-pad)
        crop = image[top:min(len(image), target.y+bottom+pad),
                     left:min(image.shape[1], target.x+right+pad)]
        if crop.shape[0] < h or crop.shape[1] < w:
            return False
        response = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
        _, correlation, _, (cx, cy) = cv2.minMaxLoc(response)
        patch = crop[cy:cy+h, cx:cx+w]
        difference = np.abs(patch[mask > 0].astype(np.float32)-color[mask > 0]).mean()
        return correlation >= .70 and difference <= 42

    def _search(self, image, scales, region=None, *, detail=False):
        x, y, w, h = region or (0, 0, image.shape[1], image.shape[0])
        refs = self._detail_refs if detail else self.refs
        observed = detail_image(image, float(scales[0])) if detail else image
        candidates = self.matcher._search(observed[y:y+h, x:x+w], refs['shop'],
            scales, .88 if detail else .86, self.cancel, max_peaks=4)
        verified = []
        for candidate in candidates:
            if self.cancel():
                return ()
            target = VisualTarget(candidate.x+x, candidate.y+y,
                                  candidate.width, candidate.height, candidate.score)
            # Require both the distinctive counter rim and the plank pattern at
            # the same scale/position. Roof skins, the tray's notification, and
            # nearby decorations contribute no matching pixels.
            if (all(self._part(observed, target, name, refs) for name in ('counter', 'deck'))
                    and (not detail or plausible_lighting(image, target, self.refs, self.matcher))):
                verified.append(target)
        return tuple(verified)

    @staticmethod
    def _unique(hits):
        return hits[0] if hits and (len(hits) == 1 or hits[0].score-hits[1].score > .045) else None

    def find(self, image):
        if self.cancel():
            return None
        height, width = image.shape[:2]
        previous = self._previous if self._size == (width, height) else None
        if previous is not None:
            pad = round(previous.width*.45)
            left, top = max(0, previous.x-pad), max(0, previous.y-pad)
            right, bottom = min(width, previous.x+previous.width+pad), min(height, previous.y+previous.height+pad)
            scale = previous.width/self.refs['shop'].shape[1]
            hits = self._search(image, np.array([scale]), (left, top, right-left, bottom-top))
            if not hits:
                hits = self._search(image, np.array([scale]), (left, top, right-left, bottom-top), detail=True)
            target = self._unique(hits)
            if target:
                self._previous = target
                return target
        # Search the actual current pixels when the shop/camera moved. The prior
        # location only accelerates recognition; it is never an input coordinate.
        base = height/1045
        hits = self._search(image, np.array([base]))
        if not hits:
            # Clouds can wash out base colors while its fine geometry remains
            # visible. This fallback still requires both parts and wood colors.
            hits = self._search(image, np.array([base]), detail=True)
        if not hits:
            hits = self._search(image, np.geomspace(base*.35, base*1.5, 23))
        if not hits:
            # Coarse world scales can straddle the counter's fine plank lines.
            # A weak peak only proposes a neighborhood: the refined match must
            # pass the original score, counter and platform checks.
            proposals = self.matcher._search(image, self.refs['shop'],
                np.geomspace(base*.35, base*1.5, 23), .75, self.cancel, max_peaks=2)
            refined = []
            for proposal in proposals:
                pad = round(proposal.width*.15)
                left, top = max(0, proposal.x-pad), max(0, proposal.y-pad)
                right = min(width, proposal.x+proposal.width+pad)
                bottom = min(height, proposal.y+proposal.height+pad)
                scale = proposal.width/self.refs['shop'].shape[1]
                refined.extend(self._search(image, np.linspace(.95,1.05,21)*scale,
                    (left,top,right-left,bottom-top)))
            hits = tuple(sorted(refined, key=lambda t: -t.score))
        target = self._unique(hits)
        if target:
            self._previous, self._size = target, (width, height)
        return target

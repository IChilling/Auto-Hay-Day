"""Read two observed game dialogs; unknown prompts never acquire a tap target."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode


@dataclass(frozen=True)
class DialogFeature:
    name: str
    target: VisualTarget


@dataclass(frozen=True)
class DialogObservation:
    kind: str
    target: VisualTarget
    bounds: tuple[int, int, int, int]
    score: float
    features: tuple[DialogFeature, ...]


class DialogVision:
    """Match the exact fuel tutorial or Connection Lost wording and UI geometry.

    This class never connects to a device. A caller must reobserve before input,
    limit dismissal/reconnect attempts, and verify the resulting scene afterward.
    """

    def __init__(self, reference_path: Path | None = None):
        source = Path(__file__).resolve().parents[2]/'images/dialogs'
        self.reference_path = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/dialogs')
        try:
            manifest = json.loads((self.reference_path/'manifest.json').read_text('utf-8'))
            self.specifications = manifest['dialogs']
            if manifest.get('version') != 1 or not (
                    {'fuel_tutorial', 'connection_lost'} <= set(self.specifications)
                    <= {'fuel_tutorial', 'connection_lost', 'server_maintenance'}):
                raise ValueError('Unsupported dialog references.')
            self.references = {}
            for kind, entry in self.specifications.items():
                references = []
                if len(entry['features']) < 3:
                    raise ValueError('Dialog text requires multiple independent features.')
                for feature in entry['features']:
                    name = feature['file']
                    if Path(name).name != name:
                        raise ValueError('Invalid dialog reference filename.')
                    rgba = _decode((self.reference_path/name).read_bytes(), True)
                    box = feature['box']
                    if (rgba.ndim != 3 or rgba.shape[2] != 4
                            or rgba.shape[:2] != (box[3]-box[1], box[2]-box[0])
                            or np.count_nonzero(rgba[:, :, 3]) < 40):
                        raise ValueError('Dialog references require original RGB with alpha text masks.')
                    references.append(rgba)
                self.references[kind] = references
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Known dialog references are missing or invalid.') from exc
        # This matcher is read-only and uses original-pixel masked refinement.
        self._matcher = ResourceVision()

    @staticmethod
    def _check(cancel):
        if cancel():
            raise RuntimeError('Dialog recognition cancelled.')

    @staticmethod
    def _mapped(box, anchor, source, scale):
        x, y, width, height = box
        return (round(anchor.x+(x-source[0])*scale),
                round(anchor.y+(y-source[1])*scale),
                max(1, round(width*scale)), max(1, round(height*scale)))

    def _feature(self, image, rgba, box, anchor, source, scale):
        x0, y0, x1, y1 = box
        x, y, width, height = self._mapped((x0, y0, x1-x0, y1-y0), anchor, source, scale)
        radius = max(4, round(9*scale))
        color, mask = self._matcher._scaled(rgba, scale)
        th, tw = mask.shape
        left, top = max(0, x-radius), max(0, y-radius)
        right = min(image.shape[1], x+width+radius)
        bottom = min(image.shape[0], y+height+radius)
        local = image[top:bottom, left:right]
        if local.shape[0] < th or local.shape[1] < tw:
            return None
        response = cv2.matchTemplate(cv2.cvtColor(local, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
        _, correlation, _, (dx, dy) = cv2.minMaxLoc(response)
        patch = local[dy:dy+th, dx:dx+tw]
        difference = float(np.abs(patch[mask > 0].astype(float)-color[mask > 0]).mean())
        score = .8*correlation+.2*max(0, 1-difference/100)
        if score < .945:
            return None
        return VisualTarget(left+dx, top+dy, tw, th, score)

    def _candidate(self, image, kind, anchor, cancel, scale):
        entry = self.specifications[kind]
        source = entry['features'][0]['box']
        matched = []
        for feature, rgba in zip(entry['features'], self.references[kind], strict=True):
            self._check(cancel)
            found = self._feature(image, rgba, feature['box'], anchor, source, scale)
            if found is None:
                return None
            matched.append(DialogFeature(Path(feature['file']).stem, found))
        bounds = self._mapped(entry['bounds'], anchor, source, scale)
        x, y, width, height = bounds
        if x < -3 or y < -3 or x+width > image.shape[1]+3 or y+height > image.shape[0]+3:
            return None
        for background in entry['background']:
            bx, by, bw, bh = self._mapped(background['box'], anchor, source, scale)
            patch = image[max(0, by):by+bh, max(0, bx):bx+bw]
            if (not patch.size or np.abs(np.median(patch, axis=(0, 1))-background['color_bgr']).mean() > 18
                    or patch.std(axis=(0, 1)).mean() > 20):
                return None
        score = min(feature.target.score for feature in matched)
        if kind in {'connection_lost', 'server_maintenance'}:
            target = matched[-1].target  # The observed TRY AGAIN text, never a cached button point.
        else:
            target = VisualTarget(*self._mapped(entry['target'], anchor, source, scale), score)
        return DialogObservation(kind, target, bounds, score, tuple(matched))

    def _refined(self, image, kind, anchor, cancel):
        entry = self.specifications[kind]
        source = entry['features'][0]['box']
        source_width = source[2]-source[0]
        # Short words find a candidate efficiently. A long, independent text
        # line then resolves subpixel scale: one pixel of width error on the
        # small anchor would otherwise distort hundreds of letter pixels.
        index = max(range(1, len(entry['features'])),
                    key=lambda i: entry['features'][i]['box'][2]-entry['features'][i]['box'][0])
        feature = entry['features'][index]
        proposals = []
        for width in np.linspace(anchor.width-3, anchor.width+3, 33):
            self._check(cancel)
            scale = width/source_width
            matched = self._feature(image, self.references[kind][index], feature['box'],
                                    anchor, source, scale)
            if matched:
                proposals.append((matched.score, scale))
        best = None
        for _, scale in sorted(proposals, reverse=True)[:5]:
            candidate = self._candidate(image, kind, anchor, cancel, scale)
            if candidate and (best is None or candidate.score > best.score):
                best = candidate
        return best

    def observe(self, png: bytes, cancel: Callable[[], bool] | None = None) -> DialogObservation | None:
        cancel = cancel or (lambda: False)
        self._check(cancel)
        image = _decode(png)
        base = image.shape[0]/1080
        scales = np.unique(np.r_[np.geomspace(base*.55, base*1.5, 23), base])
        accepted = []
        for kind, references in self.references.items():
            self._check(cancel)
            anchors = self._matcher._search(image, references[0], scales, .78, cancel, max_peaks=3)
            for anchor in anchors:
                observation = self._refined(image, kind, anchor, cancel)
                if observation:
                    accepted.append(observation)
        self._check(cancel)
        if len(accepted) != 1:
            return None
        return accepted[0]

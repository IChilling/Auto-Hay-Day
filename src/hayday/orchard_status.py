"""Recognize an exhausted source reached through a known fruit's own link."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode


@dataclass(frozen=True)
class ExhaustedOrchard:
    species: str
    saw: VisualTarget
    help: VisualTarget
    arrow: VisualTarget


class OrchardStatusVision:
    def __init__(self, reference_path=None):
        checkout = Path(__file__).resolve().parents[2] / 'images/orchard_status'
        root = Path(reference_path) if reference_path else (
            checkout if checkout.is_dir() else Path(__file__).parent / 'assets/orchard_status')
        self.manifest = json.loads((root / 'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported orchard status references.')
        self.references = {name: _decode((root / (name+'.png')).read_bytes(), True)
                           for name in ('saw', 'help', 'arrow', 'apple', 'cherry')}
        for reference in self.references.values():
            if reference.ndim != 3 or reference.shape[2] != 4 or np.count_nonzero(reference[:, :, 3]) < 40:
                raise ValueError('Orchard status references require original pixels and alpha masks.')
        self.matcher = ResourceVision()

    def exhausted(self, png, desired_icon, cancel=lambda: False):
        desired = _decode(desired_icon)
        species = [name for name in self.manifest['species']
            if self.matcher._search(desired, self.references[name],
                np.unique(np.r_[np.geomspace(.45, 1.8, 24), 1.]), .93, cancel, max_peaks=2)]
        if len(species) != 1:
            return None
        image = _decode(png)
        base = image.shape[0]/1080
        scales = np.unique(np.r_[np.geomspace(base*.72, base*1.32, 17), base])
        saws = self.matcher._search(image, self.references['saw'], scales, .91, cancel, max_peaks=3)
        source = self.manifest['features']['saw']['box']
        results = []
        for saw in saws:
            scale = saw.width/(source[2]-source[0])
            controls = []
            for name in ('help', 'arrow'):
                box = self.manifest['features'][name]['box']
                expected = (saw.x+(box[0]-source[0])*scale, saw.y+(box[1]-source[1])*scale)
                margin = round(24*base)
                left, top = max(0, round(expected[0])-margin), max(0, round(expected[1])-margin)
                right = min(image.shape[1], round(expected[0]+(box[2]-box[0])*scale)+margin)
                bottom = min(image.shape[0], round(expected[1]+(box[3]-box[1])*scale)+margin)
                if right <= left or bottom <= top:
                    break
                matches = self.matcher._search(image[top:bottom, left:right], self.references[name],
                    (scale,), .90, cancel, max_peaks=2)
                if not matches:
                    break
                target = matches[0]
                if math.dist((left+target.x, top+target.y), expected) > 20*base:
                    break
                controls.append(VisualTarget(left+target.x, top+target.y,
                    target.width, target.height, target.score))
            if len(controls) == 2:
                results.append(ExhaustedOrchard(species[0], saw, *controls))
        return results[0] if len(results) == 1 else None

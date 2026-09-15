"""Positive farm HUD evidence, independent of loading artwork and farm layout."""
import json
import threading
from pathlib import Path

import numpy as np

from hayday.resource_vision import ResourceVision, _decode


class FarmSceneVision:
    def __init__(self):
        source = Path(__file__).resolve().parents[2]/'images/launcher'
        root = source if source.is_dir() else Path(__file__).parent/'assets/launcher'
        hud = json.loads((root/'manifest.json').read_text('utf-8'))['farm_hud']
        self.references = {name: _decode((root/spec['file']).read_bytes(), True)
                           for name, spec in hud['features'].items()}
        self.matcher = ResourceVision()
        self._last_png = None
        self._last_ready = False
        self._anchors = {}

    def _feature(self, image, name, scales):
        reference = self.references[name]
        anchor = self._anchors.get(name)
        if anchor is not None and anchor[0] == image.shape:
            target = anchor[1]
            margin = max(5, round(target.width*.08))
            left, top = max(0, target.x-margin), max(0, target.y-margin)
            local = image[top:target.y+target.height+margin, left:target.x+target.width+margin]
            nearby = np.array([target.width-1, target.width, target.width+1])/reference.shape[1]
            hits = self.matcher._search(local, reference, nearby, .91, lambda: False, max_peaks=2)
            if len(hits) == 1:
                # This is a new pixel match on every capture, not cached truth.
                return True
        hits = self.matcher._search(image, reference, scales, .91, lambda: False, max_peaks=2)
        if len(hits) != 1:
            self._anchors.pop(name, None)
            return False
        self._anchors[name] = image.shape, hits[0]
        return True

    def ready(self, png):
        if png != self._last_png:
            self._last_ready = self._ready(png)
            self._last_png = png
        return self._last_ready

    def _ready(self, png):
        image = _decode(png)
        height, width = image.shape[:2]
        base = height/1080
        scales = np.unique(np.r_[np.geomspace(base*.65, base*1.3, 16), base])
        if not self._feature(image[:round(height*.23), :round(width*.17)], 'star', scales):
            return False
        return self._feature(image[:round(height*.24), round(width*.70):], 'diamond', scales)


_thread_vision = threading.local()


def farm_scene_vision():
    if not hasattr(_thread_vision, 'vision'):
        _thread_vision.vision = FarmSceneVision()
    return _thread_vision.vision

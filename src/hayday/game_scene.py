"""Positive farm HUD evidence, independent of loading artwork and farm layout."""
import json
from functools import lru_cache
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

    def ready(self, png):
        image = _decode(png)
        height, width = image.shape[:2]
        base = height/1080
        scales = np.unique(np.r_[np.geomspace(base*.65, base*1.3, 16), base])
        star = self.matcher._search(image[:round(height*.23), :round(width*.17)],
            self.references['star'], scales, .91, lambda: False, max_peaks=2)
        if len(star) != 1:
            return False
        diamond = self.matcher._search(image[:round(height*.24), round(width*.70):],
            self.references['diamond'], scales, .91, lambda: False, max_peaks=2)
        return len(diamond) == 1


@lru_cache(maxsize=1)
def farm_scene_vision():
    return FarmSceneVision()

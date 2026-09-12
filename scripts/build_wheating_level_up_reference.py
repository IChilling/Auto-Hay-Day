"""Extract fixed level-up controls; level numbers and reward artwork are excluded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT/'images/wheating/level_up.png'
    image = Image.open(source).convert('RGBA')
    output = ROOT/'src/hayday/assets/wheating/level_up'
    output.mkdir(parents=True, exist_ok=True)
    boxes = {
        # Align crop edges to common device-scale sampling grids so thin text
        # is resized with the same phase as its surrounding screenshot.
        'continue': [840, 920, 1080, 1020],
        'level_up': [640, 60, 1260, 220],
        'congratulations': [740, 240, 1160, 320],
    }
    for name, box in boxes.items():
        image.crop(box).save(output/f'{name}.png')
    manifest = {
        'version': 1,
        'source': 'images/wheating/level_up.png',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_size': list(image.size),
        'provenance': 'Unmodified RGB cutouts from a live BlueStacks capture. '
                      'The changing level number and all reward artwork are excluded.',
        'bounds': [234, 45, 1452, 1028],
        'target': [840, 920, 240, 100],
        'background': [],
        'features': [{'file': name+'.png', 'box': box} for name, box in boxes.items()],
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    build()

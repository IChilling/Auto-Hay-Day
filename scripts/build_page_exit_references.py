"""Preserve observed page headings and exit controls for order recovery."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'achievements': ('artifacts/orders-cow-recovery/milk_141152/0100.png',
                     [(756, 66, 420, 84), (231, 213, 249, 66), (1614, 36, 138, 141)]),
    'county_fair_tutorial': ('artifacts/orders-cow-recovery/fair-before.png',
                            [(654, 72, 612, 81), (702, 273, 159, 33), (1662, 54, 129, 138)]),
    'county_fair': ('artifacts/orders-cow-recovery/fair-main.png',
                   [(159, 201, 906, 30), (1392, 843, 234, 30), (1776, 33, 114, 111)]),
    'farm_expansion': ('artifacts/orders-cow-recovery/accidental-popup.png',
                       [(618, 63, 690, 111), (1200, 294, 420, 135), (1605, 36, 138, 141)]),
    'catalog': ('artifacts/orders-cow-recovery/grouped-pens/20261001T161556_977074Z/0018.png',
                [(537, 209, 350, 66), (501, 514, 278, 62), (1717, 76, 111, 114)]),
}


def build():
    fixtures = ROOT/'tests/fixtures/orders'
    outputs = [ROOT/'images/page_exits', ROOT/'src/hayday/assets/page_exits']
    for directory in outputs:
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'size': [1920, 1080], 'pages': {}}
    for kind, (artifact, boxes) in SPECS.items():
        source = fixtures/f'page_{kind}.png'
        if not source.exists():
            source.write_bytes((ROOT/artifact).read_bytes())
        png = source.read_bytes()
        image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
        assert image.shape == (1080, 1920, 3)
        features = []
        for index, (x, y, width, height) in enumerate(boxes):
            if kind in {'farm_expansion', 'catalog'} and index == 2:
                # Same generic exit as Achievements; reuse its original pixels.
                features.append({'file': 'achievements_2.png', 'box': [x, y, width, height]})
                continue
            name = f'{kind}_{index}.png'
            encoded = cv2.imencode('.png', image[y:y+height, x:x+width])[1].tobytes()
            for output in outputs:
                (output/name).write_bytes(encoded)
            features.append({'file': name, 'box': [x, y, width, height]})
        manifest['pages'][kind] = {'source': source.name, 'sha256': hashlib.sha256(png).hexdigest(),
                                   'artifact': artifact, 'features': features}
    for output in outputs:
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()

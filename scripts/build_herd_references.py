"""Crop native feed controls and troughs for the remaining production animals."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    'pig_feed': ('pig-menu.png', (983, 207, 96, 87)),
    'pig_tool': ('pig-menu.png', (735, 273, 123, 146)),
    'pig_trough': ('all-pens.png', (979, 393, 64, 53)),
    'goat_feed': ('feed-mill.png', (290, 582, 97, 91)),
    'goat_feed_inactive': ('goat-menu.png', (1232, 217, 99, 91)),
    'goat_tool': ('goat-menu.png', (1001, 291, 132, 133)),
    'goat_trough': ('all-pens.png', (1333, 373, 70, 59)),
    'sheep_trough': ('winter-cliff.png', (424, 822, 80, 66)),
}


def build():
    root = ROOT/'artifacts/orders-cow-recovery/feed-study'
    outputs = (ROOT/'images/animals/herd', ROOT/'src/hayday/assets/animals/herd')
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'features': {}, 'feed_count': {
        'pig': [-125, -49, 146, 89], 'goat': [-106, -56, 144, 88]}}
    for name, (source, box) in FEATURES.items():
        path = ROOT/'tests/fixtures/orders'/('herd_'+source)
        if not path.exists():
            path.write_bytes((root/source).read_bytes())
        png = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        mask = np.full((h, w), 255, np.uint8)
        if name == 'pig_trough':
            mask[:] = 0
            cv2.fillPoly(mask, [np.array([(3,15),(23,3),(62,19),(56,34),(49,51),(14,39)])], 255)
        elif name == 'goat_trough':
            mask[:] = 0
            cv2.fillPoly(mask, [np.array([(3,28),(32,5),(66,3),(63,32),(48,51),(8,48)])], 255)
        elif name == 'sheep_trough':
            old = cv2.imread(str(ROOT/'images/animals/pen_trough.png'), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(old[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
        encoded = cv2.imencode('.png', np.dstack((crop, mask)))[1].tobytes()
        for output in outputs:
            (output/(name+'.png')).write_bytes(encoded)
        manifest['features'][name] = {'file': name+'.png', 'source': path.name,
            'source_sha256': hashlib.sha256(png).hexdigest(), 'box': box,
            'mask': 'Original BGR pixels; opaque rectangle or trough inclusion polygon.'}
    for output in outputs:
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()

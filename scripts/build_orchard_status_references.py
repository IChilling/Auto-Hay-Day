"""Preserve native fruit and exhausted-tree menu pixels for read-only recognition."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/orders'
SPECS = {
    'saw': ([409, 559, 577, 724], [[416, 670], [440, 652], [443, 635],
        [479, 599], [537, 576], [542, 563], [559, 562], [573, 579], [573, 597],
        [560, 599], [555, 583], [551, 585], [550, 616], [462, 705], [445, 698],
        [438, 720], [420, 711], [411, 688]]),
    'help': ([634, 387, 742, 558], [[639, 400], [736, 391], [733, 486],
        [702, 491], [702, 553], [681, 554], [680, 493], [646, 498]]),
    'arrow': ([483, 637, 611, 737], [[513, 641], [553, 671], [561, 657],
        [606, 726], [603, 732], [541, 716], [549, 701], [490, 694]]),
}


def build():
    source = FIXTURES / 'apple_exhausted.png'
    png = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
    manifest = {'version': 1, 'features': {}, 'species': ['apple', 'cherry'],
                'source': source.relative_to(ROOT).as_posix(),
                'sha256': hashlib.sha256(png).hexdigest(),
                'purpose': 'Read-only deferral. These targets never authorize saw or help input.'}
    outputs = (ROOT / 'images/orchard_status', ROOT / 'src/hayday/assets/orchard_status')
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
    for name, (box, polygon) in SPECS.items():
        x0, y0, x1, y1 = box
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillPoly(mask, [np.array(polygon)-[x0, y0]], 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        data = cv2.imencode('.png', np.dstack((image[y0:y1, x0:x1], mask)))[1].tobytes()
        for path in outputs:
            (path / (name+'.png')).write_bytes(data)
        manifest['features'][name] = {'box': box, 'polygon': polygon}
    item_source = FIXTURES / 'apple_juice_recipe.png'
    item_png = item_source.read_bytes()
    image = cv2.imdecode(np.frombuffer(item_png, np.uint8), 1)
    box = [1053, 554, 1133, 646]
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 110))
            | ((hsv[:, :, 0] > 25) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 90)))
    mask = cv2.dilate(mask.astype(np.uint8)*255, np.ones((3, 3), np.uint8))
    data = cv2.imencode('.png', np.dstack((crop, mask)))[1].tobytes()
    manifest['desired'] = {'source': item_source.relative_to(ROOT).as_posix(), 'box': box,
                           'sha256': hashlib.sha256(item_png).hexdigest()}
    cherry_source = ROOT / 'images/fruit/cherry_item.png'
    cherry = cherry_source.read_bytes()
    manifest['cherry'] = {'source': cherry_source.relative_to(ROOT).as_posix(),
                          'sha256': hashlib.sha256(cherry).hexdigest(),
                          'provenance': 'Native fruit crop and alpha from build_fruit_references.py.'}
    for path in outputs:
        (path / 'apple.png').write_bytes(data)
        (path / 'cherry.png').write_bytes(cherry)
        (path / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()

"""Build finished-product references from our own direct ADB captures."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
captures = ROOT/'images/reference_captures'
output = ROOT/'images/production_ready'
output.mkdir(exist_ok=True)
for name in ('resource_cookie_ready', 'resource_cookie_ready_obscured'):
    data = (captures/f'{name}.png').read_bytes()
    (captures/f'{name}.json').write_text(json.dumps({
        'source': 'Direct BlueStacks ADB screenshot', 'sha256': hashlib.sha256(data).hexdigest(),
        'transformations': 'None; original screenshot pixels',
        'purpose': 'Finished Cookie on Bakery output table, with and without production UI',
    }, indent=2))
image = cv2.imread(str(captures/'resource_cookie_ready.png'))


def feature(name, box, points):
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    cv2.imwrite(str(output/f'{name}.png'), np.dstack((crop, mask)))
    return {'file': name+'.png', 'box': box}


dome = feature('bakery_dome', [888, 513, 62, 92],
               [(32, 2), (39, 4), (37, 17), (50, 25), (60, 43), (45, 48),
                (36, 70), (17, 88), (2, 81), (1, 59), (9, 37), (20, 22)])
barrel = feature('bakery_barrel', [845, 582, 41, 58],
                 [(3, 10), (15, 2), (30, 4), (39, 12), (36, 42), (25, 55), (8, 49), (1, 31)])
cookie = feature('cookie_ready', [939, 664, 41, 29],
                 [(1, 16), (7, 8), (22, 1), (34, 2), (39, 8), (38, 17),
                  (27, 25), (12, 28), (3, 24)])
learned = ROOT/'images/learned_items'
for suffix, target in [('.png', 'cookie_identifier.png'), ('.title.png', 'cookie_title.png')]:
    (output/target).write_bytes((learned/('f0bda6945ba0a2947f9e'+suffix)).read_bytes())
manifest = {'version': 1, 'reference_height': 1080, 'products': [{
    'name': 'Cookie', 'machine': 'Bakery', 'identifier': 'cookie_identifier.png',
    'title': 'cookie_title.png', 'features': {'anchor': dome, 'support': barrel, 'output': cookie},
    'tap': [943, 597], 'source_capture': 'resource_cookie_ready.png',
    'mask': 'Original pixels with polygon alpha masks; surrounding farm and menu pixels excluded.',
}]}
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
installed = ROOT/'src/hayday/assets/production_ready'
installed.mkdir(exist_ok=True)
for path in output.iterdir():
    if path.is_file():
        (installed/path.name).write_bytes(path.read_bytes())
print('Built finished Cookie and independent Bakery features.')

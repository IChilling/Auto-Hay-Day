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
image = cv2.imread(str(captures/'resource_brown_sugar_ready.png'))
rollers = feature('sugar_mill_rollers', [814, 430, 73, 78],
                  [(4, 18), (35, 1), (54, 4), (56, 18), (67, 32), (71, 60),
                   (65, 76), (11, 69), (5, 40)])
sugar_barrel = feature('sugar_mill_barrel', [862, 513, 45, 54],
                      [(4, 11), (20, 1), (41, 8), (44, 23), (37, 48), (11, 49), (1, 35)])
sugar = feature('brown_sugar_ready', [833, 542, 37, 44],
                [(14, 1), (33, 2), (28, 14), (31, 30), (26, 38), (12, 39),
                 (5, 43), (1, 40), (4, 33), (15, 31), (14, 15)])
for suffix, target in [('.png', 'brown_sugar_identifier.png'), ('.title.png', 'brown_sugar_title.png')]:
    (output/target).write_bytes((learned/('f49caaf98109c16ad920'+suffix)).read_bytes())
data = (captures/'resource_brown_sugar_ready.png').read_bytes()
proof = {
    'source': 'Direct MuMu ADB screenshot',
    'original': 'artifacts/orders-truck/delivery-followup/20261001T200818_910759Z/0020.png',
    'sha256': hashlib.sha256(data).hexdigest(),
    'transformations': 'None; original screenshot pixels',
    'purpose': 'Brown Sugar beside its Sugar Mill; Dairy wall is a misleading generic icon match.',
}
(captures/'resource_brown_sugar_ready.json').write_text(json.dumps(proof, indent=2))
manifest['products'].append({
    'name': 'Brown Sugar', 'machine': 'Sugar Mill', 'identifier': 'brown_sugar_identifier.png',
    'title': 'brown_sugar_title.png',
    'features': {'anchor': rollers, 'support': sugar_barrel, 'output': sugar},
    'tap': [839, 491], 'source_capture': 'resource_brown_sugar_ready.png',
    'source_sha256': proof['sha256'],
    'mask': 'Original pixels with polygon alpha masks; surrounding farm pixels excluded.',
})
image = cv2.imread(str(captures/'resource_wool_hat_ready.png'))
loom_frame = feature('loom_frame', [944, 523, 70, 116],
                     [(3, 1), (65, 31), (65, 114), (49, 109), (49, 42), (4, 20)])
loom_cloth = feature('loom_cloth', [909, 574, 94, 81],
                     [(2, 20), (50, 1), (91, 24), (50, 52), (55, 79), (19, 58)])
hat = feature('blue_woolly_hat_ready', [943, 657, 38, 36],
              [(8, 1), (15, 1), (16, 8), (27, 16), (30, 28), (36, 32),
               (30, 35), (2, 35), (0, 29), (5, 23), (8, 9)])
for suffix, target in [('.png', 'blue_woolly_hat_identifier.png'),
                       ('.title.png', 'blue_woolly_hat_title.png')]:
    (output/target).write_bytes((learned/('f0fc912ca522841ae35d'+suffix)).read_bytes())
data = (captures/'resource_wool_hat_ready.png').read_bytes()
proof = {
    'source': 'Direct MuMu ADB screenshot',
    'original': 'HayDayAutomation/diagnostics/orders/20261001T201723_b60f2998/0025_final.png',
    'sha256': hashlib.sha256(data).hexdigest(),
    'transformations': 'None; original screenshot pixels',
    'purpose': 'Blue Woolly Hat waiting beside its empty Loom.',
}
(captures/'resource_wool_hat_ready.json').write_text(json.dumps(proof, indent=2))
manifest['products'].append({
    'name': 'Blue Woolly Hat', 'machine': 'Loom', 'identifier': 'blue_woolly_hat_identifier.png',
    'title': 'blue_woolly_hat_title.png',
    'features': {'anchor': loom_frame, 'support': loom_cloth, 'output': hat},
    'tap': [969, 606], 'source_capture': 'resource_wool_hat_ready.png',
    'source_sha256': proof['sha256'],
    'mask': 'Original pixels with polygon alpha masks; surrounding farm pixels excluded.',
})
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
installed = ROOT/'src/hayday/assets/production_ready'
installed.mkdir(exist_ok=True)
for path in output.iterdir():
    if path.is_file():
        (installed/path.name).write_bytes(path.read_bytes())
print('Built finished Cookie/Bakery, Brown Sugar/Sugar Mill and Blue Woolly Hat/Loom references.')

"""Build fishing references from the recorded MuMu session's original pixels."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/orders/fishing_menu.png'


def build():
    image = cv2.imread(str(SOURCE))
    features = {
        'red_lure': (690, 389, 849, 531),
        'blue_lure': (653, 124, 792, 265),
        'home': (25, 910, 174, 1048),
    }
    manifest = {'version': 1, 'source': str(SOURCE.relative_to(ROOT)),
                'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'reference_height': 1080, 'features': features,
                'stock_box': [642, 342, 136, 78], 'tool_point': [770, 455]}
    production = {
        'workbench_title': ('fishing_workbench_empty.png', (1243, 380, 1646, 443)),
        'free_red_recipe': ('fishing_free_lure_recipe.png', (1120, 500, 1302, 641)),
        'red_lure_title': ('fishing_lure_location.png', (1032, 372, 1447, 445)),
        'queued_red_lure': ('fishing_lure_queued.png', (1040, 700, 1170, 815)),
        'workbench_anchor': ('fishing_workbench_empty.png', (1025, 420, 1187, 541)),
    }
    manifest['production'] = {
        name: {'source': 'tests/fixtures/orders/'+filename, 'box': box,
               'sha256': hashlib.sha256((SOURCE.parent/filename).read_bytes()).hexdigest()}
        for name, (filename, box) in production.items()
    }
    for target in (ROOT/'images/fishing', ROOT/'src/hayday/assets/fishing'):
        target.mkdir(parents=True, exist_ok=True)
        for name, (x, y, right, bottom) in features.items():
            crop = image[y:bottom, x:right].copy()
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            if name == 'red_lure':
                mask = cv2.inRange(hsv, np.array([0, 100, 60]), np.array([25, 255, 255]))
                # Exclude the adjacent gold drag arrow.
                polygon = np.zeros(mask.shape, np.uint8)
                points = np.array([(48,2),(77,9),(88,41),(78,57),(86,70),(121,57),
                                   (157,83),(153,122),(139,141),(126,139),(132,109),
                                   (115,95),(95,97),(86,116),(45,126),(13,111),
                                   (1,87),(12,64),(45,39),(58,26),(37,17)])
                cv2.fillPoly(polygon, [points], 255)
                mask &= polygon
            elif name == 'blue_lure':
                mask = cv2.inRange(hsv, np.array([80, 170, 65]), np.array([110, 255, 255]))
            else:
                mask = np.zeros(crop.shape[:2], np.uint8)
                cv2.ellipse(mask, (74, 69), (69, 65), 0, 0, 360, 255, -1)
            cv2.imwrite(str(target/f'{name}.png'), np.dstack((crop, mask)))
        (target/'fish_item.png').write_bytes((ROOT/'tests/fixtures/orders/fish_item.png').read_bytes())
        for name, (filename, (x, y, right, bottom)) in production.items():
            crop = cv2.imread(str(SOURCE.parent/filename))[y:bottom, x:right].copy()
            mask = np.full(crop.shape[:2], 255, np.uint8)
            if name == 'workbench_title':
                # Only the outlined letters, excluding the moving cloud/trees.
                mask = ((crop.min(axis=2) > 215) | (crop.max(axis=2) < 65)).astype(np.uint8)*255
                white = (crop.min(axis=2) > 215).astype(np.uint8)*255
                mask &= cv2.dilate(white,np.ones((5,5),np.uint8))
            elif name == 'queued_red_lure':
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0, 100, 65]), np.array([25, 255, 255]))
                polygon = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(polygon, [np.array([(19,0),(47,3),(55,20),(45,36),(51,51),
                    (80,49),(104,48),(122,66),(116,90),(104,107),(94,105),(100,79),
                    (85,72),(70,76),(60,91),(28,95),(4,82),(0,68),(6,50),(28,32),
                    (31,19),(14,13)])], 255)
                mask &= polygon
            elif name == 'workbench_anchor':
                # The producer menu overlaps the lower-left corner. Keep the
                # wooden fish sign, excluding the arrow and surrounding foliage
                # so this reference also identifies the closed workbench.
                mask[:] = 0
                polygon = np.array([(34,18),(66,4),(94,8),(120,17),
                    (140,40),(155,67),(156,88),(143,113),(106,117),
                    (94,95),(70,84),(47,68),(34,50),(12,49),(8,29)])
                cv2.fillPoly(mask,[polygon],255)
            else:
                # The popup's straight cream edge and complete title/duration;
                # leave out the variable tackle-box stock to its right.
                mask[:, :8] = 0
            cv2.imwrite(str(target/f'{name}.png'), np.dstack((crop, mask)))
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()

"""Build exact references for the observed server-not-responding dialog."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root/'images/wheating/connection_server_not_responding.png'
png = source.read_bytes()
image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
destination = root/'src/hayday/assets/wheating/recovery'
destination.mkdir(parents=True, exist_ok=True)
features = []
for name, box, threshold in (
    ('server_anchor', (1088, 24, 1280, 144), 95),
    ('server_title', (620, 24, 1290, 144), 95),
    ('server_body_1', (260, 626, 1650, 683), 185),
    ('server_body_2', (756, 685, 1150, 738), 185),
    ('server_retry', (780, 948, 1140, 1040), 95),
):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    alpha = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < threshold).astype(np.uint8)*255
    alpha = cv2.dilate(alpha, np.ones((5, 5), np.uint8))
    assert (alpha > 0).sum() > 40
    cv2.imwrite(str(destination/(name+'.png')), np.dstack((crop, alpha)))
    features.append({'file': name+'.png', 'box': box,
        'mask': f'Original grayscale < {threshold}; dilated 5x5; original RGB unchanged.'})
spec = {'source': source.name, 'source_sha256': hashlib.sha256(png).hexdigest(),
        'source_size': [image.shape[1], image.shape[0]], 'features': features,
        'bounds': [173, 2, 1549, 1066], 'target': [780, 948, 360, 92],
        'background': []}
for x, y, w, h in ((360, 440, 120, 90), (1460, 780, 100, 60)):
    spec['background'].append({'box': [x, y, w, h],
        'color_bgr': np.median(image[y:y+h, x:x+w], axis=(0, 1)).tolist()})
(destination/'manifest.json').write_text(json.dumps(spec, indent=2))

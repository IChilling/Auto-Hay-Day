"""Build fixed badge/lettering references, excluding buyer names and prices."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT/'images/wheating/sold_receipt_reference.png'
    image = cv2.imread(str(source))
    root = ROOT/'src/hayday/assets/wheating/sold_receipt'
    root.mkdir(exist_ok=True)
    rooster, label = (88, 47, 150, 149), (60, 4, 164, 72)
    x1, y1, x2, y2 = rooster
    mask = cv2.inRange(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV),
                       (15, 140, 130), (28, 240, 220))
    cv2.imwrite(str(root/'rooster.png'), mask)
    x1, y1, x2, y2 = label
    crop = image[y1:y2, x1:x2]
    dark = (crop.max(axis=2) < 95).astype(np.uint8)*255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alpha = np.zeros_like(dark)
    cv2.drawContours(alpha, contours, -1, 255, cv2.FILLED)
    cv2.imwrite(str(root/'label.png'), np.dstack((crop, alpha)))
    (root/'manifest.json').write_text(json.dumps(dict(version=1, source=source.name,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rooster=rooster,
        label=label, reference_height=1080), indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    build()

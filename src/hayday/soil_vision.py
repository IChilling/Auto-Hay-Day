"""Shared soil texture references for both isometric furrow orientations."""
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import _decode


def soil_references():
    root = Path(__file__).parent/'assets/wheating'
    result = {}
    for name in ('soil_texture', 'soil_orientation_1', 'soil_orientation_2'):
        reference = _decode((root/f'{name}.png').read_bytes(), True)
        if name != 'soil_texture':
            # The supplied crops include white/grass corners. Match their soil
            # pixels only, so those backgrounds cannot dictate field identity.
            hsv = cv2.cvtColor(reference[:, :, :3], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (7, 55, 35), (27, 255, 255))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            if reference.shape[2] == 4:
                mask = cv2.bitwise_and(mask, reference[:, :, 3])
                reference = reference.copy()
            else:
                reference = cv2.cvtColor(reference, cv2.COLOR_BGR2BGRA)
            reference[:, :, 3] = mask
        result[name] = reference
    return result


def distinct_soil_targets(targets):
    """Merge overlapping reference matches without losing adjacent tiles."""
    result = []
    for target in sorted(targets, key=lambda t: -t.score):
        if all(np.linalg.norm(np.subtract(target.center, old.center))
               > min(target.width, old.width)*.30 for old in result):
            result.append(target)
    return tuple(result)

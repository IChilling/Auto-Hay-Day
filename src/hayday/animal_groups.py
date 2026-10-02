"""Bound collection to adjacent, independently recognized same-species pens."""
from dataclasses import replace

import cv2
import numpy as np

from hayday.animals import SheepVision


def adjacent_pens(pens, seed):
    """Return the seed's connected group without bridging unverified pens."""
    remaining = list(pens)
    group = [seed]
    remaining.remove(seed)
    while remaining:
        connected = []
        for candidate in remaining:
            for neighbor in group:
                first, second = candidate[2], neighbor[2]
                gap = min(-cv2.pointPolygonTest(np.array(poly, np.float32), tuple(map(float, p)), True)
                          for poly, points in ((first, second), (second, first)) for p in points)
                if gap <= min(candidate[0].width, neighbor[0].width)*.85:
                    connected.append(candidate)
                    break
        if not connected:
            break
        group.extend(connected)
        remaining = [pen for pen in remaining if pen not in connected]
    return tuple(group)


def grouped_harvest(guide, pens, seed, width_factor):
    """An enabled tool is global readiness, not proof that the seed is ready."""
    group = adjacent_pens(pens, seed)
    paths = tuple(SheepVision.sweep(polygon, landmark.width*width_factor)
                  for landmark, _, polygon in group)
    return replace(guide, target=guide.target or seed[1], fruit_features=tuple(pen[1] for pen in group),
                   pen_polygon=seed[2], pen_polygons=tuple(pen[2] for pen in group),
                   sweep_path=paths[0], sweep_paths=paths)

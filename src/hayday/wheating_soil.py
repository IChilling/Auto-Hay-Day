"""Independent lattice-cell inspection for partially obscured wheat fields."""
from __future__ import annotations

import cv2
import numpy as np

from hayday.soil_vision import distinct_soil_targets


def empty_tiles(vision, frame, plot, limit):
    if not getattr(vision, 'full_outline', False) or limit < 1:
        return []
    image, hsv = vision._frame(frame.png), vision._hsv(frame.png)
    scale = plot.width/117
    matches = distinct_soil_targets([
        hit for ref in vision._soil_textures
        for hit in vision._matcher._search(image, ref, np.array([scale]), .91, vision.cancel, 32)])
    offsets = []
    for first in matches:
        for second in matches:
            x, y = abs(first.center[0]-second.center[0]), abs(first.center[1]-second.center[1])
            if plot.width*.4 < x < plot.width*.51 and plot.height*.34 < y < plot.height*.52:
                offsets.append((x, y))
    dx, dy = np.median(offsets, axis=0) if len(offsets) >= 4 else ((plot.width-10)/2, (plot.height-10)/2)
    if min(dx, dy) < 6:
        return [plot.center]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_scale = dx/53.5
    furrows = []
    for texture in vision._soil_textures:
        color, _ = vision._matcher._scaled(texture, float(texture_scale))
        height, width = color.shape[:2]
        rx, ry = max(3, round(12*texture_scale)), max(3, round(6*texture_scale))
        furrows.append(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)[
            height//2-ry:height//2+ry, width//2-rx:width//2+rx])
    # Inspect the visible lattice independently. Rejected cells, grass holes,
    # growing neighbors and the picker cannot cut off other visible soil.
    left, right = frame.width*.10, frame.width*.90
    top, bottom = frame.height*.15, frame.height*.90
    corners = np.array([(left, top), (right, top), (left, bottom), (right, bottom)])
    delta = (corners-plot.center)/(dx, dy)
    ij = np.c_[delta[:, 0]+delta[:, 1], delta[:, 1]-delta[:, 0]]/2
    lo, hi = np.floor(ij.min(axis=0)).astype(int), np.ceil(ij.max(axis=0)).astype(int)
    cells = [(i, j) for i in range(lo[0], hi[0]+1) for j in range(lo[1], hi[1]+1)]
    cells.sort(key=lambda cell: (abs(cell[0])+abs(cell[1]), cell))
    points = []
    for i, j in cells:
        if vision.cancel():
            return []
        x, y = round(plot.center[0]+(i-j)*dx), round(plot.center[1]+(i+j)*dy)
        if not (left < x < right and top < y < bottom):
            continue
        pixels = vision.tile_pixels(hsv, (x, y), dx, dy)
        if pixels is None:
            continue
        hue, sat, value = pixels.T
        soil = ((hue >= 8) & (hue <= 18) & (sat >= 55) & (value < 250)).mean()
        if soil < .72:
            continue
        rx, ry = max(4, round(24*texture_scale)), max(4, round(13*texture_scale))
        interior = gray[max(0, y-ry):y+ry, max(0, x-rx):x+rx]
        scores = [float(cv2.matchTemplate(interior, texture, cv2.TM_CCOEFF_NORMED).max())
                  for texture in furrows
                  if all(a >= b for a, b in zip(interior.shape, texture.shape, strict=True))]
        texture = max(scores, default=0.)
        # Furrows establish each tile independently; brown pens and roads do
        # not qualify merely by matching soil color or neighboring a field.
        if texture < (.82 if soil >= .88 else .90):
            continue
        points.append((x, y))
        if len(points) >= min(limit, 512):
            break
    return points

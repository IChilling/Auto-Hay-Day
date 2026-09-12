"""Strict shop numerals, including the visible quantity suffix x."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from hayday.quantities import _references, glyph_components


@lru_cache(maxsize=1)
def references():
    root = Path(__file__).parent/'assets/wheating/numbers'
    entries = json.loads((root/'manifest.json').read_text('utf-8'))
    return tuple(_references('outlined'))+tuple(
        (item['glyph'], cv2.imdecode(np.frombuffer((root/item['file']).read_bytes(), np.uint8), 0))
        for item in entries)


def ranking(glyph, singles=False, templates=None):
    scores = {}
    a = cv2.dilate(glyph, np.ones((3, 3), np.uint8)).astype(float)/255
    for label, ref in references() if templates is None else templates:
        if singles and (len(label) != 1 or not label.isdecimal()):
            continue
        b = cv2.dilate(ref, np.ones((3, 3), np.uint8)).astype(float)/255
        score = float(2*np.minimum(a, b).sum()/max(1, a.sum()+b.sum()))
        scores[label] = max(scores.get(label, 0), score)
    return sorted(scores.items(), key=lambda entry: entry[1], reverse=True)


def white_components(png, box, *, image=None):
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in box)):
        return ()
    try:
        if image is None:
            image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
    except (TypeError, ValueError, cv2.error):
        return ()
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return ()
    x, y, width, height = box
    if min(x, y) < 0 or min(width, height) <= 0 or x+width > image.shape[1] or y+height > image.shape[0]:
        return ()
    hsv = cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 210), (179, 25, 255))
    mask |= cv2.inRange(hsv, (135, 35, 180), (175, 175, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = []
    for index, (gx, gy, w, h, area) in enumerate(stats[1:count], 1):
        if h < height*.38 or area < height**2*.015 or w > h*1.2:
            continue
        piece = (labels[gy:gy+h, gx:gx+w] == index).astype(np.uint8)*255
        factor = min(28/w, 36/h)
        shape = max(1, round(w*factor)), max(1, round(h*factor))
        canvas = np.zeros((40, 32), np.uint8)
        left, top = (32-shape[0])//2, (40-shape[1])//2
        canvas[top:top+shape[1], left:left+shape[0]] = cv2.resize(piece, shape, interpolation=cv2.INTER_AREA)
        parts.append(((int(x+gx), int(y+gy), int(w), int(h)), canvas))
    return tuple(sorted(parts, key=lambda part: part[0][0]))


@lru_cache(maxsize=1)
def white_references():
    root = Path(__file__).parent/'assets/wheating/numbers'
    manifest = root/'white_manifest.json'
    if not manifest.exists():
        return ()
    return tuple((item['glyph'], cv2.imdecode(np.frombuffer((root/item['file']).read_bytes(), np.uint8), 0))
                 for item in json.loads(manifest.read_text('utf-8')))


def glyph_holes(glyph, threshold=128):
    contours, hierarchy = cv2.findContours((glyph >= threshold).astype(np.uint8)*255,
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(h[3] >= 0 and cv2.contourArea(c) >= 8
               for c, h in zip(contours, hierarchy[0], strict=True))


@lru_cache(maxsize=1)
def white_holes():
    counts = {}
    for label, glyph in white_references():
        counts.setdefault(label, set()).add(glyph_holes(glyph))
    return {label: next(iter(values)) for label, values in counts.items() if len(values) == 1}


def read_white(png, box, suffix, *, image=None):
    parts, refs = white_components(png, box, image=image), white_references()
    if not refs or not 1 <= len(parts) <= 5:
        return None
    heights = [b[3] for b, _ in parts]
    bottoms = [b[1]+b[3] for b, _ in parts]
    if min(heights) < max(heights)*.65 or max(bottoms)-min(bottoms) > max(heights)*.25:
        return None
    text = ''
    for _, glyph in parts:
        scores = ranking(glyph, templates=refs)
        margin = scores[0][1]-scores[1][1]
        if scores[0][1] >= .94 and margin < .075:
            # Pink 0/8 interiors can overlap strongly after dilation. Use
            # stable enclosed holes as independent evidence; never promote a
            # lower-scoring digit or lower the visual-confidence threshold.
            holes = glyph_holes(glyph, 96)
            expected = white_holes()
            if holes == glyph_holes(glyph, 160) and expected.get(scores[0][0]) == holes:
                compatible = [score for label, score in scores[1:] if expected.get(label) == holes]
                margin = scores[0][1]-max(compatible, default=0.)
        if scores[0][1] < .90 or margin < .075:
            return None
        text += scores[0][0]
    if suffix:
        if not text.endswith('x'):
            return None
        text = text[:-1]
    if not text.isascii() or not text.isdecimal() or len(text) > 1 and text.startswith('0'):
        return None
    return int(text)


def touching_digits(png, bounds):
    """Split thin outline bridges only when every resulting digit is decisive.

    Wheat counts change each harvest. Their touching outlines cannot require a
    separate reference for every possible multi-digit inventory count.
    """
    image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
    x, y, width, height = bounds
    if width < height*.75 or width > height*3:
        return None
    raw = (image[y:y+height, x:x+width].max(axis=2) < 90).astype(np.uint8)*255
    minimum, maximum = max(3, round(height*.16)), round(height*.75)
    cuts = {i for i in range(minimum, width-minimum+1) if (raw[:, i] > 0).mean() < .31}
    cuts.add(width)

    @lru_cache(None)
    def parse(start, remaining):
        options = []
        if not remaining:
            return options
        for end in sorted(cuts):
            if not minimum <= end-start <= maximum:
                continue
            piece = raw[:, start:end]
            yy, xx = np.nonzero(piece)
            if not len(xx) or np.ptp(yy)+1 < height*.8:
                continue
            piece = piece[yy.min():yy.max()+1, xx.min():xx.max()+1]
            h, w = piece.shape
            scale = min(28/w, 36/h)
            shape = max(1, round(w*scale)), max(1, round(h*scale))
            canvas = np.zeros((40, 32), np.uint8)
            left, top = (32-shape[0])//2, (40-shape[1])//2
            canvas[top:top+shape[1], left:left+shape[0]] = cv2.resize(piece, shape, interpolation=cv2.INTER_AREA)
            scores = ranking(canvas, singles=True)
            if scores[0][1] < .90 or scores[0][1]-scores[1][1] < .10:
                continue
            digit, score = scores[0]
            tails = [('', 1.)] if end == width else parse(end, remaining-1)
            options.extend((digit+tail, min(score, confidence)) for tail, confidence in tails)
        return sorted(options, key=lambda entry: -entry[1])[:8]

    choices = {}
    for value, score in parse(0, 4):
        if len(value) > 1:
            choices[value] = max(score, choices.get(value, 0))
    ordered = sorted(choices.items(), key=lambda entry: -entry[1])
    if not ordered or len(ordered) > 1 and ordered[0][1]-ordered[1][1] < .075:
        return None
    return ordered[0][0]


def read_number(png, box, suffix=False, *, image=None):
    # White interiors stay separate even when neighboring black outlines join.
    white = read_white(png, box, suffix, image=image)
    if white is not None:
        return white
    parts = glyph_components(png, box, 'outlined')
    if not 1 <= len(parts) <= 5:
        return None
    heights = [bounds[3] for bounds, _ in parts]
    bottoms = [bounds[1]+bounds[3] for bounds, _ in parts]
    if min(heights) < max(heights)*.65 or max(bottoms)-min(bottoms) > max(heights)*.25:
        return None
    text = ''
    for bounds, glyph in parts:
        scores = ranking(glyph)
        if scores[0][1] < .9 or scores[0][1]-scores[1][1] < .065:
            split = touching_digits(png, bounds)
            if split is None:
                return None
            text += split
        else:
            text += scores[0][0]
    if suffix:
        if not text.endswith('x'):
            return None
        text = text[:-1]
    if not text.isascii() or not text.isdecimal() or len(text) > 1 and text.startswith('0'):
        return None
    return int(text)

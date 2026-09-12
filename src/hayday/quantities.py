"""Conservative digit recognition from original Hay Day quantity glyphs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Quantity:
    available: int
    required: int


def glyph_components(png, box, font='quantity'):
    """Return source-pixel bounds and normalized shapes, with no OCR guesses."""
    if font not in ('quantity', 'outlined'):
        return ()
    if not isinstance(box, (tuple, list)) or len(box) != 4 or not all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in box
    ):
        return ()
    try:
        picture = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    except (TypeError, ValueError, cv2.error):
        return ()
    if picture is None:
        return ()
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > picture.shape[1] or y+h > picture.shape[0]:
        return ()
    crop = picture[y:y+h, x:x+w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if font == 'outlined':
        mask = (hsv[:, :, 2] < 90).astype(np.uint8)*255
    else:
        mask = (((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 32) & (hsv[:, :, 1] > 75)
                 & (hsv[:, :, 2] < 185)) | ((hsv[:, :, 0] < 15) & (hsv[:, :, 1] > 155)
                                          & (hsv[:, :, 2] > 100))).astype(np.uint8)*255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    pieces = []
    for index in range(1, n):
        gx, gy, gw, gh, area = stats[index]
        if gh < h*(.38 if font == 'outlined' else .30) or area < h*h*.015:
            continue
        if font == 'outlined' and any(
            other != index and ox <= gx and oy <= gy and ox+ow >= gx+gw and oy+oh >= gy+gh
            for other, (ox, oy, ow, oh, _) in enumerate(stats[1:], 1)
        ):
            continue
        # The centers of outlined 0 and 6 can be separate black components.
        # Preserve those inside the outer glyph, without treating them as digits.
        piece = (mask[gy:gy+gh, gx:gx+gw] if font == 'outlined' else
                 (labels[gy:gy+gh, gx:gx+gw] == index).astype(np.uint8)*255)
        canvas = np.zeros((40, 32), np.uint8)
        factor = min(28/gw, 36/gh)
        shape = max(1, round(gw*factor)), max(1, round(gh*factor))
        resized = cv2.resize(piece, shape, interpolation=cv2.INTER_AREA)
        left, top = (32-shape[0])//2, (40-shape[1])//2
        canvas[top:top+shape[1], left:left+shape[0]] = resized
        pieces.append(((int(x+gx), int(y+gy), int(gw), int(gh)), canvas))
    return tuple(sorted(pieces, key=lambda piece: piece[0][0]))


def glyphs(png, box, font='quantity'):
    """Compatibility helper for the normalized per-character training images."""
    return tuple(image for _, image in glyph_components(png, box, font))


@lru_cache(maxsize=2)
def _references(font='quantity'):
    if font not in ('quantity', 'outlined'):
        return ()
    project = Path(__file__).resolve().parents[2]
    root = (project / 'images' / 'quantities' if (project / 'pyproject.toml').is_file()
            else Path(__file__).parent / 'assets' / 'quantities')
    try:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or not isinstance(manifest.get('features'), list):
            return ()
        references = []
        for entry in manifest['features']:
            if not isinstance(entry, dict):
                return ()
            if entry.get('font', 'quantity') != font:
                continue
            name, label = entry['file'], entry['glyph']
            if (not isinstance(name, str) or Path(name).name != name
                    or not isinstance(label, str) or not label
                    or (font == 'quantity' and (len(label) != 1 or label not in '0123456789/'))
                    or (font == 'outlined' and (len(label) > 4 or not label.isascii() or not label.isdecimal()))):
                return ()
            template = cv2.imdecode(np.frombuffer((root/name).read_bytes(), np.uint8), 0)
            if template is None or template.shape != (40, 32) or np.count_nonzero(template) < 25:
                return ()
            references.append((label, template))
        return tuple(references)
    except (OSError, ValueError, KeyError, TypeError, cv2.error):
        return ()


def _read_text(png, box, font, minimum, maximum):
    references = _references(font)
    components = glyph_components(png, box, font)
    if not references or not minimum <= len(components) <= maximum:
        return None
    bottoms = [bounds[1]+bounds[3] for bounds, _ in components]
    heights = [bounds[3] for bounds, _ in components]
    if (max(bottoms)-min(bottoms) > max(3, np.median(heights)*.18)
            or min(heights) < max(heights)*.70):
        return None
    text = ''
    for _, part in components:
        scores = {}
        kernel = np.ones((3, 3), np.uint8)
        # A one-pixel tolerance after normalization handles anti-aliasing of
        # the thin black outline, while retaining shape and label separation.
        a = (cv2.dilate(part, kernel) if font == 'outlined' else part).astype(float)/255
        for label, template in references:
            if template is None or template.shape != part.shape:
                continue
            b = (cv2.dilate(template, kernel) if font == 'outlined' else template).astype(float)/255
            # Shape agreement, not color or a guessed OCR substitution.
            score = float(2*np.minimum(a,b).sum() / max(1, a.sum()+b.sum()))
            scores[label] = max(scores.get(label, 0), score)
        ranking = sorted(scores.items(), key=lambda p: p[1], reverse=True)
        if not ranking or ranking[0][1] < .88 or (len(ranking)>1 and ranking[0][1]-ranking[1][1] < .07):
            return None
        text += ranking[0][0]
    return text


def read_quantity(png: bytes, box: tuple[int, int, int, int]) -> Quantity | None:
    text = _read_text(png, box, 'quantity', 3, 9)
    if text is None:
        return None
    if text.count('/') != 1:
        return None
    left, right = text.split('/')
    if (not left.isdecimal() or not right.isdecimal() or not 1 <= int(right) <= 9999
            or len(left) > 1 and left.startswith('0') or len(right) > 1 and right.startswith('0')):
        return None
    return Quantity(int(left), int(right))


def read_count(png: bytes, box: tuple[int, int, int, int], font: str = 'outlined') -> int | None:
    """Read a known outlined stock count; unknown is None, recognized zero is 0.

    The caller supplies the actual count-bubble bounds. Templates for touching
    digits (currently the captured 11) are retained as complete ligatures.
    No digit splitting, font substitution or inferred stock values is attempted.
    """
    if font != 'outlined':
        return None
    text = _read_text(png, box, font, 1, 4)
    if (text is None or not text.isdecimal() or not 1 <= len(text) <= 4
            or len(text) > 1 and text.startswith('0')):
        return None
    return int(text)

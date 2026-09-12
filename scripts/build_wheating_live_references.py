"""Build Wheating-only controls/numerals from its live verification captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hayday.quantities import glyph_components  # noqa: E402
from hayday.wheating_numbers import white_components  # noqa: E402


def build(source):
    root = ROOT/'src/hayday/assets/wheating'
    manifest = json.loads((root/'manifest.json').read_text('utf-8'))
    specs = {
        'composer_adjust': ('live_wheat_10_36.png', (1260, 389, 1476, 430)),
        'composer_title': ('live_composer.png', (1264, 153, 1468, 218)),
        'composer_wheat_title': ('live_wheat_10_36.png', (1296, 152, 1433, 218)),
        'composer_close': ('live_composer.png', (1560, 45, 1705, 188)),
        'composer_submit': ('live_wheat_10_36.png', (1260, 977, 1473, 1031)),
        'silo_tab': ('live_composer.png', (191, 263, 299, 410)),
        'wheat_inventory': ('live_composer.png', (414, 393, 531, 517)),
        'ad_check': ('live_wheat_ad_checked.png', (1084, 474, 1179, 568)),
        'ad_marker': ('live_wheat_advertised.png', (394, 473, 462, 557)),
        'free_ad_in': ('live_ad_cooldown.png', (1180, 661, 1342, 711)),
        'quantity_plus': ('live_wheat_selected.png', (1490, 242, 1609, 362)),
        'price_max': ('live_wheat_selected.png', (1407, 534, 1495, 620)),
        'soil_full': ('live_seed_menu.png', (1000, 577, 1110, 638)),
        'wheat_sale': ('live_wheat_listed.png', (467, 283, 610, 439)),
        'advertise_button_live': ('live_wheat_ad_checked.png', (752, 690, 1145, 744)),
        'ripe_live': ('live_wheat_fields.png', (1002, 486, 1113, 565)),
        'ripe_texture': ('live_before_harvest.png', (1020, 495, 1055, 530)),
        'ripe_texture_dense': ('live_before_harvest.png', (1100, 500, 1140, 545)),
        'ripe_full_a': ('full_mature_wheat.png', (1200, 515, 1240, 560)),
        'ripe_full_b': ('full_mature_wheat.png', (1000, 670, 1040, 715)),
        'soil_texture': ('live_wheat_fields.png', (975, 460, 1025, 495)),
        'sold_live': ('live_sold.png', (474, 266, 574, 331)),
        'empty_outline': ('replant_picker.png', (800, 636, 930, 713)),
    }
    for name, (filename, box) in specs.items():
        path = source/filename
        crop = Image.open(path).convert('RGBA').crop(box)
        mask = Image.new('L', crop.size, 255)
        if name == 'empty_outline':
            hsv = cv2.cvtColor(np.array(crop)[:, :, :3], cv2.COLOR_RGB2HSV)
            white = cv2.inRange(hsv, (0, 0, 205), (179, 55, 255))
            mask = Image.fromarray(cv2.dilate(white, np.ones((7, 7), np.uint8)))
        elif name in {'wheat_inventory', 'wheat_sale', 'ripe_live', 'silo_tab', 'ad_check', 'soil_full'}:
            rgb = np.array(crop)[:, :, :3]
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            if name in {'wheat_inventory', 'wheat_sale', 'ripe_live'}:
                keep = cv2.inRange(hsv, (17, 80, 120), (42, 255, 255))
                if name == 'wheat_inventory':
                    # Three-digit stock counts cover more of the lower-right
                    # icon than the two-digit count in this source capture.
                    keep[52:, 48:] = 0
            elif name == 'ad_check':
                keep = cv2.inRange(hsv, (35, 80, 70), (85, 255, 255))
            elif name == 'soil_full':
                keep = cv2.inRange(hsv, (8, 75, 50), (25, 240, 240))
            else:
                keep = np.zeros(hsv.shape[:2], np.uint8)
                keep[4:-7, 10:-13] = 255
            mask = Image.fromarray(keep)
        elif name in {'quantity_plus', 'price_max', 'composer_close'}:
            mask.paste(0, (0, 0, crop.width, crop.height))
            ImageDraw.Draw(mask).ellipse((5, 5, crop.width-5, crop.height-5), fill=255)
        elif name in {'advertise_button_live', 'composer_submit', 'sold_live'}:
            rgb = np.array(crop)[:, :, :3]
            dark = (rgb.max(axis=2) < 95).astype(np.uint8)*255
            contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            keep = np.zeros_like(dark)
            cv2.drawContours(keep, contours, -1, 255, cv2.FILLED)
            mask = Image.fromarray(keep)
        crop.putalpha(mask)
        crop.save(root/f'{name}.png')
        manifest['features'][name] = {'file': f'{name}.png', 'source': filename,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'box': box, 'reference_height': 1080}
    # New number templates stay isolated from the other agent's quantity reader.
    numbers = root/'numbers'
    numbers.mkdir(exist_ok=True)
    labels = [
        ('live_composer.png', (496, 276, 80, 84), '27'),
        ('live_composer.png', (726, 276, 82, 84), '21'),
        ('live_composer.png', (949, 276, 75, 84), '19'),
        ('live_composer.png', (498, 450, 80, 84), '16'),
        ('live_composer.png', (949, 450, 75, 84), '14'),
        ('live_composer.png', (498, 628, 80, 84), '12'),
        ('live_composer.png', (498, 808, 80, 80), '10'),
        ('live_composer.png', (1257, 266, 95, 73), '0x'),
        ('live_wheat_selected.png', (1257, 266, 95, 73), '8x'),
        ('live_wheat_10_36.png', (1257, 266, 95, 73), '10x'),
        ('live_wheat_max.png', (1338, 439, 78, 80), '28'),
        ('live_wheat_10_36.png', (1338, 439, 78, 80), '36'),
        ('stock_57.png', (654, 394, 125, 82), '57'),
        ('shop_stock_109.png', (471, 286, 24, 69), '1'),
    ]
    entries = []
    for index, (filename, box, text) in enumerate(labels):
        parts = glyph_components((source/filename).read_bytes(), box, 'outlined')
        glyph_labels = [text] if len(parts) == 1 else list(text)
        if len(parts) != len(glyph_labels):
            raise ValueError(f'{filename}: expected {text}, got {len(parts)} components')
        for sub, (label, (_, glyph)) in enumerate(zip(glyph_labels, parts, strict=True)):
            name = f'{index}_{sub}_{label}.png'
            Image.fromarray(glyph).save(numbers/name)
            entries.append({'file': name, 'glyph': label})
    (numbers/'manifest.json').write_text(json.dumps(entries, indent=2)+'\n', 'utf-8')
    white_entries = []
    white_labels = [entry for entry in labels if entry[0] != 'stock_57.png']
    white_labels += [('shop_stock_79.png', (456, 274, 137, 87), '79'),
                     ('shop_stock_79.png', (860, 96, 72, 58), '350')]
    for index, (filename, box, text) in enumerate(white_labels):
        parts = white_components((source/filename).read_bytes(), box)
        if len(parts) != len(text):
            raise ValueError(f'{filename}: expected separate white {text}, got {len(parts)} components')
        for sub, (label, (_, glyph)) in enumerate(zip(text, parts, strict=True)):
            name = f'white_{index}_{sub}_{label}.png'
            Image.fromarray(glyph).save(numbers/name)
            white_entries.append({'file': name, 'glyph': label})
    (numbers/'white_manifest.json').write_text(json.dumps(white_entries, indent=2)+'\n', 'utf-8')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?', default=ROOT/'images/wheating')
    build(parser.parse_args().source)

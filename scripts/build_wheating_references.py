"""Extract original reference pixels for the separate Wheating workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def build(shop, overview, edit):
    output = ROOT / 'src/hayday/assets/wheating'
    output.mkdir(parents=True, exist_ok=True)
    captures = ROOT / 'images/reference_captures'
    specs = {
        'shop': (shop, (0, 0, 225, 223)),
        'shop_header': (overview, (832, 64, 1278, 132)),
        'shop_close': (overview, (1492, 30, 1629, 176)),
        'empty_sale': (overview, (422, 637, 609, 737)),
        'sold': (overview, (438, 250, 566, 319)),
        'edit_title': (edit, (800, 164, 1020, 231)),
        'edit_close': (edit, (1125, 68, 1257, 207)),
        'newspaper': (edit, (1021, 465, 1102, 550)),
        'advertise_button': (edit, (728, 676, 1091, 723)),
        'advertise_now': (edit, (698, 500, 963, 558)),
        'wheat_seed': (captures/'farming_seed_menu.png', (773, 417, 913, 566)),
        'wheat_ripe': (captures/'farming_corn_harvested.png', (936, 777, 979, 832)),
        'soil': (captures/'farming_seed_menu.png', (885, 665, 967, 718)),
    }
    manifest = {'version': 1, 'features': {}, 'provenance':
                'Original RGB pixels from user references and existing farm captures; alpha masks exclude surroundings. No farm coordinates are input targets.'}
    for name, (source, box) in specs.items():
        source = Path(source)
        crop = Image.open(source).convert('RGBA').crop(box)
        rgb = np.array(crop)[:, :, :3]
        mask = Image.new('L', crop.size, 255)
        draw = ImageDraw.Draw(mask)
        if name == 'shop':
            mask.paste(0, (0, 0, crop.width, crop.height))
            draw.polygon([(43, 26), (129, 0), (224, 42), (208, 64), (52, 57)], fill=255)
            draw.polygon([(58, 113), (91, 147), (181, 124), (168, 183), (108, 204), (38, 162)], fill=255)
        elif name.endswith('close'):
            mask.paste(0, (0, 0, crop.width, crop.height))
            draw.ellipse((8, 8, crop.width-8, crop.height-8), fill=255)
        elif name == 'wheat_seed':
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            keep = cv2.inRange(hsv, (17, 65, 120), (42, 255, 255))
            # The picker arrow beneath the icon must not become part of the seed.
            yy, xx = np.indices(keep.shape)
            keep[(yy > 78) & (xx > 93)] = 0
            mask = Image.fromarray(cv2.erode(keep, np.ones((2, 2), np.uint8)))
        elif name == 'soil':
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask = Image.fromarray(cv2.inRange(hsv, (8, 85, 55), (24, 235, 230)))
        crop.putalpha(mask)
        crop.save(output/f'{name}.png')
        manifest['features'][name] = {'file': f'{name}.png', 'source': source.name,
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'box': box}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('shop', 'overview', 'edit'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.shop, args.overview, args.edit)

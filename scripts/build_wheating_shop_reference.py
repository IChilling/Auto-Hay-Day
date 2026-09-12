"""Mask the shared shop base, excluding roof skins, notifications, and scenery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'codex-clipboard-79766fde-faa2-49b9-8583-116c252c51cc.png'


def build():
    source = ROOT/'images/wheating'/SOURCE
    original = Image.open(source).convert('RGBA')
    output = ROOT/'src/hayday/assets/wheating/shop_building'
    output.mkdir(parents=True, exist_ok=True)
    polygons = {
        'counter': [(54, 134), (128, 168), (167, 139), (166, 153),
                    (129, 179), (61, 149)],
        'deck': [(14, 153), (44, 137), (54, 147), (61, 155), (100, 174),
                 (123, 185), (133, 185), (168, 160), (176, 162), (165, 179),
                 (193, 177), (138, 208), (22, 163)],
    }
    combined = Image.new('L', original.size)
    for name, polygon in polygons.items():
        mask = Image.new('L', original.size)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=255)
        # Preserve the original reference bounds so the verified tap and camera
        # anchor remain compatible. Alpha, not the outer box, defines evidence.
        draw.rectangle((96, 76, 145, 132), fill=0)
        cutout = original.copy()
        cutout.putalpha(mask)
        cutout.save(output/f'{name}.png')
        combined.paste(255, mask=mask)
    original.putalpha(combined)
    original.save(output/'shop.png')
    manifest = {
        'version': 2, 'source': f'images/wheating/{SOURCE}',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_size': list(original.size), 'features': polygons,
        'excluded_notification_ltrb': [96, 76, 145, 132],
        'provenance': 'Original RGB pixels; binary polygon alpha masks only. '
                      'Only the counter rim and platform identify the shop. Roof skins, supports, '
                      'notifications, tray contents, surroundings, and decorations are excluded. '
                      'Coordinates describe the reference artwork, never a farm position.',
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    build()

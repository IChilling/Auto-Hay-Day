"""Preserve the Hay Day icon/label spacing on the Smart Downloads launcher."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT/'images/wheating/launcher_smart_downloads.png'
    image = Image.open(source).convert('RGBA')
    output = ROOT/'src/hayday/assets/wheating/launcher'
    output.mkdir(parents=True, exist_ok=True)
    boxes = {'hay_day_icon': [1060, 312, 1160, 412],
             'hay_day_label': [1060, 452, 1160, 494]}
    for name, box in boxes.items():
        image.crop(box).save(output/f'{name}.png')
    manifest = {'version': 1, 'reference_height': image.height,
                'source': source.relative_to(ROOT).as_posix(),
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'features': {name: {'file': name+'.png', 'box': box}
                             for name, box in boxes.items()}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    build()

"""Extract labeled quantity glyphs from inspected original ADB captures."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from hayday.quantities import glyph_components


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root/'images'/'quantities'
    destination.mkdir(parents=True, exist_ok=True)
    samples = [
        ('orders_ready.png', (1273,526,53,35), '1/1'),
        ('orders_ready.png', (1598,526,65,35), '2/1'),
        ('orders_ready.png', (1426,685,67,35), '5/1'),
        ('orders_school_missing.png', (1590,525,80,40), '0/1'),
        ('resource_icecream_recipe.png', (1120,476,97,61), '6/1'),
        ('orders_slot_1.png', (1424,525,69,35), '4/1'),
        ('orders_slot_5.png', (1244,526,82,35), '13/1'),
        ('orders_slot_5.png', (1424,525,69,35), '4/1'),
    ]
    manifest = {
        'version': 1, 'source': 'Original BlueStacks ADB captures',
        'normalization_shape': [40, 32],
        'label_policy': 'Labels are transcribed from inspected live captures; untrained glyphs remain unknown.',
        'features': [],
    }
    samples = [('quantity', source, box, tuple(expected)) for source, box, expected in samples]
    samples += [
        ('outlined', 'farming_seed_menu.png', (590,201,120,78), ('3',)),
        ('outlined', 'farming_seed_menu.png', (355,334,128,85), ('5',)),
        ('outlined', 'farming_seed_menu.png', (682,414,129,83), ('2',)),
        # Their black outlines touch: retain the inspected 11 as a ligature.
        ('outlined', 'farming_seed_menu.png', (231,568,140,81), ('11',)),
        # Exact first digit from that same inspected 11, excluding its neighbor.
        ('outlined', 'farming_seed_menu.png', (280,581,17,52), ('1',)),
        ('outlined', 'farming_seed_menu.png', (478,548,125,81), ('6',)),
        ('outlined', 'resource_icecream_guided.png', (596,260,140,82), ('0',)),
        ('outlined', 'resource_chicken_relocated.png', (1424,181,25,50), ('4',)),
        ('outlined', 'resource_chicken_relocated.png', (1408,182,15,47), ('1',)),
        ('outlined', 'resource_chicken_fed.png', (652,225,32,54), ('9',)),
        ('quantity', 'resource_pancake_eggs_collected.png', (1149,519,70,34), tuple('7/3')),
    ]
    # Exact deterministic resamplings preserve the real outline font at common
    # screenshot resolutions. Very small broken outlines are deliberately absent.
    samples = [(font, source, box, expected, scale)
               for font, source, box, expected in samples
               for scale in ((1,) if box[2] <= 20 else
                             (1, 1.25, 1.5) if source.startswith('resource_chicken_') else
                             (1, .66, .75, 1.25, 1.5) if font == 'outlined' and box[2] > 20 else (1,))]
    for font, source, box, expected, scale in samples:
        source_data = (root/'images'/'reference_captures'/source).read_bytes()
        source_image = cv2.imdecode(np.frombuffer(source_data, np.uint8), cv2.IMREAD_COLOR)
        data = source_data
        reading_box = box
        if scale != 1:
            image = cv2.resize(source_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            data = cv2.imencode('.png', image)[1].tobytes()
            reading_box = tuple(round(value*scale) for value in box)
        parts = glyph_components(data, reading_box, font)
        if len(parts) != len(expected):
            raise ValueError(f'{source}: expected {expected}, got {len(parts)} glyphs')
        for glyph_index, (char, (component_box, part)) in enumerate(zip(expected, parts, strict=True)):
            prefix = 'outlined_' if font == 'outlined' else ''
            filename = f"{prefix}{'slash' if char=='/' else char}_{len(manifest['features']):02d}.png"
            cv2.imwrite(str(destination/filename), part)
            manifest['features'].append({'file':filename, 'glyph':char, 'font':font, 'source':source,
                                         'source_sha256':hashlib.sha256(source_data).hexdigest(),
                                         'source_size': [source_image.shape[1], source_image.shape[0]],
                                         'quantity_xywh':box, 'expected_quantity':''.join(expected),
                                         'source_resize_scale':scale,
                                         'component_coordinate_space':'Source screenshot resampled by source_resize_scale',
                                         'glyph_index':glyph_index, 'component_xywh':component_box,
                                         'processing':('Value < 90 black outline mask, enclosed dark components retained, '
                                                       if font == 'outlined' else 'Brown/red glyph mask, isolated component, ')
                                                      + 'aspect-preserving 40x32 normalization'})
    manifest['trained_glyphs'] = {font: sorted({entry['glyph'] for entry in manifest['features'] if entry['font'] == font})
                                  for font in ('quantity', 'outlined')}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    package = root/'src'/'hayday'/'assets'/'quantities'
    package.mkdir(parents=True, exist_ok=True)
    for filename in [entry['file'] for entry in manifest['features']] + ['manifest.json']:
        shutil.copy2(destination/filename, package/filename)
    print(f'Saved {len(manifest["features"])} original-font quantity references.')


if __name__ == '__main__':
    main()

"""Crop exact, observed dialog text; retain original RGB and binary alpha masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'fuel_tutorial': {
        'source': 'dialog_fuel_tutorial.png',
        'artifact': 'artifacts/reconnected-latest.png',
        'bounds': [728, 137, 1070, 662],
        'target': [848, 350, 826, 175],
        'background': [[770, 250, 80, 50], [1540, 620, 100, 60]],
        'features': {
            'fuel_anchor': [976, 348, 1208, 408],
            'fuel_line_1': [876, 348, 1652, 408],
            'fuel_line_2': [848, 408, 1676, 472],
            'fuel_line_3': [1052, 472, 1476, 528],
        },
    },
    'connection_lost': {
        'source': 'dialog_connection_lost.png',
        'artifact': 'artifacts/live-farming-tests/20260906T171516/001.png',
        'bounds': [173, 2, 1549, 1066],
        'target': [825, 952, 270, 83],
        'background': [[360, 440, 120, 90], [1460, 780, 100, 60]],
        'features': {
            'connection_anchor': [1088, 24, 1280, 144],
            'connection_title': [640, 24, 1280, 144],
            'connection_body_1': [312, 628, 1596, 684],
            'connection_body_2': [536, 680, 1380, 736],
            'connection_retry': [824, 948, 1096, 1040],
        },
    },
    'server_maintenance': {
        'source': 'dialog_server_maintenance.png',
        'artifact': 'artifacts/blue-sweater/selected-actual-pen.png',
        'bounds': [173, 2, 1549, 1066],
        'target': [796, 950, 328, 89],
        'background': [[360, 440, 120, 90], [1460, 780, 100, 60]],
        'features': {
            'maintenance_anchor': [1000, 32, 1252, 144],
            'maintenance_title': [676, 32, 1252, 144],
            'maintenance_body': [304, 624, 1420, 684],
            'maintenance_retry': [796, 948, 1124, 1040],
        },
    },
}


def build():
    captures = ROOT/'images/reference_captures'
    output = ROOT/'images/dialogs'
    package = ROOT/'src/hayday/assets/dialogs'
    for directory in (captures, output, package):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'coordinate_format': 'Feature boxes are LTRB; other boxes are XYWH.',
                'dialogs': {}}
    for kind, specification in SPECS.items():
        source = captures/specification['source']
        if not source.exists():
            source.write_bytes((ROOT/specification['artifact']).read_bytes())
        data = source.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        if image is None or image.shape != (1080, 1920, 3):
            raise ValueError(f'{source.name} requires its unscaled 1920x1080 source capture.')
        record = {key: specification[key] for key in ('source', 'bounds', 'target', 'background')}
        record['background'] = [{'box': box, 'color_bgr': np.median(
            image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], axis=(0, 1)).tolist()}
            for box in specification['background']]
        record.update(source_sha256=hashlib.sha256(data).hexdigest(), source_size=[1920, 1080],
                      original_artifact=specification['artifact'], features=[])
        for name, box in specification['features'].items():
            x0, y0, x1, y1 = box
            original = image[y0:y1, x0:x1]
            cutoff = 95 if name in ('connection_anchor', 'connection_title', 'connection_retry',
                                    'maintenance_anchor', 'maintenance_title', 'maintenance_retry') else 185
            mask = (cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) < cutoff).astype(np.uint8)*255
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            rgba = np.dstack((original, mask))
            encoded = cv2.imencode('.png', rgba)[1].tobytes()
            filename = name+'.png'
            (output/filename).write_bytes(encoded)
            (package/filename).write_bytes(encoded)
            record['features'].append({'file': filename, 'box': box,
                'mask': f'Original BGR grayscale < {cutoff}; dilate binary mask with a 5x5 square; original RGB unchanged.'})
        manifest['dialogs'][kind] = record
    encoded = json.dumps(manifest, indent=2)+'\n'
    (output/'manifest.json').write_text(encoded, encoding='utf-8')
    (package/'manifest.json').write_text(encoded, encoding='utf-8')


if __name__ == '__main__':
    build()

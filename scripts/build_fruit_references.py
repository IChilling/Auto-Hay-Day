"""Preserve original basket/arrow pixels from the observed fruit guide."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'basket': {'box': [700, 526, 844, 704], 'polygon': [
        [711, 608], [715, 580], [725, 551], [742, 535], [763, 529], [784, 536],
        [803, 555], [816, 580], [820, 603], [831, 610], [837, 633], [830, 663],
        [821, 682], [802, 695], [738, 697], [723, 687], [712, 672], [707, 642],
        [707, 624]], 'holes': [[[734, 595], [739, 572], [748, 554], [760, 550],
                              [775, 558], [785, 577], [790, 600]]]},
    'arrow': {'box': [806, 636, 904, 738], 'polygon': [
        [829, 643], [867, 670], [876, 649], [900, 729], [895, 735], [830, 720],
        [841, 701], [807, 696], [815, 681], [823, 664]], 'holes': []},
}


def build():
    source = ROOT/'images/reference_captures/resource_cherry_guided.png'
    original = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(original, np.uint8), 1)
    if image is None or image.shape != (1080, 1920, 3):
        raise ValueError('The original unscaled fruit-guide screenshot is required.')
    verified_source = ROOT/'images/reference_captures/fruit_cherry_before_success.png'
    verified_png = verified_source.read_bytes()
    verified = cv2.imdecode(np.frombuffer(verified_png, np.uint8), 1)
    result_source = ROOT/'images/reference_captures/fruit_cherry_gesture_result.png'
    result_png = result_source.read_bytes()
    if verified is None or verified.shape != image.shape:
        raise ValueError('The verified gesture requires its original unscaled before capture.')
    directory, package = ROOT/'images/fruit', ROOT/'src/hayday/assets/fruit'
    for target in (directory, package):
        target.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'source': source.name,
        'source_sha256': hashlib.sha256(original).hexdigest(), 'source_size': [1920, 1080],
        'coordinate_format': 'Source boxes are LTRB; points/polygons use original screenshot coordinates.',
        'features': {}, 'tool_point': [771, 650],
        'gesture_validation': {'verified': True, 'before': verified_source.name,
            'before_sha256': hashlib.sha256(verified_png).hexdigest(),
            'result': result_source.name, 'result_sha256': hashlib.sha256(result_png).hexdigest(),
            'drag_start': [751, 650], 'drag_end': [930, 650], 'duration_ms': 500,
            'reference_to_verified_translation': [-20, 0],
            'evidence': 'Fresh-revalidated live drag showed cherries +1 and XP +13. The basket guide remained afterward; runtime inventory observations must confirm each attempted harvest.'},
        'highlight_boxes': [[924, 480, 973, 704], [602, 743, 874, 786]],
        'provenance': 'Original RGB remains unchanged. Binary polygon alpha is intersected with pixels stable across two independently captured farm zooms, excluding changing scenery and the basket handle hole. The white wedge validates the tool menu, not a tree. Runtime endpoints are detected ripe cherry clusters at independent world scales; historical gesture coordinates are evidence only.'}
    for name, specification in SPECS.items():
        x0, y0, x1, y1 = specification['box']
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        offset = np.array([x0, y0])
        cv2.fillPoly(mask, [np.array(specification['polygon'])-offset], 255)
        for hole in specification['holes']:
            cv2.fillPoly(mask, [np.array(hole)-offset], 0)
        corresponding = verified[y0:y1, x0-20:x1-20]
        difference = np.abs(image[y0:y1, x0:x1].astype(np.int16)-corresponding.astype(np.int16))
        # The handle encloses farm scenery. Cross-capture agreement removes
        # antialiased scenery and loose polygon margins without recoloring art.
        mask[difference.max(axis=2) > 14] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        rgba = np.dstack((image[y0:y1, x0:x1], mask))
        data = cv2.imencode('.png', rgba)[1].tobytes()
        filename = name+'.png'
        for target in (directory, package):
            (target/filename).write_bytes(data)
        manifest['features'][name] = {'file': filename, **specification,
            'stable_comparison': {'source': verified_source.name,
                'source_sha256': hashlib.sha256(verified_png).hexdigest(),
                'box': [x0-20, y0, x1-20, y1]},
            'mask': 'Fill outer polygon, erase holes, retain pixels with max-channel RGB difference <= 14 versus the aligned independent capture, erode with a 3x3 square.'}
    manifest['canopy'] = {'species': 'cherry', 'references': [],
        'cluster_evidence': 'Two distinct patterns scoring at least 0.86, or one detailed multi-cherry pattern scoring at least 0.93 with surrounding textured green leaves. All candidates require cherry color and the requested item match.',
        'lighting': 'One shared RGB gain in [0.55,1.10] and offset in [-15,120] accounts for neutral cloud brightness. Per-channel hue corrections are never allowed.',
        'search': 'Independent world scales within the detected basket neighborhood. Match ripe clusters outside the tool-menu backdrop and choose the nearest supported cluster. Expose all supported clusters for fresh-frame agreement.',
        'validation': {'before': 'fruit_cherry_small_canopy_before.png',
            'result': 'fruit_cherry_small_canopy_result.png',
            'drag_start': [751, 589], 'drag_end': [975, 540], 'duration_ms': 1000,
            'evidence': 'Manual live gesture to visible ripe canopy showed cherries +1 at small farm zoom. Historical points are never runtime targets.'}}
    specifications = [
        ('cherry_cluster_large', 'fruit_cherry_before_success.png', [1043, 429, 1090, 466], [.30, 1.55]),
        ('cherry_cluster_slanted', 'fruit_cherry_before_success.png', [1110, 450, 1158, 486], [.30, 1.55]),
        ('cherry_cluster_small', 'fruit_cherry_partly_harvested_guided.png', [969, 551, 992, 569], [.65, 3.3]),
        ('cherry_item', 'orders_missing_cherry.png', [1245, 455, 1322, 516], None),
    ]
    for name, capture_name, box, scale_range in specifications:
        png = (ROOT/'images/reference_captures'/capture_name).read_bytes()
        pixels = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
        x0, y0, x1, y1 = box
        crop = pixels[y0:y1, x0:x1]
        blue, green, red = cv2.split(crop.astype(np.int16))
        mask = ((red > 80) & (red > green*1.6) & (blue > green*1.2)
                & (red > blue*1.5)).astype(np.uint8)*255
        # Keep original fruit shading and highlight holes; a one-pixel margin
        # includes the original antialiased silhouette, not new painted pixels.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        data = cv2.imencode('.png', np.dstack((crop, mask)))[1].tobytes()
        for target in (directory, package):
            (target/(name+'.png')).write_bytes(data)
        spec = {'file': name+'.png', 'source': capture_name, 'box': box,
            'source_sha256': hashlib.sha256(png).hexdigest(),
            'mask': 'Original BGR: R>80, R>1.6G, B>1.2G, R>1.5B; binary 3x3 close then one 3x3 dilation. RGB pixels remain unchanged.'}
        if scale_range:
            spec['world_scale_range'] = scale_range
            manifest['canopy']['references'].append(spec)
        else:
            manifest['canopy']['desired_item'] = spec
    for key in ('before', 'result'):
        capture_name = manifest['canopy']['validation'][key]
        png = (ROOT/'images/reference_captures'/capture_name).read_bytes()
        manifest['canopy']['validation'][key+'_sha256'] = hashlib.sha256(png).hexdigest()
    manifest['raspberry'] = {'species': 'raspberry', 'references': [], 'desired_variants': [],
        'evidence': 'Native ripe berry patterns plus surrounding leaf texture, matched beyond the verified basket arrow. Independent world zoom; no saved farm coordinates are used for harvesting.'}
    for name, capture_name, box, scale_range in (
        ('raspberry_cluster_left', 'fruit_raspberry_ripe_guided.png', [1695, 410, 1731, 443], [.65, 3.5]),
        ('raspberry_cluster_right', 'fruit_raspberry_ripe_guided.png', [1732, 397, 1775, 429], [.65, 3.5]),
        ('raspberry_item', 'fruit_raspberry_recipe_before.png', [909, 320, 1003, 421], None),
        ('raspberry_item_board', 'orchard_raspberry_board_order.png', [1245, 455, 1322, 516], None),
    ):
        original = (ROOT/'images/reference_captures'/capture_name).read_bytes()
        pixels = cv2.imdecode(np.frombuffer(original, np.uint8), 1)
        x0, y0, x1, y1 = box
        crop = pixels[y0:y1, x0:x1]
        blue, green, red = cv2.split(crop.astype(np.int16))
        mask = ((red > 80) & (red > green*1.6) & (blue > green*1.2)
                & (red > blue*1.5)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        data = cv2.imencode('.png', np.dstack((crop, mask)))[1].tobytes()
        spec = {'file': name+'.png', 'source': capture_name, 'box': box,
            'source_sha256': hashlib.sha256(original).hexdigest(),
            'mask': 'Original BGR: R>80, R>1.6G, B>1.2G, R>1.5B; binary 3x3 close/dilate. Original RGB unchanged.'}
        for target in (directory, package):
            (target/spec['file']).write_bytes(data)
        if scale_range:
            spec['world_scale_range'] = scale_range
            manifest['raspberry']['references'].append(spec)
        elif name == 'raspberry_item_board':
            manifest['raspberry']['desired_variants'].append(spec)
        else:
            manifest['raspberry']['desired_item'] = spec
    manifest['raspberry']['validation'] = {
        'before': 'fruit_raspberry_before_verified.png',
        'result': 'fruit_raspberry_harvest_result.png',
        'inventory': 'fruit_raspberry_recipe_confirmed.png',
        'evidence': 'Live basket drag collected one Raspberry. Two subsequent Red Berry Cake recipe readings confirmed inventory 0/1 to 1/1; the durable fruit intent became confirmed.'}
    for key in ('before', 'result', 'inventory'):
        name = manifest['raspberry']['validation'][key]
        original = (ROOT/'images/reference_captures'/name).read_bytes()
        manifest['raspberry']['validation'][key+'_sha256'] = hashlib.sha256(original).hexdigest()
    data = json.dumps(manifest, indent=2)+'\n'
    for target in (directory, package):
        (target/'manifest.json').write_text(data, encoding='utf-8')


if __name__ == '__main__':
    build()

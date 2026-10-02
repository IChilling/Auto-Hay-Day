"""Extract cow controls and world features from original MuMu screenshots."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/orders'


def build():
    features = {
        'bucket': ('cow_ready_pen.png', (811,275,954,436),
                   [(14,7),(92,0),(130,18),(142,42),(132,69),(130,120),(112,155),
                    (50,160),(15,147),(8,112),(0,95),(4,63),(6,36)]),
        'feed_inactive': ('cow_ready_pen.png', (1060,224,1163,303), None),
        'feed': ('cow_edge_menu.png', (1536,273,1639,352), None),
        'trough': ('cow_ready_pen.png', (1130,531,1198,583),
                   [(0,30),(14,18),(47,1),(61,0),(67,11),(61,34),(20,51),(6,46)]),
        'trough_wide': ('cow_wide_cloud.png', (1498,582,1558,628),
                   [(0,27),(12,16),(41,1),(54,0),(59,10),(54,30),(18,45),(5,41)]),
        'trough_rim': ('cow_wide_cloud.png', (1498,582,1558,628),
                   [(12,16),(41,1),(54,0),(59,10),(54,20),(48,23),
                    (40,22),(50,13),(52,9),(46,7),(23,20)]),
    }
    manifest = {'version':1, 'reference_height':1080, 'desired_item':'milk_item.png',
                'desired_variants':[{'file':'milk_recipe_item.png',
                    'source':'red_berry_cake_recipe.png', 'box':[905,532,990,626],
                    'source_sha256':hashlib.sha256((FIXTURES/'red_berry_cake_recipe.png').read_bytes()).hexdigest()}],
                'tool_point':[886,352], 'feed_point':[1112,263],
                'feed_count_box':[946,167,137,84], 'features':{}}
    for name,(filename,box,_polygon) in features.items():
        manifest['features'][name] = {'file':name+'.png','source':filename,'box':box,
            'source_sha256':hashlib.sha256((FIXTURES/filename).read_bytes()).hexdigest()}
    for root in (ROOT/'images/animals/cow',ROOT/'src/hayday/assets/animals/cow'):
        root.mkdir(parents=True,exist_ok=True)
        for name,(filename,(x,y,right,bottom),polygon) in features.items():
            crop = cv2.imread(str(FIXTURES/filename))[y:bottom,x:right].copy()
            alpha = np.zeros(crop.shape[:2],np.uint8)
            if polygon:
                cv2.fillPoly(alpha,[np.array(polygon,np.int32)],255)
            else:
                alpha[:] = 255
            cv2.imwrite(str(root/(name+'.png')),np.dstack((crop,alpha)))
        (root/'milk_item.png').write_bytes((ROOT/'images/learned_items/8311465752f9b888ebbe.png').read_bytes())
        # This full can is separated from the quantity; the historical learned
        # crop is retained only for compatibility with saved item identities.
        milk = cv2.imread(str(FIXTURES/'red_berry_cake_recipe.png'))[532:626,905:990]
        cv2.imwrite(str(root/'milk_recipe_item.png'),milk)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    build()

"""Identify finished output beside its machine, independently of queue labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode


@dataclass(frozen=True)
class ReadyProduct:
    name: str
    machine: str
    target: VisualTarget
    output: VisualTarget
    scale: float


class ReadyProductVision:
    def __init__(self, root=None):
        source = Path(__file__).resolve().parents[2]/'images/production_ready'
        self.root = Path(root) if root else (source if source.is_dir() else Path(__file__).parent/'assets/production_ready')
        manifest = json.loads((self.root/'manifest.json').read_text('utf-8'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported finished-production references.')
        self.height = manifest['reference_height']
        self.products = manifest['products']
        self.matcher = ResourceVision()
        self.references = {spec['file']: _decode((self.root/spec['file']).read_bytes(), True)
                           for product in self.products for spec in product['features'].values()}

    def supports(self, icon, title=None, cancel=lambda: False):
        return any(self._identified(p, icon, title, cancel) for p in self.products)

    def _identified(self, product, icon, title, cancel):
        if title is not None:
            return self.matcher.titles_match(title, (self.root/product['title']).read_bytes())
        return bool(self.matcher.find_item(icon, (self.root/product['identifier']).read_bytes(),
            min_scale=.5, max_scale=1.8, threshold=.95, cancel=cancel))

    def find(self, png, icon, title=None, cancel=lambda: False):
        image = _decode(png)
        base = image.shape[0]/self.height
        sizes = np.unique(np.r_[np.geomspace(base*.45, base*2.2, 30), base])
        found = []
        for product in self.products:
            if cancel():
                return ()
            if not self._identified(product, icon, title, cancel):
                continue
            features = product['features']
            spec = features['anchor']
            anchor = self.references[spec['file']]
            for candidate in self.matcher._search(image, anchor, sizes, .93, cancel, max_peaks=4):
                scale = candidate.width/anchor.shape[1]
                hits = []
                for name in ('support', 'output'):
                    other = features[name]
                    x, y, w, h = other['box']
                    px = candidate.x+(x-spec['box'][0])*scale
                    py = candidate.y+(y-spec['box'][1])*scale
                    padding = max(5, round(7*scale))
                    left, top = max(0, round(px)-padding), max(0, round(py)-padding)
                    right, bottom = min(image.shape[1], round(px+w*scale)+padding), min(image.shape[0], round(py+h*scale)+padding)
                    if right-left < w*scale or bottom-top < h*scale:
                        break
                    targets = self.matcher._search(image[top:bottom, left:right], self.references[other['file']],
                        np.array([scale*.96, scale, scale*1.04]), .91, cancel, max_peaks=2)
                    targets = [VisualTarget(t.x+left, t.y+top, t.width, t.height, t.score) for t in targets
                               if abs(t.x+left-px) < padding and abs(t.y+top-py) < padding]
                    if len(targets) != 1:
                        break
                    hits.append(targets[0])
                if len(hits) != 2:
                    continue
                x, y = product['tap']
                point = (round(candidate.x+(x-spec['box'][0])*scale),
                         round(candidate.y+(y-spec['box'][1])*scale))
                found.append(ReadyProduct(product['name'], product['machine'],
                    VisualTarget(point[0]-4, point[1]-4, 8, 8, min(candidate.score, *(t.score for t in hits))),
                    hits[-1], scale))
        return tuple(found)


def collect_ready(worker, frame, icon, title, key):
    """Dismiss obscuring controls, re-find output, then click the machine once."""
    from hayday.barn_storage import StorageReader

    if not hasattr(worker, '_ready_products'):
        worker._ready_products = ReadyProductVision()
    vision = worker._ready_products
    # The scene can display EMPTY while finished output waits on the table.
    # Remove the menu before matching instead of accepting an occluded icon.
    scene = worker._observe(frame)
    if scene.popups or scene.empty_slots:
        frame = worker._dismiss(frame)
        worker._wait(.35)
        frame = worker._capture()
        scene = worker._observe(frame)
    if scene.popups or scene.empty_slots:
        return worker._finish('waiting', 'The production menu still obscures the machine; checking other items.',
                              frame, item=key, defer_item=True, retry_after_seconds=30)
    candidates = vision.find(frame.png, icon, title, worker.cancel_event.is_set)
    if len(candidates) != 1:
        return None
    if not hasattr(worker, '_storage_reader'):
        worker._storage_reader = StorageReader()
    storage = worker._storage_reader.read(frame.png, worker.cancel_event.is_set)
    if storage.full:
        return worker._finish('waiting', 'Barn storage is full; finished-product collection is paused.',
                              frame, item=key, storage_blocked=True, defer_session=True)
    fresh = worker._capture()
    checked = vision.find(fresh.png, icon, title, worker.cancel_event.is_set)
    if (len(checked) != 1 or checked[0].name != candidates[0].name
            or not worker._near(checked[0].target.center, candidates[0].target.center, 8)):
        return worker._finish('waiting', 'Finished output moved before collection; checking again later.',
                              fresh, item=key, defer_item=True, retry_after_seconds=30)
    target = checked[0]
    worker._record(key, 'machine_collection_attempted', collection_probed=True,
                   pending_stage='awaiting_collection', collection_machine=target.machine)
    worker.progress(f'Collecting finished {target.name} by clicking its {target.machine}.')
    after = worker._tap(target.target.center, fresh)
    worker._wait(.5)
    after = worker._capture()
    storage = worker._storage_reader.read(after.png, worker.cancel_event.is_set)
    if storage.full:
        return worker._finish('waiting', 'Barn storage is full; collection awaits an inventory check.',
                              after, item=key, storage_blocked=True, defer_session=True)
    return worker._finish('waiting', 'Clicked the machine to collect finished output; verifying inventory before further production.',
                          after, item=key, wait_seconds=1)

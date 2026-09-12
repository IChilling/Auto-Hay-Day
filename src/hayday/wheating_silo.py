"""Free wheat storage after an explicit Silo Full interruption."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hayday.farming import FarmingWorker
from hayday.resource_vision import ResourceVision, _decode
from hayday.wheating_vision import WheatingVision


class SiloFullDetected(Exception):
    def __init__(self, frame, close):
        super().__init__('The silo is full.')
        self.frame, self.close = frame, close


class SiloFullVision:
    def __init__(self):
        root = Path(__file__).parent/'assets/wheating/silo_full'
        self.spec = json.loads((root/'manifest.json').read_text('utf-8'))
        self.refs = {name: _decode((root/(name+'.png')).read_bytes(), True) for name in self.spec['boxes']}
        self.matcher = ResourceVision()
        self._png = self._result = None

    @staticmethod
    def possible(image):
        small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        return (cv2.inRange(hsv, (15, 0, 185), (45, 110, 255)) > 0).mean() > .45

    def observe(self, frame, *, image=None, cancel=lambda: False):
        if frame.png == self._png:
            return self._result
        image = _decode(frame.png) if image is None else image
        result = None
        if self.possible(image):
            scale = frame.height/1080
            left, top = round(frame.width*.25), round(frame.height*.03)
            crop = image[top:round(frame.height*.25), left:round(frame.width*.82)]
            anchors = self.matcher._search(crop, self.refs['title'], np.array([scale]), .93, cancel, 2)
            if not anchors:
                anchors = self.matcher._search(crop, self.refs['title'], np.array([.85, .95, 1.05, 1.15])*scale, .93, cancel, 2)
            candidates = []
            for anchor in anchors:
                anchor = replace(anchor, x=anchor.x+left, y=anchor.y+top)
                found = {}
                for name in ('message', 'close'):
                    x1, y1, x2, y2 = self.spec['boxes'][name]
                    target = WheatingVision.relative(anchor, (x1, y1, x2-x1, y2-y1), source=self.spec['boxes']['title'])
                    x, y = max(0, target.x-8), max(0, target.y-8)
                    right, bottom = min(frame.width, target.x+target.width+8), min(frame.height, target.y+target.height+8)
                    hits = self.matcher._search(image[y:bottom, x:right], self.refs[name],
                        np.array([anchor.width/self.refs['title'].shape[1]]), .92, cancel, 1)
                    if hits:
                        found[name] = replace(hits[0], x=hits[0].x+x, y=hits[0].y+y)
                if len(found) == 2:
                    candidates.append(found['close'])
            if len(candidates) == 1:
                result = candidates[0]
        self._png, self._result = frame.png, result
        return result


class WheatSiloFull:
    def __init__(self, runner):
        self.run = runner
        self.vision = SiloFullVision()
        self.recovering = False

    def observe(self, frame):
        return self.vision.observe(frame, image=self.run.vision._image_for(frame.png), cancel=self.run.cancel_event.is_set)

    def process(self, frame):
        if not self.recovering:
            close = self.observe(frame)
            if close is not None:
                raise SiloFullDetected(frame, close)
        return frame

    def _dismiss(self, detected):
        close = detected.close
        for _ in range(4):
            fresh = self.run._capture_raw()
            checked = self.observe(fresh)
            if checked is None:
                return fresh
            if FarmingWorker._same_target(close, checked, fresh):
                self.run.tap(checked.center, fresh)
                break
            close = checked
            self.run.wait(.1)
        else:
            self.run.block('The Silo Full close control did not settle.')
        for _ in range(6):
            fresh = self.run._capture_raw()
            if self.observe(fresh) is None:
                return fresh
            self.run.wait(.1)
        self.run.block('Silo Full did not close; no repeated close tap was sent.')

    def recover(self, fields, shop, detected=None):
        """Resume after listing all wheat above the protected seed reserve.

        A recovery call is deliberately bounded.  If the shop made no progress
        during this slice, the durable counter is left intact and ``False`` is
        returned so the runner can leave and retry later.  This keeps a slow or
        temporarily full shop from becoming a permanent Wheating stop.
        """
        self.run.check()
        if self.run.state.get('pending'):
            self.run.block('A shop transaction is unconfirmed; Silo Full recovery cannot replay it.')
        recovery = self.run.state.get('silo_recovery')
        if recovery is None:
            if detected is None:
                return
            pending = {key: entry for key, entry in fields.worker.state['items'].items()
                       if entry.get('stage') in fields.worker._PENDING}
            if any(entry.get('stage') != 'harvest_attempted' for entry in pending.values()):
                self.run.block('Silo Full interrupted an unresolved planting; its evidence was retained.')
            # Archive the interrupted gesture rather than pretending it finished.
            self.run.save_json(self.run.device_root/'silo_interrupted_harvest.json', pending)
            (self.run.device_root/'silo_full.png').write_bytes(detected.frame.png)
            recovery = dict(version=1, released=0, reserve=self.run.state['seed_reserve'],
                            harvests={key: entry.get('operation') for key, entry in pending.items()})
            self.run.state['silo_recovery'] = recovery
            self.run.persist()
        if (not isinstance(recovery, dict) or recovery.get('version') != 1
                or type(recovery.get('released')) is not int or not 0 <= recovery['released'] <= 10000
                or type(recovery.get('reserve')) is not int or not 0 <= recovery['reserve'] <= 9999
                or type(recovery.get('surplus_empty', False)) is not bool
                or not isinstance(recovery.get('harvests'), dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in recovery['harvests'].items())):
            self.run.block('The saved Silo Full recovery is invalid.')
        self.recovering = True
        try:
            if detected:
                self._dismiss(detected)
            else:
                frame = self.run._capture_raw()
                close = self.observe(frame)
                if close:
                    self._dismiss(SiloFullDetected(frame, close))
            group = getattr(fields, '_group', None)
            grouped = max(len(group.points), len(group.view.cells) if group.view else 0) if group else 0
            self.run.state['seed_reserve'] = max(recovery['reserve'], len(getattr(fields, '_known_points', ())), grouped, 1)
            self.run.state['wheat_empty'] = recovery.get('surplus_empty', False)
            shop._stock_empty_until = 0.
            shop._ready_view = None
            self.run.persist()
            self.run.publish('Wheating: silo full. Listing all surplus wheat while keeping the seed reserve.')
            frame, view = shop.observe()
            if view.kind in {'composer', 'edit'}:
                shop.close(frame, view)
            self.run.open_shop()
            deadline = time.monotonic()+180
            next_notice = 0.
            while not recovery.get('surplus_empty', False):
                self.run.check()
                if time.monotonic() >= deadline:
                    try:
                        shop.close_to_farm()
                    except Exception as exc:
                        from hayday.wheating import WheatingCancelled
                        if isinstance(exc, WheatingCancelled):
                            raise
                        # Returning to the farm is itself recoverable.  Keep
                        # the durable release counter and let the runner's
                        # normal restart supervisor restore the workspace.
                        self.run.publish(f'Wheating: silo recovery could not close the shop yet ({exc}); retrying.')
                    self.run.publish(
                        f'Wheating: shop recovery paused after listing {recovery["released"]} wheat; '
                        'keeping the checkpoint and retrying the remaining surplus.'
                    )
                    return False
                before = recovery['released']
                idle = shop.service(deadline=deadline)
                if recovery['released'] == before:
                    if idle and time.monotonic() >= next_notice:
                        self.run.publish(f'Wheating: waiting for shop space; '
                                         f'{before} wheat listed for silo recovery. '
                                         'The interrupted field is preserved.')
                        next_notice = time.monotonic()+15
                    self.run.wait(.5)
            shop.close_to_farm()
            for key, operation in recovery['harvests'].items():
                entry = fields.worker.state['items'].get(key)
                if entry is not None:
                    if entry.get('stage') != 'harvest_attempted' or entry.get('operation') != operation:
                        self.run.block('The interrupted harvest record changed during Silo Full recovery.')
                    fields.worker.state['items'].pop(key)
            self.run.save_json(fields.worker.state_path, fields.worker.state)
            self.run.state['seed_reserve'] = recovery['reserve']
            self.run.state.pop('silo_recovery')
            self.run.persist()
            fields.next_harvest = 0.
            self.run.publish(f'Wheating: freed space for {recovery["released"]} wheat. Resuming the remaining harvest and planting.')
            return True
        finally:
            self.recovering = False

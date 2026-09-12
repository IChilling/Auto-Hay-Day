"""Choose another observed chicken pen without reusing stale farm coordinates."""
import hashlib
import time
from datetime import UTC, datetime

import numpy as np

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker


class ChickenCare:
    def __init__(self, owner):
        self.owner = owner

    @staticmethod
    def _center(polygon):
        return tuple(map(int, np.mean(polygon, axis=0)))

    def recently_collected(self, frame, key, observed):
        owner = self.owner
        entry = owner.state['items'].get(key, {})
        name = entry.get('before_capture')
        if entry.get('stage') != 'confirmed' or not isinstance(name, str):
            return False
        observed_at = entry.get('harvest_observed_at', entry.get('inventory_before', {}).get('captured_at'))
        if isinstance(observed_at, str):
            try:
                # This is a revisit cooldown, not a claim that eggs are ready.
                if (datetime.now(UTC)-datetime.fromisoformat(observed_at)).total_seconds() > 600:
                    return False
            except (ValueError, TypeError):
                pass
        from pathlib import Path
        if Path(name).name != name:
            return False
        path = owner.state_path.parent/'fruit_evidence'/name
        if not path.is_file():
            return False
        png = path.read_bytes()
        if hashlib.sha256(png).hexdigest() != entry.get('before_sha256'):
            return False
        polygon = entry.get('pen_polygon')
        if not polygon:
            old = owner.vision.basket(png, desired_icon=owner.vision._chicken.item)
            polygon = old.pen_polygon if old and old.species == 'egg' else ()
        if not polygon or not observed.pen_polygon:
            return False
        previous = Screenshot(png, frame.width, frame.height, 'saved-pen')
        projected = FarmingWorker._translated_plot(previous, frame, self._center(polygon), require_visible=False)
        return projected is not None and np.linalg.norm(np.subtract(
            projected, self._center(observed.pen_polygon))) < max(30, observed.scale*50)

    def next_pen(self, frame, icon, excluded):
        """Search fresh nearby views and open one different, fully observed pen."""
        owner = self.owner
        vision = owner.vision._chicken
        ground = CameraNavigator._grass_start(frame, 0, 0)
        if ground is None:
            return None
        owner.client.tap(*ground[:2], width=frame.width, height=frame.height)
        owner.cancel_event.wait(.35)
        clear = owner._frame()
        # Overlapping neighboring views; no zoom or persistent world position.
        pans = ((-.28, 0), (0, -.22), (.28, 0), (.28, 0),
                (0, .22), (0, .22), (-.28, 0), (-.28, 0))
        deadline = time.monotonic()+45
        for index in range(len(pans)+1):
            owner._check()
            if time.monotonic() >= deadline:
                break
            candidates = vision.enclosures(clear.png, owner.cancel_event.is_set)
            for coop, target, polygon in candidates:
                center = self._center(polygon)
                prior = [FarmingWorker._translated_plot(old, clear, point, require_visible=False) for old, point in excluded]
                if any(point is None for point in prior):
                    # Without a camera transform the same-looking coops cannot
                    # be claimed as new. A later resource visit can try again.
                    continue
                if any(np.linalg.norm(np.subtract(center, point)) < max(35, target.width*1.5) for point in prior):
                    continue
                fresh = owner._frame()
                observed_at = time.monotonic()
                matches = [(c, t, p) for c, t, p in vision.enclosures(fresh.png, owner.cancel_event.is_set)
                           if np.linalg.norm(np.subtract(c.center, coop.center)) < 6
                           and abs(c.width-coop.width) < 4]
                if len(matches) != 1 or time.monotonic()-observed_at > 3:
                    continue
                _, target, polygon = matches[0]
                owner._check()
                owner.progress('Opening another freshly detected chicken pen.')
                owner.client.tap(*target.center, width=fresh.width, height=fresh.height)
                owner.cancel_event.wait(.4)
                opened = owner._frame()
                guide = owner._read(opened, icon)
                if guide is None:
                    guide = owner.vision.chicken_feed(opened.png, icon, owner.cancel_event.is_set)
                projected = FarmingWorker._translated_plot(fresh, opened, self._center(polygon), require_visible=False)
                if (guide is not None and guide.species == 'egg' and guide.pen_polygon
                        and projected is not None and np.linalg.norm(np.subtract(
                            projected, self._center(guide.pen_polygon))) < target.width*2):
                    return opened, guide
                return None
            if index == len(pans):
                break
            drag = CameraNavigator._grass_start(clear, *pans[index])
            if drag is None or CameraNavigator._modal_visible(clear):
                break
            owner._check()
            owner.progress('Searching an adjacent farm view for another chicken pen.')
            owner.client.swipe(*drag, width=clear.width, height=clear.height, duration_ms=650)
            owner.cancel_event.wait(.45)
            clear = owner._frame()
        return None

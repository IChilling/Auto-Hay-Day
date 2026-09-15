"""Group fresh, adjacent plot evidence and synchronize interrupted crop cycles.

This geometry schedules work only. Crop controls and the existing harvest/soil
detectors still provide every input route; historical coordinates are not joined.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, _decode


@dataclass(frozen=True)
class FieldGroupView:
    cells: tuple  # (x, y, 'ripe' | 'growing' | 'bare') in this frame only
    pitch: tuple

    def count(self, kind):
        return sum(cell[2] == kind for cell in self.cells)


def lattice_pitch(points):
    """Estimate both row axes from actual verified tile neighbors."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None
    delta = points[:, None, :]-points[None, :, :]
    dx, dy = abs(delta[:, :, 0]), abs(delta[:, :, 1])
    valid = (dx > 10) & (dy > 5) & (dx/dy.clip(1) > 1.7) & (dx/dy.clip(1) < 2.3)
    if valid.sum() < 4:
        return None
    shortest = np.min(dx[valid])
    adjacent = valid & (dx < shortest*1.15)
    pitch = np.median(dx[adjacent]), np.median(dy[adjacent])
    # Off-grid points cannot seed another lattice or inflate its spacing.
    origin = points[0]
    delta = points-origin
    ij = np.c_[delta[:, 0]/pitch[0]+delta[:, 1]/pitch[1],
               delta[:, 1]/pitch[1]-delta[:, 0]/pitch[0]]/2
    if np.max(abs(ij-np.rint(ij))) > .15:
        return None
    return tuple(map(float, pitch))


def observe_group(frame, seeds, pitch, soil_texture, *, bounded=False):
    """Follow one observed lattice, stopping at grass, roads and other objects."""
    from hayday.wheating_crop import WheatFarmingVision
    image = _decode(frame.png)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dx, dy = pitch
    origin = np.asarray(seeds[0], dtype=float)
    indices = []
    for point in seeds:
        x, y = np.subtract(point, origin)/(dx, dy)
        indices.append((round((x+y)/2), round((y-x)/2)))
    known = set(indices)
    seen, found, queue = set(), {}, deque(indices)
    textures = None

    def furrows(x, y):
        nonlocal textures
        if textures is None:
            references = soil_texture if isinstance(soil_texture, (tuple, list)) else (soil_texture,)
            textures = []
            for reference in references:
                color, mask = ResourceVision._scaled(reference, dx/53.5)
                textures.append((cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), mask))
        for ref, mask in textures:
            rh, rw = ref.shape
            crop = image[max(0, y-rh-8):min(frame.height, y+rh+8),
                         max(0, x-rw-8):min(frame.width, x+rw+8)]
            if crop.shape[0] < rh or crop.shape[1] < rw:
                continue
            scores = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), ref,
                                       cv2.TM_CCOEFF_NORMED, mask=mask)
            if float(np.nan_to_num(scores, nan=0., posinf=0., neginf=0.).max()) >= .91:
                return True
        return False

    while queue and len(seen) < 512*5:
        ij = queue.popleft()
        if ij in seen:
            continue
        seen.add(ij)
        if bounded and ij not in known:
            # The complete bare-soil footprint was recorded before planting.
            # Yellow overhangs and grass beside buildings are not extra plots.
            # New/disconnected soil is still discovered by the field worker.
            continue
        i, j = ij
        x, y = np.rint(origin+((i-j)*dx, (i+j)*dy)).astype(int)
        if not (frame.width*.12 < x < frame.width*.88 and frame.height*.20 < y < frame.height*.84):
            continue
        pixels = WheatFarmingVision.tile_pixels(hsv, (x, y), dx, dy)
        if pixels is None:
            continue
        hue, saturation, value = pixels.T
        soil = ((hue >= 8) & (hue <= 18) & (saturation >= 65) & (value < 245)).mean()
        green = ((hue >= 28) & (hue <= 90) & (saturation > 70) & (value > 70)).mean()
        yellow = ((hue >= 18) & (hue <= 31) & (saturation >= 140) & (value >= 160)).mean()
        # Color plus aligned furrow/plant texture supports scheduling. It never
        # authorizes a sickle/seed drag, nor identifies unrelated green grass.
        if yellow >= .55:
            kind = 'ripe'
        elif soil >= .90 and (ij in known or furrows(x, y)):
            kind = 'bare'
        elif ((green >= .40 and soil >= .055 and green+soil >= .78)
              or (ij in known and green >= .035 and soil >= .45 and yellow < .15)):
            kind = 'growing'
        else:
            continue
        found[ij] = (int(x), int(y), kind)
        queue.extend(((i+1, j), (i-1, j), (i, j+1), (i, j-1)))
        if len(found) > 512:
            return None
    if len(found) < 3 or sum(ij in found for ij in known) < max(2, len(known)*.65):
        return None
    # Starting from several seed cells must not merge disconnected islands.
    # Keep the connected component containing most freshly supported seeds.
    components, remaining = [], set(found)
    while remaining:
        component, pending = set(), [next(iter(remaining))]
        while pending:
            ij = pending.pop()
            if ij not in remaining:
                continue
            remaining.remove(ij)
            component.add(ij)
            i, j = ij
            pending.extend(((i+1,j), (i-1,j), (i,j+1), (i,j-1)))
        components.append(component)
    selected = max(components, key=lambda c: (len(c & known), len(c)))
    return FieldGroupView(tuple(found[ij] for ij in sorted(selected)), pitch)


class WheatFieldGroup:
    def __init__(self, fields):
        self.fields = fields
        self.until = 0.
        self.expires = 0.
        self.waiting = False
        self.suspended_until = 0.
        self.view = None
        self._frame = None
        self._notice = None
        self._coverage = 0
        self.full_route = False
        # Keep one actual verified planting as the geometric reference. A later
        # one-tile repair must not erase its neighbors. No coordinate union is
        # performed: every replacement comes from one completed planting.
        self.reference = getattr(fields, '_known_before', None)
        self.points = list(getattr(fields, '_known_points', []))
        layout = getattr(fields.run, 'state', {}).get('field_layout')
        self.proof = layout.get('before') if isinstance(layout,dict) else None
        observed = layout.get('observed_points') if isinstance(layout, dict) else None
        if (self.reference is not None and isinstance(observed, list) and 3 <= len(observed) <= 512
                and all(isinstance(p, list) and len(p) == 2 and all(type(n) is int for n in p)
                        and 0 <= p[0] < self.reference.width and 0 <= p[1] < self.reference.height
                        for p in observed)
                and len({tuple(p) for p in observed}) == len(observed)
                and lattice_pitch(observed) is not None):
            self.points = [tuple(p) for p in observed]
        self._load_reference()

    def _load_reference(self):
        saved = getattr(self.fields.run, 'state', {}).get('field_group', {})
        if not isinstance(saved,dict) or saved.get('version') != 1:
            return
        points, proof = saved.get('points'), saved.get('reference')
        if (not isinstance(points, list) or not len(self.points) < len(points) <= 512
                or not isinstance(proof, dict)):
            return
        width, height = proof.get('width'), proof.get('height')
        if (any(type(n) is not int or not 2 <= n <= 16384 for n in (width,height))
                or any(not isinstance(p,list) or len(p)!=2 or any(type(n) is not int for n in p)
                       or not 0<=p[0]<width or not 0<=p[1]<height for p in points)
                or len({tuple(p) for p in points}) != len(points) or lattice_pitch(points) is None):
            return
        worker = self.fields.worker
        size = worker._size
        worker._size = width,height
        try:
            reference = worker._saved_frame(proof)
        finally:
            worker._size = size
        if reference is not None:
            self.reference,self.points,self.proof = reference,points,proof

    @property
    def active(self):
        return self.until > 0.

    @property
    def seed_reserve(self):
        return self.view.count('bare') if self.view and self.active else 0

    def observe(self, frame):
        if frame is self._frame:
            return self.view
        fields = self.fields
        self._frame, self.view = frame, None
        sources = [(self.reference,self.points), (fields._known_before,fields._known_points)]
        seen = set()
        for before,points in sources:
            if before is None or id(before) in seen:
                continue
            seen.add(id(before))
            pitch = lattice_pitch(points)
            if pitch is None:
                continue
            moved = fields.worker._translated_plot(before, frame, points[0], require_visible=False)
            if moved is None:
                continue
            seeds = np.asarray(points)+np.subtract(moved, points[0])
            self.view = observe_group(frame, seeds, pitch, fields.worker.vision._soil_textures, bounded=True)
            if self.view is not None:
                if before is fields._known_before and before is not self.reference:
                    # The larger historical reference failed live validation.
                    # Adopt this run's newly verified geometry, including a
                    # smaller/rearranged farm, rather than retaining stale size.
                    self.reference,self.points = before,list(points)
                    self.proof = fields.run.state.get('field_layout', {}).get('before')
                break
        return self.view

    def plan(self, frame):
        """Return wait/repair only for a freshly observed split growth cycle."""
        fields, now = self.fields, time.monotonic()
        self.waiting = False
        if now < self.suspended_until:
            return None
        view = self.observe(frame)
        if view is None:
            if self.active and now <= self.expires:
                self.until = max(self.until, now+5)
                fields.next_harvest = self.until
                self.waiting = True
                return 'wait'
            if self.active:
                self.until = self.expires = 0.
                self.suspended_until = now+fields.worker.growth_duration
            return None
        ripe, growing, bare = (view.count(k) for k in ('ripe', 'growing', 'bare'))
        if self.active:
            self._coverage = max(self._coverage, len(view.cells))
        coverage_ready = not self.active or len(view.cells) >= self._coverage*.90
        if not growing and coverage_ready:
            if ripe and (self.active or ripe > len(getattr(fields, '_known_points', []))):
                self.full_route = True
            if self.active:
                self._record(frame, 'synchronized')
            self.until = self.expires = 0.
            return None
        if not self.active and not (ripe or bare):
            return None
        starting = not self.active
        if starting:
            self.until = now+fields.worker.growth_duration+2
            self.expires = self.until+30
            self._coverage = len(view.cells)
        if now > self.expires:
            # An uncertain/changed crop cannot impose an indefinite wait.
            self._record(frame, 'observation_expired')
            self.until = self.expires = 0.
            self.suspended_until = now+fields.worker.growth_duration
            return None
        self.until = max(self.until, now+5)
        fields.next_harvest = self.until
        fields.has_fields = True
        fields.run.state['field_ready_at'] = time.time()+self.until-now
        fields.run.state['seed_reserve'] = max(fields.run.state.get('seed_reserve', 0), bare)
        if starting:
            self._record(frame, 'synchronizing')
        # If all known surplus was sold, wait for the combined harvest to supply
        # seeds. Otherwise existing stock checks can plant the bare sections.
        if bare and not fields.run.state.get('wheat_empty', False):
            return 'repair'
        self.waiting = True
        return 'wait'

    def planted(self):
        self._frame = None
        planted = (getattr(self.fields.worker, 'field_points', None)
                   or getattr(self.fields.worker, 'planted_points', []))
        actual = self.fields.run.state.get('field_layout', {})
        if planted and len(planted) >= len(self.points) and lattice_pitch(planted) is not None:
            self.reference = self.fields.worker.plant_before
            self.points = list(planted)
            self.proof = actual.get('before')
            self.full_route = False
        if self.proof and self.points:
            saved = self.fields.run.state.get('field_group')
            if not isinstance(saved,dict):
                saved = self.fields.run.state['field_group'] = {}
            saved.update(version=1,reference=self.proof,points=[list(p) for p in self.points])
        self.view = None
        if self.active:
            ready = self.fields.worker.growth_ready_at
            if ready:
                self.until = max(self.until, ready+2)
                self.expires = max(self.expires, self.until+30)
                self.fields.next_harvest = self.until
                self.fields.run.state['field_ready_at'] = time.time()+max(0., self.until-time.monotonic())

    def _record(self, frame, stage):
        run, view = self.fields.run, self.view
        if stage == self._notice:
            return
        self._notice = stage
        previous = run.state.get('field_group')
        report = {**(previous if isinstance(previous,dict) else {}), 'version':1,
                  'reference':self.proof, 'points':[list(p) for p in self.points],
                  'stage': stage, 'at': time.time(), 'pitch': list(view.pitch),
                  'cells': [list(c) for c in view.cells], 'captured_at': frame.captured_at}
        run.state['field_group'] = report
        run.persist()
        if stage == 'synchronizing':
            run.publish(f'Wheating: nearby plots are out of sync ({view.count("ripe")} ripe, '
                        f'{view.count("growing")} growing, {view.count("bare")} bare). '
                        'Combining their next harvest while checking wheat sales.')
        elif stage == 'synchronized':
            run.publish('Wheating: neighboring plots are ready together. Resuming one harvest and planting cycle.')
        # One fixed anomaly snapshot per device; no accumulating image archive.
        try:
            (run.device_root/'field_group.png').write_bytes(frame.png)
        except OSError:
            pass

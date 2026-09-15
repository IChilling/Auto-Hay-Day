"""Finish freshly observed ripe tiles before opening the planting picker."""
from __future__ import annotations

import time
import uuid

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import _decode
from hayday.resources import ResourceChanged
from hayday.wheating_selection import same_harvest


def ripe_tiles(worker, frame, grid):
    """Return indices and current foliage centers on a verified soil grid."""
    if CameraNavigator._modal_visible(frame):
        return None
    before, points, _ = grid
    moved = worker._translated_plot(before, frame, points[0], require_visible=False)
    if moved is None:
        return None
    shift = np.subtract(moved, points[0])
    hsv = cv2.cvtColor(_decode(frame.png), cv2.COLOR_BGR2HSV)
    base = frame.height/1080
    rx, ry = max(3, round(12*base)), max(3, round(8*base))
    found = {}
    for index, point in enumerate(points):
        x, y = map(int, np.add(point, shift)-(0, round(16*base)))
        if not (frame.width*.12 < x < frame.width*.88 and frame.height*.15 < y < frame.height*.86):
            return None
        patch = hsv[y-ry:y+ry+1, x-rx:x+rx+1]
        wheat = cv2.inRange(patch, (18, 140, 160), (31, 255, 255))
        if float((wheat > 0).mean()) >= .55:
            found[index] = (x, y)
    return found


def repair_remaining(worker, frame, key, grid):
    """Bounded repair; every drag needs new crop/tool evidence, even on resume.

    Flying harvest rewards are also yellow. They must clear and the same grid
    cells must remain ripe in two separate captures before any tile is selected.
    The durable repair record precedes touch, so interruption never implies success.
    """
    remaining = ripe_tiles(worker, frame, grid)
    if not remaining:
        return frame
    for _ in range(3):
        worker._pause(2.5)
        clear = worker._quick_frame()
        first = ripe_tiles(worker, clear, grid)
        worker._pause(.25)
        fresh = worker._quick_frame()
        second = ripe_tiles(worker, fresh, grid)
        if first is None or second is None or not fresh.captured_at or fresh.captured_at == clear.captured_at:
            raise ResourceChanged('The remaining wheat needs fresh aligned observations before repair.')
        indices = sorted(first.keys() & second.keys())
        if not indices:
            return fresh
        repairs = worker.state['items'][key].get('harvest_repairs', [])
        if len(repairs) >= 3:
            worker.progress('Wheating: remaining ripe plots will be revisited after planting the cleared soil.')
            return fresh
        origin = second[indices[0]]
        worker.progress(f'Wheating: finishing {len(indices)} freshly verified ripe plots before replanting.')
        worker._check()
        worker.client.tap(*origin, width=fresh.width, height=fresh.height)
        worker._pause(.25)
        previous = previous_frame = None
        for _ in range(6):
            selected = worker._quick_frame()
            started = time.monotonic()
            harvest = worker.vision.harvest(selected.png)
            moved = worker._translated_plot(fresh, selected, origin)
            highlight = harvest.highlight if harvest else None
            associated = (moved is not None and highlight is not None
                          and np.linalg.norm(np.subtract(moved, highlight.center))
                          < max(80, highlight.width, highlight.height))
            if (associated and previous_frame is not None and selected.captured_at
                    and selected.captured_at != previous_frame.captured_at
                    and same_harvest(previous, harvest, selected)):
                break
            previous = harvest if associated else None
            previous_frame = selected
            worker._pause(.1)
        else:
            raise ResourceChanged('The remaining wheat sickle did not settle; no repair drag was sent.')
        # The sickle fan can cover several just-verified crops. Translate the
        # two clear observations through the verified camera movement instead
        # of requiring covered foliage to remain visible behind the controls.
        shift = np.subtract(moved, origin)
        projected = {i: tuple(map(int, np.add(second[i], shift))) for i in indices}
        if any(not (selected.width*.1 < x < selected.width*.9 and selected.height*.15 < y < selected.height*.89)
               for x, y in projected.values()):
            raise ResourceChanged('The remaining wheat needs camera positioning before its repair gesture.')
        from hayday.wheating_routes import diagonal_order, diagonal_trace
        path = diagonal_trace(diagonal_order([harvest.target.center,
            *[projected[i] for i in indices]]), bounds=(selected.width*.1,
                selected.height*.15, selected.width*.9, selected.height*.89))
        operation = worker.state['items'][key]['operation']+'_repair_'+uuid.uuid4().hex
        worker._fresh(started)
        proof = worker._evidence(selected, operation, 'before')
        if worker.state['items'][key].get('replant_grid') is None:
            before, points, pitch = grid
            grid_proof = {**worker._evidence(before, operation, 'grid'),
                          'points': [list(p) for p in points], 'pitch': list(pitch)}
            worker._record(key, 'harvested_needs_replant', replant_grid=grid_proof)
        worker._record(key, 'harvested_needs_replant', harvest_repairs=[*repairs,
            {'before': proof, 'indices': indices, 'points': [list(p) for p in path]}])
        worker._fresh(started)
        worker.client.drag_path([harvest.tool.center, *path], width=selected.width,
            height=selected.height, duration_ms=worker._sweep_duration(path),
            min_waypoint_ms=120, max_step_px=max(1, round(24*selected.height/1080)),
            cancel_event=worker.cancel_event)
        worker._pause(.3)
        frame = worker._quick_frame()
        remaining = ripe_tiles(worker, frame, grid)
        if remaining == {}:
            return frame
    worker.progress('Wheating: harvest repair paused; continuing with cleared soil before revisiting ripe plots.')
    return frame

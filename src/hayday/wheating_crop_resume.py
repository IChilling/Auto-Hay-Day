"""Reopen an interrupted picker only after relocating and verifying bare soil."""
import uuid
from dataclasses import replace

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.game_scene import farm_scene_vision
from hayday.resource_vision import _decode
from hayday.resources import ResourceResult
from hayday.wheating_restart_camera import farm_transform


def bare_grid(worker, before, frame, points, plot):
    """Return current centers only when every saved tile is visibly bare."""
    if not farm_scene_vision().ready(frame.png) or CameraNavigator._modal_visible(frame):
        return None
    matrix = farm_transform(before, frame)
    if matrix is None:
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    centers = np.c_[np.asarray(points), np.ones(len(points))] @ matrix.T
    hsv = cv2.cvtColor(_decode(frame.png), cv2.COLOR_BGR2HSV)
    for x, y in centers:
        if not (frame.width*.13 < x < frame.width*.87 and frame.height*.20 < y < frame.height*.82):
            return None
        pixels = worker.vision.tile_pixels(hsv, (x, y), (plot.width-10)*scale/2, (plot.height-10)*scale/2)
        if pixels is None:
            return None
        soil = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 18) & (pixels[:, 1] >= 65) & (pixels[:, 2] < 245)
        if soil.mean() < .93:
            return None
    return [tuple(map(int, np.rint(p))) for p in centers]


def saved_soil_grid(worker, entry):
    from hayday.wheating_transition import saved_grid

    picker = worker._saved_frame(entry.get('replant_evidence'))
    soil = worker.soil_frame
    if picker is None or soil is None:
        return None
    plot = worker.vision.empty_plot(picker.png)
    if plot is None or not worker.vision.full_outline:
        return None
    grid = saved_grid(worker, entry)
    if grid:
        return grid[0], grid[1], plot
    if entry.get('replant_grid') is not None:
        return None
    origin = worker._translated_plot(picker, soil, plot.center, require_visible=False)
    if origin is None:
        return None
    soil_plot = replace(plot, x=origin[0]-plot.width//2, y=origin[1]-plot.height//2)
    points = worker.vision.empty_tiles(soil, soil_plot, 512)
    return (soil, points, plot) if points else None


def resume_bare_field(worker, frame, icon, key, entry):
    # A planting gesture may have partially succeeded. Never replay that stage.
    if entry.get('stage') != 'harvested_needs_replant':
        return None
    grid = saved_soil_grid(worker, entry)
    if grid is None:
        return rediscover_bare_field(worker, frame, icon, key, entry)
    soil, points, plot = grid
    previous = None
    for _ in range(6):
        current = bare_grid(worker, soil, frame, points, plot)
        if (current and previous and frame.captured_at != previous[0]
                and np.max(np.linalg.norm(np.subtract(current, previous[1]), axis=1)) <= 3):
            break
        previous = (frame.captured_at, current) if current else None
        worker._pause(.25)
        frame = worker._frame()
    else:
        return rediscover_bare_field(worker, frame, icon, key, entry)
    worker.progress(f'Wheating: verified {len(points)} bare plots. Resuming the interrupted seed picker.')
    worker.soil_frame = frame
    worker._picker_before = frame
    opened = worker._tap(current[0], frame)
    opened, selected = worker._await_seed_picker(opened, current[0])
    if selected is None:
        return ResourceResult('changed', 'The recovered soil picker did not settle; no planting gesture was sent.')
    evidence_id = entry['operation']+'_resume_'+uuid.uuid4().hex
    soil_proof = worker._evidence(frame, evidence_id, 'harvest_clear')
    proof = worker._evidence(opened, evidence_id, 'replant_picker')
    # Replace the matched pair together. A failed picker must not combine an
    # old zoom's picker with a new zoom's clear soil on the next retry.
    worker._record(key, 'harvested_needs_replant', harvest_after=[soil_proof],
                   replant_evidence={**proof, 'plot': list(selected.box)}, replant_grid=None)
    return worker._plant_selected(opened, icon, key, selected, True, selected.center)


def rediscover_bare_field(worker, frame, icon, key, entry):
    """Replace stale geometry only before any planting gesture was attempted."""
    if (entry.get('stage') != 'harvested_needs_replant'
            or entry.get('replanted') or entry.get('plant_before') or entry.get('points')):
        return None
    from hayday.wheating_vision import WheatingVision
    vision = WheatingVision(cancel=worker.cancel_event.is_set)
    if not vision.farm(frame) or CameraNavigator._modal_visible(frame):
        return None
    candidates = vision.plots(frame, 'empty')
    if not candidates:
        return None
    fresh = worker._frame()
    if (not fresh.captured_at or fresh.captured_at == frame.captured_at
            or not vision.farm(fresh) or CameraNavigator._modal_visible(fresh)):
        return None
    origin = candidates[0].center
    moved = worker._translated_plot(frame, fresh, origin)
    checked = next((p for p in vision.plots(fresh, 'empty')
                    if moved is not None and np.linalg.norm(np.subtract(p.center, moved)) < 35), None)
    if checked is None:
        return None
    worker.soil_frame = fresh
    worker._picker_before = fresh
    opened = worker._tap(checked.center, fresh)
    opened, selected = worker._await_seed_picker(opened, checked.center)
    if selected is None:
        return ResourceResult('changed', 'Fresh bare soil was found; waiting for its seed picker to settle.')
    operation = entry['operation']+'_rediscovered_'+uuid.uuid4().hex
    from hayday.resources import _save_json
    _save_json(worker.state_path.parent/'farming_evidence'/f'{operation}.json',
               {'previous': entry, 'reason': 'Saved grid did not match; fresh bare soil and picker verified.'})
    soil_proof = worker._evidence(fresh, operation, 'harvest_clear')
    picker_proof = worker._evidence(opened, operation, 'replant_picker')
    worker._record(key, 'harvested_needs_replant', harvest_after=[soil_proof],
                   replant_evidence={**picker_proof, 'plot': list(selected.box)}, replant_grid=None)
    worker._replant_grid = None
    worker.progress('Wheating: rebuilding the planting grid from freshly verified bare soil.')
    return worker._plant_selected(opened, icon, key, selected, True, selected.center)

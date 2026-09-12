"""Restore the saved wheat workspace after a game restart, using farm landmarks."""
from __future__ import annotations

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import _decode


def farm_transform(before, after):
    """Allow zoom changes only for recovery; crop targeting keeps its own rules."""
    matrix = _farm_transform(before, after)
    return matrix if matrix is not None else _farm_transform(before, after, displaced=True)


def _farm_transform(before, after, *, displaced=False):
    if (before.width, before.height) != (after.width, after.height):
        return None
    images = [cv2.cvtColor(_decode(f.png), cv2.COLOR_BGR2GRAY) for f in (before, after)]
    height, width = images[0].shape
    mask = np.zeros_like(images[0])
    if displaced:
        # A reconnect can move the few shared farm landmarks to an edge. HUD
        # matches are discarded below in this fallback, not used as landmarks.
        mask[round(height*.06):round(height*.90), round(width*.10):round(width*.90)] = 255
    else:
        mask[round(height*.16):round(height*.84), round(width*.14):round(width*.86)] = 255
    sift = cv2.SIFT_create(nfeatures=3500)
    (old_keys, old_desc), (new_keys, new_desc) = [sift.detectAndCompute(im, mask) for im in images]
    if old_desc is None or new_desc is None:
        return None
    matches = cv2.BFMatcher().knnMatch(old_desc, new_desc, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .68*pair[1].distance]
    if displaced:
        good = [m for m in good if np.linalg.norm(np.subtract(
            old_keys[m.queryIdx].pt, new_keys[m.trainIdx].pt)) > 5*height/1080]
    if len(good) < 25:
        return None
    old = np.float32([old_keys[m.queryIdx].pt for m in good])
    new = np.float32([new_keys[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3)
    if (matrix is None or inliers is None or not np.isfinite(matrix).all()
            or inliers.sum() < max(22, len(good)*.55)):
        return None
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    tracked = old[inliers.ravel().astype(bool)]
    if (not .35 < scale < 3.5 or abs(matrix[1, 0]/scale) > .025
            or np.ptp(tracked[:, 0]) < width*.17 or np.ptp(tracked[:, 1]) < height*.13):
        return None
    return matrix


def _settled(run):
    previous = None
    for _ in range(8):
        run.wait(.2)
        frame = run.capture()
        if not run.vision.farm(frame) or CameraNavigator._modal_visible(frame):
            # Login tutorials can arrive after launcher recovery returns. Let
            # the normal capture handlers recognize/dismiss them on fresh frames
            # before abandoning workspace restoration.
            previous = None
            continue
        current = CameraNavigator._view(frame)
        if previous is not None and CameraNavigator._same_view(previous, current):
            return frame
        previous = current
    run.block('The farm camera did not settle after restarting.')


def restore_workspace(run, fields):
    reference = getattr(fields, '_known_before', None)
    points = getattr(fields, '_known_points', [])
    frame = _settled(run)
    bare = None
    worker = getattr(fields, 'worker', None)
    if worker:
        worker._bind_size(frame)
        pending = [e for e in worker.state['items'].values() if e.get('stage') in worker._PENDING]
        if len(pending) == 1 and pending[0].get('stage') == 'harvested_needs_replant':
            proofs = pending[0].get('harvest_after', [])
            worker.soil_frame = worker._saved_frame(proofs[-1]) if proofs else None
            from hayday.wheating_crop_resume import saved_soil_grid
            bare = saved_soil_grid(worker, pending[0])
            if bare:
                reference, points, _ = bare
    zooms = 0
    for step in range(7):
        run.check()
        if run.diagnostics:
            (run.diagnostics/f'restart_workspace_{step}.png').write_bytes(frame.png)
        bounds = run.vision.field_bounds(frame)
        shop = run.vision.shop_building(frame)
        if shop and run.vision.plots(frame, 'empty'):
            # Bare soil is a valid workspace, including when a legacy saved
            # grid is wrong. The crop worker verifies a fresh picker and grid.
            run._shop_anchor = frame, shop
            return frame
        # A just-harvested field has no yellow foliage bounds. Its verified
        # soil grid is the workspace; zooming repeatedly cannot reveal wheat.
        if bare and shop:
            from hayday.wheating_crop_resume import bare_grid
            if bare_grid(worker, reference, frame, points, bare[2]):
                run._shop_anchor = frame, shop
                return frame
        if (bounds and shop and bounds.x > frame.width*.12
                and bounds.x+bounds.width < frame.width*.88 and bounds.y > frame.height*.18
                and bounds.y+bounds.height < frame.height*.83):
            run._shop_anchor = frame, shop
            return frame
        matrix = farm_transform(reference, frame) if reference is not None and points else None
        scale = np.hypot(matrix[0, 0], matrix[1, 0]) if matrix is not None else None
        if zooms < 2 and (matrix is None or scale > 1.08):
            area = CameraNavigator._pinch_area(frame)
            if area is None:
                run.block('No clear grass is available to restore the farm zoom after restarting.')
            run.client.pinch_zoom_out(width=frame.width, height=frame.height,
                                     cancel_event=run.cancel_event, center=area[0], span=area[1])
            zooms += 1
        elif matrix is not None:
            projected = np.c_[np.asarray(points), np.ones(len(points))] @ matrix.T
            center = (projected.min(axis=0)+projected.max(axis=0))/2
            dx = float(np.clip(.58-center[0]/frame.width, -.30, .30))
            dy = float(np.clip(.52-center[1]/frame.height, -.28, .28))
            if abs(dx)+abs(dy) < .025:
                run.block('The saved farm area returned, but its wheat field and shop are not both recognized.')
            drag = CameraNavigator._grass_start(frame, dx, dy)
            if drag is None:
                run.block('No clear grass is available to restore the saved field position.')
            run.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=300)
        else:
            run.block('The saved wheat workspace could not be matched after restarting; no blind camera scan was sent.')
        frame = _settled(run)
    run.block('The saved wheat field and shop could not be restored after restarting.')

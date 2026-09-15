"""Restore the saved wheat workspace after a game restart, using farm landmarks."""
from __future__ import annotations

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import VisualTarget, _decode


def _distributed_translation(old, new, width, height):
    """Find a dominant translation despite unrelated repeated-tree matches."""
    offsets = new-old
    bins, counts = np.unique(np.floor(offsets/3).astype(int), axis=0, return_counts=True)
    modes = []
    for index in np.argsort(counts)[-16:]:
        center = bins[index]*3+1.5
        for _ in range(2):
            supported = np.linalg.norm(offsets-center, axis=1) <= 2.5
            if not supported.any():
                break
            center = np.median(offsets[supported], axis=0)
        supported = np.linalg.norm(offsets-center, axis=1) <= 2.5
        modes.append((int(supported.sum()), center, supported))
    if not modes:
        return None
    count, shift, supported = max(modes, key=lambda value: value[0])
    if count < 40 or any(n >= count*.8 and np.linalg.norm(center-shift) > 5
                         for n, center, _ in modes):
        return None
    tracked = old[supported]
    cells = {(int(x/(width/4)), int(y/(height/4))) for x, y in tracked}
    if (len(cells) < 4 or len(np.unique(np.rint(tracked), axis=0)) < 25
            or np.ptp(tracked[:, 0]) < width*.25 or np.ptp(tracked[:, 1]) < height*.20):
        return None
    return np.float64([[1, 0, shift[0]], [0, 1, shift[1]]])


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
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    if inliers.sum() < max(22, len(good)*.55):
        # A large shop return shares only a narrow part of the farm; repeated
        # leaves elsewhere create many unrelated descriptor matches. Accept a
        # pure translation only when at least 40 distributed landmarks agree
        # within 2.5 pixels. This fallback cannot authorize a zoom or rotation.
        return _distributed_translation(old, new, width, height) if displaced else None
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


def _field_center(bounds, soil, matrix, points):
    if bounds is not None:
        return bounds.center
    if soil:
        return ((min(p.x for p in soil)+max(p.x+p.width for p in soil))/2,
                (min(p.y for p in soil)+max(p.y+p.height for p in soil))/2)
    if matrix is not None:
        projected = np.c_[np.asarray(points), np.ones(len(points))] @ matrix.T
        return (projected.min(axis=0)+projected.max(axis=0))/2
    return None


def _workspace_drag(frame, center, shop):
    if center is not None:
        # Leave room below and left of the field for the roadside shop.
        dx, dy = .58-center[0]/frame.width, .48-center[1]/frame.height
    elif shop is not None:
        dx, dy = .34-shop.center[0]/frame.width, .67-shop.center[1]/frame.height
    else:
        # Restart opens around the farmhouse. A short right/up grass swipe
        # reveals the roadside area to its left and lifts the low field into
        # view. This is a bounded search gesture, never a saved tap coordinate.
        dx, dy = .30, -.10
    dx, dy = float(np.clip(dx, -.30, .30)), float(np.clip(dy, -.28, .28))
    if abs(dx)+abs(dy) < .025:
        return None
    return next((drag for fraction in (1., .5, .25)
                 if (drag := CameraNavigator._grass_start(frame, dx*fraction, dy*fraction))), None)


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
    unchanged = 0
    # At most seven gestures, including at most one zoom, with a final
    # observation. The zoom budget belongs to the running game, not this call:
    # local recovery retries must not start zooming out all over again.
    for step in range(8):
        run.check()
        if run.diagnostics:
            (run.diagnostics/f'restart_workspace_{step}.png').write_bytes(frame.png)
        bounds = run.vision.field_bounds(frame)
        if bounds is None:
            bounds = getattr(run.vision, 'edge_field_bounds', lambda _: None)(frame)
        shop = run.vision.shop_building(frame)
        soil = run.vision.plots(frame, 'empty')
        group = getattr(fields, '_group', None)
        if bounds is None and shop and group is not None and len(group.points) >= 3:
            view = group.observe(frame)
            if (view is not None and len(view.cells) >= len(group.points)
                    and view.count('growing') == len(view.cells)):
                # Current crop pixels must support every saved tile. A growing
                # field has neither bare furrows nor yellow harvest foliage.
                x, y = np.asarray(view.cells, dtype=object)[:, :2].astype(float).T
                dx, dy = view.pitch
                bounds = VisualTarget(round(x.min()-dx), round(y.min()-dy),
                    round(np.ptp(x)+2*dx), round(np.ptp(y)+2*dy), 1.)
        if shop and soil:
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
        if step == 7:
            break
        if unchanged >= 2:
            run.block('The farm camera did not move toward the roadside shop after two grass drags.')
        matrix = farm_transform(reference, frame) if reference is not None and points else None
        scale = np.hypot(matrix[0, 0], matrix[1, 0]) if matrix is not None else None
        center = _field_center(bounds, soil, matrix, points)
        too_large = bounds is not None and (bounds.width > frame.width*.74
                                            or bounds.height > frame.height*.60)
        needs_zoom = (too_large or (scale is not None and scale > 1.08)
                      or (center is None and shop is None))
        if not getattr(run, '_workspace_zoom_attempted', False) and needs_zoom:
            run._workspace_zoom_attempted = True
            area = CameraNavigator._pinch_area(frame)
            if area is not None:
                run.publish('Wheating: zooming out once before positioning the roadside shop and field.')
                run.check()
                run.client.pinch_zoom_out(width=frame.width, height=frame.height,
                                         cancel_event=run.cancel_event, center=area[0], span=area[1])
                frame = _settled(run)
                continue
        drag = _workspace_drag(frame, center, shop)
        if drag is None:
            run.block('The roadside workspace is not recognized or has no clear grass for camera positioning.')
        run.publish('Wheating: moving the camera toward the roadside shop and field.')
        before = CameraNavigator._view(frame)
        run.check()
        run.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=300)
        frame = _settled(run)
        moved = not CameraNavigator._same_view(before, CameraNavigator._view(frame))
        unchanged = 0 if moved else unchanged+1
    run.block('The saved wheat field and shop could not be restored after restarting.')

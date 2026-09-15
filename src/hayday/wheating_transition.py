"""Recognize cleared soil at the already verified wheat-grid coordinates."""
import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import VisualTarget, _decode


def tracked_grid(worker, frame):
    plan = getattr(worker, 'harvest_grid_plan', None) or (worker.harvest_plan if worker.harvest_plot_centers else None)
    if not plan:
        return None
    before, points = plan
    if len(points) < 4 or CameraNavigator._modal_visible(frame):
        return None
    origin = points[0]
    moved = worker._translated_plot(before, frame, origin, require_visible=False)
    if moved is None:
        return None
    coords = np.asarray(points)
    pitch = []
    for axis in (0, 1):
        differences = np.diff(np.unique(coords[:, axis]))
        if not len(differences):
            return None
        pitch.append(float(np.median(differences)))
    dx, dy = pitch
    base = frame.height/1080
    if not (20*base < dx < 100*base and 10*base < dy < 60*base and 1.6 < dx/dy < 2.4):
        return None
    centers = coords+np.subtract(moved, origin)+(0, round(16*base))
    if any(not (frame.width*.13 < x < frame.width*.87 and frame.height*.2 < y < frame.height*.84)
           for x, y in centers):
        return None
    return [tuple(map(int, p)) for p in centers], (dx, dy)


def tracked_soil(worker, frame):
    grid = tracked_grid(worker, frame)
    if grid is None:
        return None
    centers, (dx, dy) = grid
    hsv = cv2.cvtColor(_decode(frame.png), cv2.COLOR_BGR2HSV)
    destination = np.array((frame.width*.49, frame.height*.59))
    order = sorted(range(len(centers)), key=lambda i: np.linalg.norm(np.subtract(centers[i], destination)))
    for index in order:
        center = centers[index]
        # The picker recenters the selected tile. Do not choose an exposed
        # edge that would push the opposite end of the field behind the HUD.
        projected = np.asarray(centers)+destination-center
        if any(not (frame.width*.14 < x < frame.width*.86 and frame.height*.21 < y < frame.height*.83)
               for x, y in projected):
            continue
        pixels = worker.vision.tile_pixels(hsv, center, dx, dy)
        if pixels is None:
            continue
        soil = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 16) & (pixels[:, 1] >= 65) & (pixels[:, 2] < 245)
        if soil.mean() < .93:
            continue
        # Only this soil tap is authorized here. Planting separately verifies
        # the wheat picker and aligns the entire known grid; every plot still
        # has to confirm growth after the one planting sweep.
        before, original = getattr(worker, 'harvest_grid_plan', None) or worker.harvest_plan
        original = [(x, y+round(16*frame.height/1080)) for x, y in original]
        worker._replant_grid = before, original, (dx, dy)
        worker._replant_origin = original[index]
        # Reuse the translation already proven while locating this soil tile;
        # residual diagnostics must not pay for a second feature alignment.
        worker._harvest_shift = np.subtract(center, original[index])
        x, y = center
        return VisualTarget(x-8, y-8, 16, 16, 1.)
    return None


def saved_grid(worker, entry):
    proof = entry.get('replant_grid')
    if not isinstance(proof, dict):
        return None
    frame = worker._saved_frame(proof)
    points, pitch = proof.get('points'), proof.get('pitch')
    if frame is None or not isinstance(points, list) or not 4 <= len(points) <= 512:
        return None
    if (any(not isinstance(p, list) or len(p) != 2 or any(type(n) is not int for n in p)
            or not 0 < p[0] < frame.width or not 0 < p[1] < frame.height for p in points)
            or len({tuple(p) for p in points}) != len(points)
            or not isinstance(pitch, list) or len(pitch) != 2
            or any(type(n) not in (float, int) or not np.isfinite(n) for n in pitch)):
        return None
    dx, dy = pitch
    base = frame.height/1080
    if not (20*base < dx < 100*base and 10*base < dy < 60*base and 1.6 < dx/dy < 2.4):
        return None
    return frame, [tuple(p) for p in points], (dx, dy)


def planting_grid(worker, frame, plot, grid):
    before, points, (dx, dy) = grid
    if not worker.vision.full_outline:
        return None
    if abs((plot.width-6)/2-dx) > dx*.12 or abs((plot.height-6)/2-dy) > dy*.15:
        return None
    moved = worker._translated_plot(before, frame, points[0], require_visible=False)
    if moved is None:
        return None
    centers = np.asarray(points)+np.subtract(moved, points[0])
    selected = int(np.argmin(np.linalg.norm(centers-plot.center, axis=1)))
    if np.linalg.norm(centers[selected]-plot.center) > 6*frame.height/1080:
        return None
    # Use the observed outline as the exact seed anchor, retaining grid pitch.
    centers += np.subtract(plot.center, centers[selected])
    if any(not (frame.width*.12 < x < frame.width*.88 and frame.height*.18 < y < frame.height*.88)
           for x, y in centers):
        return None
    return [tuple(map(int, centers[i])) for i in [selected, *range(selected), *range(selected+1, len(points))]]

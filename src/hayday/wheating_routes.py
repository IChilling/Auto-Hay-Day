"""Tile-aligned, bounded routes. Route coverage never depends on tile counts."""
from __future__ import annotations

import math

import numpy as np


def tile_slope(points):
    """Measure the isometric axes, with bounded work even on large layouts."""
    coords = np.asarray(points, dtype=float)
    if len(coords) < 2:
        return .5
    # A representative subset is sufficient for the camera's common angle.
    sample = coords[::max(1, math.ceil(len(coords)/64))]
    delta = np.abs(sample[:, None, :]-coords[None, :, :])
    dx, dy = delta[:, :, 0], delta[:, :, 1]
    slopes = dy/np.maximum(dx, 1.)
    distance = np.hypot(dx, dy)
    valid = (dx > 5) & (dy > 3) & (slopes > .40) & (slopes < .65)
    nearest = np.where(valid, distance, np.inf).argmin(axis=1)
    rows = np.arange(len(sample))
    usable = valid[rows, nearest]
    if not np.any(usable):
        return .5
    # Fit across longer collinear neighbors as well, avoiding a one-pixel
    # rounding bias on small/odd-sized tiles (13 versus 14 pixels of rise).
    approximate = float(np.median(slopes[rows, nearest][usable]))
    aligned = valid & (np.abs(slopes-approximate) < .04)
    return float(np.sum(dx[aligned]*dy[aligned])/np.sum(dx[aligned]**2))


def diagonal_order(points):
    """Visit exact centers along successive diagonal rows in O(n log n)."""
    points = list(dict.fromkeys(tuple(map(int, p)) for p in points))
    if len(points) < 3:
        return points
    slope = tile_slope(points)
    candidates = []
    for sign in (-1, 1):
        # A few raster pixels of row jitter must not split a long row.
        ordered = sorted(points, key=lambda p: (p[1]-sign*slope*p[0], p[0]))
        rows = []
        intercept = None
        tolerance = max(2., min(6., np.ptp(np.asarray(points)[:, 1])*.015))
        for point in ordered:
            value = point[1]-sign*slope*point[0]
            if intercept is None or value-intercept > tolerance:
                rows.append([])
                intercept = value
            rows[-1].append(point)
        for reverse in (False, True):
            for side in (False, True):
                route = [points[0]]
                for number, row in enumerate(reversed(rows) if reverse else rows):
                    route.extend(p for p in sorted(row, reverse=bool(number % 2) ^ side)
                                 if p != points[0])
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))
                candidates.append((length, route))
    return min(candidates, key=lambda item: item[0])[1]


def diagonal_trace(points, *, slope=None, bounds=None):
    """Join row ends along either tile axis, retaining all exact destinations.

    Extra corner points are travel, not additional claimed or counted plots.
    Choose the inward corner near the viewport edge. The seed/sickle pickup
    segment is deliberately supplied separately by the worker.
    """
    if not points:
        return []
    slope = slope or tile_slope(points)
    route = [tuple(map(int, points[0]))]
    pending = list(reversed(points[1:]))
    while pending:
        end = pending.pop()
        end = tuple(map(int, end))
        x, y = route[-1]
        dx, dy = end[0]-x, end[1]-y
        if not dx and not dy:
            continue
        if abs(abs(dy)-slope*abs(dx)) > 2:
            corners = []
            for sign in (-1, 1):
                travel = (dx+dy/(sign*slope))/2
                corner = (round(x+travel), round(y+sign*slope*travel))
                if bounds is None or (bounds[0] <= corner[0] <= bounds[2]
                                      and bounds[1] <= corner[1] <= bounds[3]):
                    corners.append(corner)
            if corners:
                center = ((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2) if bounds else end
                corner = min(corners, key=lambda p: math.dist(p, center))
                if corner != route[-1] and corner != end:
                    route.append(corner)
            elif (bounds is not None and max(abs(dx), abs(dy)) > 4
                  and all(bounds[0] <= p[0] <= bounds[2] and bounds[1] <= p[1] <= bounds[3]
                          for p in ((x, y), end))):
                # A wide, shallow field may not fit either single corner.
                # Split the crossing into smaller diagonal zigzags, never a
                # long horizontal fallback or a detour outside the viewport.
                midpoint = (round((x+end[0])/2), round((y+end[1])/2))
                pending.extend((end, midpoint))
                continue
        route.append(end)
    return route


def diagonal_mask_sweep(mask, *, step, margin=0, slope=.5):
    """Cover the whole observed footprint with overlapping diagonal lanes.

    Bin every foreground pixel, rather than sampling isolated scan lines: a
    small edge clump or a one-pixel neck cannot disappear between lanes.
    Endpoints include a modest overrun. No rectangular farm shape is assumed.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    intercepts = ys-slope*xs
    origin = float(intercepts.min())
    bins = np.floor((intercepts-origin)/step).astype(np.int32)
    count = int(bins.max())+1
    if count > 512:
        return []
    left, right = np.full(count, mask.shape[1], dtype=float), np.full(count, -1., dtype=float)
    np.minimum.at(left, bins, xs)
    np.maximum.at(right, bins, xs)
    path = []
    for row in range(count):
        if right[row] < 0:
            continue
        intercept = origin+(row+.5)*step
        ends = [(max(0, round(left[row]-margin)), 0),
                (min(mask.shape[1]-1, round(right[row]+margin)), 0)]
        ends = [(x, min(mask.shape[0]-1, max(0, round(intercept+slope*x)))) for x, _ in ends]
        path.extend(reversed(ends) if row % 2 else ends)
    return path

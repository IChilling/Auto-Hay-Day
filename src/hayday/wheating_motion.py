"""Bounded movement variation over already verified wheat coordinates."""
from __future__ import annotations

import math
import random


def route_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))


class WheatMotion:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self._previous = {}

    def _choose(self, candidates, kind):
        unique = {tuple(route): route for route in candidates}
        choices = list(unique.values())

        def signature(route):
            x, y = route[0]
            return tuple((px-x, py-y) for px, py in route)

        alternatives = [route for route in choices
                        if signature(route) != self._previous.get(kind)]
        chosen = self.rng.choice(alternatives or choices)
        self._previous[kind] = signature(chosen)
        return list(chosen)

    def plots(self, points, kind):
        """Keep the selected tile first and visit every exact center once."""
        points = list(points)
        if len(points) < 3:
            return points
        distances = [[math.dist(a, b) for b in points] for a in points]

        def ordered(vary, heading=None):
            route, remaining = [0], set(range(1, len(points)))
            while remaining:
                nearest = min(remaining, key=lambda i: (distances[route[-1]][i], i))
                if vary:
                    limit = distances[route[-1]][nearest]*1.08
                    nearby = [i for i in sorted(remaining) if distances[route[-1]][i] <= limit]
                    if heading is None:
                        nearest = self.rng.choice(nearby)
                    else:
                        x, y = points[route[-1]]
                        nearest = max(nearby, key=lambda i: (points[i][0]-x)*heading[0]
                                      + (points[i][1]-y)*heading[1])
                route.append(nearest)
                remaining.remove(nearest)
            return [points[i] for i in route]

        baseline = ordered(False)
        length_limit = route_length(baseline)*1.06
        hop_limit = max(math.dist(a, b) for a, b in zip(baseline, baseline[1:], strict=False))
        candidates = [baseline]
        # Directional sweeps give regular large fields useful alternatives
        # without depending on random walks finding an efficient full route.
        headings = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]
        trials = [ordered(True, heading) for heading in headings]
        trials.extend(ordered(True) for _ in range(64))
        for route in trials:
            if (route_length(route) <= length_limit
                    and all(math.dist(a, b) <= hop_limit+1e-6
                            for a, b in zip(route, route[1:], strict=False))):
                candidates.append(route)
        return self._choose(candidates, kind)

    def sweep(self, points):
        """Vary row direction while preserving every foliage scan segment."""
        points = list(points)
        # The first point is the selected crop. Only the paired horizontal
        # scan lines have interchangeable directions; arbitrary traces do not.
        rows = [points[i:i+2] for i in range(1, len(points), 2)]
        if not rows or any(len(row) != 2 or row[0][1] != row[1][1] for row in rows):
            return points
        candidates = []
        for reverse_rows in (False, True):
            for reverse_sides in (False, True):
                chosen = rows[::-1] if reverse_rows else rows
                route = [points[0], *(p for row in chosen
                                     for p in (row[::-1] if reverse_sides else row))]
                if route_length(route) <= route_length(points)*1.06:
                    candidates.append(route)
        return self._choose(candidates, 'foliage')

    def timing(self, segments):
        """Small, smoothly changing delays; never faster than verified timing."""
        value = self.rng.uniform(1., 1.10)
        factors = []
        for _ in range(segments):
            value = .65*value+.35*self.rng.uniform(1., 1.10)
            factors.append(value)
        return tuple(factors)

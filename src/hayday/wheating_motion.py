"""Bounded movement variation over already verified wheat coordinates."""
from __future__ import annotations

import math
import random


def route_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))


class WheatMotion:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()

    def plots(self, points, kind):
        """Follow tile-aligned rows without quadratic random route searches."""
        from hayday.wheating_routes import diagonal_order
        return diagonal_order(points)

    def sweep(self, points):
        """Keep the complete diagonal coverage lanes in their planned order."""
        return list(points)

    def timing(self, segments):
        """Small, smoothly changing delays; never faster than verified timing."""
        value = self.rng.uniform(1., 1.10)
        factors = []
        for _ in range(segments):
            value = .65*value+.35*self.rng.uniform(1., 1.10)
            factors.append(value)
        return tuple(factors)

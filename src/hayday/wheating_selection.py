"""Fresh wheat selection shared directly with the single harvest gesture."""
from __future__ import annotations

import time
from dataclasses import dataclass

from hayday.adb import Screenshot
from hayday.farming import FarmingWorker
from hayday.farming_vision import HarvestTarget


def same_harvest(first, second, frame):
    """Keep the tool fixed while allowing small changes in swaying white tips."""
    if not first or not second or not FarmingWorker._same_target(first.tool, second.tool, frame):
        return False
    a, b = first.highlight, second.highlight
    if a is None or b is None:
        return False
    if any(abs(x-y) > max(4, 8*frame.height/1080)
           for x, y in zip(a.center, b.center, strict=True)):
        return False
    overlap = max(0, min(a.x+a.width, b.x+b.width)-max(a.x, b.x))*max(
        0, min(a.y+a.height, b.y+b.height)-max(a.y, b.y))
    return overlap >= .75*max(a.width*a.height, b.width*b.height, 1)


@dataclass(frozen=True)
class PreparedHarvest:
    frame: Screenshot
    harvest: HarvestTarget
    path: tuple[tuple[int, int], ...]
    plan: object
    serial: str
    observed_at: float

    def current(self, worker, frame):
        return (self.frame is frame and self.plan is worker.harvest_plan
                and self.serial == worker.serial == worker.client.serial
                and 0 <= time.monotonic()-self.observed_at <= 1.)


def select_wheat(run, worker, before, target):
    """Two actual captures confirm the tool/crop; the last one owns the route."""
    worker._prepared_harvest = None
    # The sickle expands for about 200 ms when opening. Capturing during that
    # animation forces a broad scale search and another stabilization frame.
    run.tap(target.center, before, settle=.25)
    previous_frame = previous = None
    last_capture = 0.
    for _ in range(6):
        # Screenshot transfer and recognition normally exceed this interval.
        # Fast devices still receive independently sampled observations.
        run.wait(max(0., .10-(time.monotonic()-last_capture)))
        last_capture = time.monotonic()
        selected = run.capture(fast=True)
        observed_at = time.monotonic()
        expected = worker._translated_plot(before, selected, target.center)
        harvest = worker.vision.harvest(selected.png)
        control = harvest.highlight if harvest else None
        associated = (expected is not None and control is not None
                      and sum((a-b)**2 for a, b in zip(expected, control.center, strict=True))
                      < max(80, control.width, control.height)**2)
        distinct = (previous_frame is not None and bool(selected.captured_at)
                    and selected.captured_at != previous_frame.captured_at)
        if associated and distinct and same_harvest(previous, harvest, selected):
            path = worker._planned_harvest(selected, harvest)
            if path:
                worker._prepared_harvest = PreparedHarvest(selected, harvest, tuple(path),
                    worker.harvest_plan, worker.serial, observed_at)
                return selected, expected, control, True
        previous = harvest if associated else None
        previous_frame = selected
    return selected, expected, control, False

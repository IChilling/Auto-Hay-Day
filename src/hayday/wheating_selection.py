"""Fresh wheat selection shared directly with the single harvest gesture."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import cv2

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker
from hayday.farming_vision import HarvestTarget
from hayday.wheating import WheatingPrerequisite, WheatingStaleControl


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


def _wheat_at(run, frame, point):
    """Use the same local crop evidence as the verified-layout selector."""
    x, y = map(int, point)
    if not (frame.width*.12 < x < frame.width*.88 and frame.height*.20 < y < frame.height*.83):
        return False
    image = run.vision._image_for(frame.png)
    patch = cv2.cvtColor(image[y-8:y+9, x-12:x+13], cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(patch, (18, 140, 160), (31, 255, 255)) > 0).mean()) >= .4


def _tap_fresh_wheat(run, worker, before, target):
    # Layout projection and route building can outlive the input deadline under
    # fleet load. Keep that route anchored to its original proof, but reobserve
    # the selection point after those calculations. Do not recompute the whole
    # layout in the final capture-to-tap interval.
    previous_stamp = before.captured_at
    for attempt in range(3):
        run.check()
        fresh = run.capture(fast=True)
        distinct = bool(fresh.captured_at) and fresh.captured_at != previous_stamp
        previous_stamp = fresh.captured_at
        if (not distinct or (fresh.width, fresh.height) != (before.width, before.height)
                or not run.vision.farm(fresh) or CameraNavigator._modal_visible(fresh)
                or run.client.foreground_package() != 'com.supercell.hayday'):
            continue
        point = worker._translated_plot(before, fresh, target.center)
        if point is None or not _wheat_at(run, fresh, point):
            continue
        checked = replace(target, x=point[0]-target.width//2, y=point[1]-target.height//2)
        if run.client.foreground_package() != 'com.supercell.hayday':
            continue
        try:
            run.tap(checked.center, fresh, settle=.25)
        except WheatingStaleControl:
            # This exception is raised only before input. ADB errors or an
            # uncertain post-tap outcome must never repeat the selection tap.
            run.publish(f'Wheating: refreshing expired wheat selection ({attempt+1}/3); no tap was sent.')
            continue
        return fresh, checked
    raise WheatingPrerequisite('A fresh wheat selection could not be confirmed after three observations; no selection tap was sent.')


def select_wheat(run, worker, before, target):
    """Two actual captures confirm the tool/crop; the last one owns the route."""
    worker._prepared_harvest = None
    # The sickle expands for about 200 ms when opening. Capturing during that
    # animation forces a broad scale search and another stabilization frame.
    before, target = _tap_fresh_wheat(run, worker, before, target)
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

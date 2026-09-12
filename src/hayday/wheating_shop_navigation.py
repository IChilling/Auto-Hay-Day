"""Clear verified crop controls without guessing a hidden shop position."""
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker


def dismiss_crop_controls(run, frame):
    detect = getattr(run.vision, 'crop_controls', None)
    overlay = detect(frame) if detect else None
    if overlay is None or not run.vision.farm(frame) or CameraNavigator._modal_visible(frame):
        return None
    fresh = run.capture(fast=True)
    checked = detect(fresh)
    if (not run.vision.farm(fresh) or CameraNavigator._modal_visible(fresh)
            or not fresh.captured_at or fresh.captured_at == frame.captured_at
            or checked is None or checked[0] != overlay[0]
            or not FarmingWorker._same_target(overlay[1], checked[1], fresh)):
        return None
    grass = CameraNavigator._grass_start(fresh, 0, 0)
    if grass is None:
        return None
    run.tap(grass[:2], fresh, settle=.15)
    prior = None
    for _ in range(8):
        after = run.capture(fast=True)
        if (run.vision.farm(after) and not CameraNavigator._modal_visible(after)
                and detect(after) is None):
            view = CameraNavigator._view(after)
            if (prior is not None and after.captured_at and after.captured_at != prior[0]
                    and CameraNavigator._same_view(prior[1], view)):
                run.publish('Wheating: closed crop controls covering the roadside shop.')
                return after
            prior = after.captured_at, view
        else:
            prior = None
        run.wait(.1)
    run.block('The crop controls did not clear before opening the roadside shop.')

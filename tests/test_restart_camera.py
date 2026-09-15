"""Camera recovery must find the roadside field without a saved-image match."""
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hayday import wheating_restart_camera as camera
from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingBlocked, WheatingCancelled, WheatingRunner
from hayday.wheating_restart import WheatRestart, WheatStalled

SHOP = VisualTarget(550, 630, 230, 230, 1.)
FIELD = VisualTarget(600, 280, 750, 380, 1.)


@pytest.fixture
def scene(monkeypatch):
    frames = [Screenshot(str(i).encode(), 1920, 1080, str(i)) for i in range(10)]
    vision = SimpleNamespace(field_bounds=Mock(return_value=None),
                             shop_building=Mock(return_value=None),
                             plots=Mock(return_value=()))
    run = SimpleNamespace(client=Mock(spec=['swipe', 'pinch_zoom_out']),
                          vision=vision, diagnostics=None, check=Mock(), publish=Mock(),
                          cancel_event=threading.Event(), block=WheatingRunner.block)
    monkeypatch.setattr(camera, '_settled', Mock(side_effect=frames))
    monkeypatch.setattr(camera, 'farm_transform', Mock(return_value=None))
    monkeypatch.setattr(camera.CameraNavigator, '_pinch_area', Mock(return_value=((800, 500), 400)))
    monkeypatch.setattr(camera.CameraNavigator, '_view', lambda f: f.captured_at)
    monkeypatch.setattr(camera.CameraNavigator, '_same_view', lambda a, b: a == b)
    monkeypatch.setattr(camera.CameraNavigator, '_grass_start',
                        lambda f, dx, dy: (800, 500, 800+round(f.width*dx), 500+round(f.height*dy)))
    return run, frames


def ready_on(run, frame, *, bare=False):
    run.vision.shop_building.side_effect = lambda f: SHOP if f is frame else None
    if bare:
        run.vision.plots.side_effect = lambda f, kind: (FIELD,) if f is frame else ()
    else:
        run.vision.field_bounds.side_effect = lambda f: FIELD if f is frame else None


@pytest.mark.parametrize('bare', [False, True])
def test_unmatched_restart_zooms_once_then_pans_to_wheat_or_soil(scene, bare):
    run, frames = scene
    ready_on(run, frames[2], bare=bare)
    assert camera.restore_workspace(run, None) is frames[2]
    assert run._shop_anchor == (frames[2], SHOP)
    run.client.pinch_zoom_out.assert_called_once()
    run.client.swipe.assert_called_once()
    x, y, end_x, end_y = run.client.swipe.call_args.args
    # The recorded restarted view puts the field off the left/lower edge:
    # bring that ground right and slightly up, not farther out of the screen.
    assert end_x > x and end_y < y
    assert abs(end_y-y) < abs(end_x-x)


@pytest.mark.parametrize('bare', [False, True])
def test_visible_field_guides_pan_without_saved_geometry_or_more_zoom(scene, bare):
    run, frames = scene
    clipped = VisualTarget(80, 670, 300, 200, 1.)
    ready_on(run, frames[1], bare=bare)
    if bare:
        run.vision.plots.side_effect = lambda f, kind: (FIELD if f is frames[1] else clipped,)
    else:
        run.vision.field_bounds.side_effect = lambda f: FIELD if f is frames[1] else clipped
    assert camera.restore_workspace(run, None) is frames[1]
    run.client.pinch_zoom_out.assert_not_called()
    x, y, end_x, end_y = run.client.swipe.call_args.args
    assert end_x > x and end_y < y


def test_unchanged_zoom_does_not_prevent_panning_with_a_saved_match(scene):
    run, frames = scene
    ready_on(run, frames[2])
    # Even a stale reference that keeps reporting a larger scale must not
    # force another pinch before repositioning the current field.
    camera.farm_transform.return_value = np.array([[1.4, 0, 0], [0, 1.4, 0]])
    fields = SimpleNamespace(_known_before=frames[-1], _known_points=[(100, 500), (250, 600)])
    assert camera.restore_workspace(run, fields) is frames[2]
    run.client.pinch_zoom_out.assert_called_once()
    run.client.swipe.assert_called_once()


def test_local_retry_does_not_reset_zoom_budget_and_stops_at_camera_edge(scene, monkeypatch):
    run, frames = scene
    monkeypatch.setattr(camera, '_settled', Mock(return_value=frames[0]))
    for _ in range(2):
        with pytest.raises(WheatingBlocked, match='after two grass drags'):
            camera.restore_workspace(run, None)
    run.client.pinch_zoom_out.assert_called_once()
    assert run.client.swipe.call_count == 4


def test_missing_pinch_origin_falls_back_to_a_grass_pan(scene):
    run, frames = scene
    camera.CameraNavigator._pinch_area.return_value = None
    ready_on(run, frames[1])
    assert camera.restore_workspace(run, None) is frames[1]
    run.client.pinch_zoom_out.assert_not_called()
    run.client.swipe.assert_called_once()


def test_top_edge_field_guides_camera_down_instead_of_blind_upward_search(scene):
    run, frames = scene
    ready_on(run, frames[1])
    run.vision.edge_field_bounds = Mock(return_value=VisualTarget(1032, 43, 438, 179, .90))
    assert camera.restore_workspace(run, None) is frames[1]
    run.client.pinch_zoom_out.assert_not_called()
    _, y, _, end_y = run.client.swipe.call_args.args
    assert end_y > y


def test_pan_uses_shorter_drag_when_full_distance_has_no_clear_origin(scene, monkeypatch):
    run, frames = scene
    run._workspace_zoom_attempted = True
    ready_on(run, frames[1])
    grass = Mock(side_effect=[None, (800, 500, 1088, 446)])
    monkeypatch.setattr(camera.CameraNavigator, '_grass_start', grass)
    assert camera.restore_workspace(run, None) is frames[1]
    assert grass.call_args.args[1:] == (.15, -.05)
    run.client.swipe.assert_called_once_with(800, 500, 1088, 446,
                                           width=1920, height=1080, duration_ms=300)


def test_no_clear_grass_never_sends_a_pan(scene, monkeypatch):
    run, _ = scene
    run._workspace_zoom_attempted = True
    monkeypatch.setattr(camera.CameraNavigator, '_grass_start', lambda *args: None)
    with pytest.raises(WheatingBlocked, match='no clear grass'):
        camera.restore_workspace(run, None)
    assert not run.client.mock_calls


def test_recovery_is_bounded_but_checks_the_last_gesture(scene):
    run, frames = scene
    ready_on(run, frames[7])
    assert camera.restore_workspace(run, None) is frames[7]
    assert run.client.swipe.call_count == 6
    run.client.pinch_zoom_out.assert_called_once()


def test_unrecognized_farm_stops_after_gesture_budget(scene):
    run, _ = scene
    with pytest.raises(WheatingBlocked, match='could not be restored'):
        camera.restore_workspace(run, None)
    assert run.client.swipe.call_count == 6
    run.client.pinch_zoom_out.assert_called_once()


def test_cancellation_after_recognition_prevents_camera_input(scene):
    run, _ = scene
    run.check.side_effect = [None, WheatingCancelled('cancelled')]
    with pytest.raises(WheatingCancelled):
        camera.restore_workspace(run, None)
    assert not run.client.mock_calls


def test_successful_app_restart_resets_the_zoom_budget(scene, tmp_path, monkeypatch):
    simulated, frames = scene
    ready_on(simulated, frames[2])
    client = Mock(serial='test')
    client.hay_day_running.return_value = False
    client.foreground_package.return_value = 'com.supercell.hayday'
    run = WheatingRunner(client, tmp_path, vision=simulated.vision)
    run._lock_file = object()
    run._workspace_zoom_attempted = True
    run._recovery = SimpleNamespace(host_popup=Mock())
    run.wait = Mock()
    run.publish = Mock()
    run.save_json = Mock()
    launch = Mock()
    launch.process.return_value = frames[1]
    monkeypatch.setattr('hayday.wheating_recovery.WheatRecovery',
                        lambda run: SimpleNamespace(launch=launch))
    monkeypatch.setattr('hayday.wheating_vision.WheatingVision',
                        lambda **kwargs: simulated.vision)
    restart = WheatRestart(run)
    restart._capture = Mock(return_value=frames[0])
    restart.recover(WheatStalled('stalled'), None)
    assert restart.stage == 'ready'
    client.force_stop_hay_day.assert_called_once()
    client.pinch_zoom_out.assert_called_once()
    client.swipe.assert_called_once()


def test_recorded_offscreen_restart_has_safe_pan_toward_the_field():
    from hayday.wheating_vision import WheatingVision

    path = Path(__file__).parent/'fixtures/restart_offscreen.png'
    frame = Screenshot(path.read_bytes(), 1920, 1080, 'restart')
    vision = WheatingVision()
    assert vision.farm(frame)
    assert not camera.CameraNavigator._modal_visible(frame)
    assert vision.field_bounds(frame) is None
    assert vision.shop_building(frame) is None
    assert not vision.plots(frame, 'empty')
    drag = camera._workspace_drag(frame, None, None)
    assert drag is not None
    x, y, end_x, end_y = drag
    assert end_x > x and end_y < y

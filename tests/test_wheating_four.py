"""Regressions recorded while four different MuMu farms ran concurrently."""
import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from hayday.ad_text import AdTextLine, AdTextObservation
from hayday.adb import Screenshot
from hayday.game_scene import FarmSceneVision
from hayday.resource_vision import _decode
from hayday.wheating import WheatingBlocked, WheatingCancelled, WheatingPrerequisite, WheatingRunner
from hayday.wheating_event_board import WheatEventBoard
from hayday.wheating_fields import WheatFields
from hayday.wheating_recovery import WheatLaunchRecovery
from hayday.wheating_restart_camera import restore_workspace
from hayday.wheating_vision import WheatingVision

ROOT = Path(__file__).parent/'fixtures/mumu_four'


def shot(name, at='first'):
    return Screenshot((ROOT/(name+'.png')).read_bytes(), 1920, 1080, at)


def handler():
    catalog = json.loads((ROOT/'text.json').read_text())
    readings = {shot(name).png: AdTextObservation(lines=tuple(AdTextLine(**line) for line in lines))
                for name, lines in catalog.items()}
    text = Mock(read=lambda png: readings.get(png, AdTextObservation()))
    run = SimpleNamespace(vision=WheatingVision(text_reader=text), diagnostics=None,
        client=SimpleNamespace(serial='test', foreground_package=Mock(return_value='com.supercell.hayday')),
        cancel_event=threading.Event(), wait=Mock(), publish=Mock(), tap=Mock())
    return run, WheatEventBoard(run)


@pytest.mark.parametrize('page', ['unlocked', 'board', 'posters', 'unveil_poster', 'info',
                                  'open_details', 'close_details', 'complete'])
def test_recorded_introduction_controls(page):
    run, recovery = handler()
    observed = recovery.observe(shot(page))
    assert observed is not None and observed[0] == page
    assert recovery.observe(shot('growing')) is None
    run.tap.assert_not_called()


def test_animated_poster_remains_an_unveil_action_when_ocr_loses_its_badge():
    run, recovery = handler()
    assert recovery.observe(shot('animated_poster'))[0] == 'unveil_poster'
    run.tap.assert_not_called()


def test_entire_introduction_drains_before_crop_work_resumes():
    run, recovery = handler()
    pages = ['unlocked', 'board', 'posters', 'unveil_poster', 'info',
             'open_details', 'close_details', 'complete', 'growing']
    queue = []
    for i, page in enumerate(pages[:-1]):
        queue.append(shot(page, f'checked-{i}'))
        queue.extend(shot(pages[i+1], f'after-{i}-{j}') for j in range(2))
        if i < len(pages)-2:
            queue.append(shot(pages[i+1], f'settled-{i}'))
    run._capture_raw = Mock(side_effect=queue)
    result = recovery.process(shot('unlocked'))
    assert result.png == shot('growing').png
    assert run.tap.call_count == 8
    assert run._workspace_needs_restore and not run._workspace_zoom_attempted
    for call in run.tap.call_args_list:
        assert call.args[1].captured_at.startswith('checked-')


def test_wrong_foreground_and_stale_frames_never_tap_the_introduction():
    run, recovery = handler()
    frame = shot('complete')
    run.client.foreground_package.return_value = 'com.google.android.gms'
    assert recovery.process(frame) is frame
    run.tap.assert_not_called()
    run.client.foreground_package.return_value = 'com.supercell.hayday'
    run._capture_raw = Mock(return_value=frame)
    with pytest.raises(WheatingPrerequisite, match='changed before'):
        recovery.process(frame)
    run.tap.assert_not_called()


def test_failed_tutorial_tap_is_not_repeated():
    run, recovery = handler()
    frame = shot('complete')
    run._capture_raw = Mock(side_effect=[replace(frame, captured_at=str(i)) for i in range(13)])
    with pytest.raises(WheatingPrerequisite, match='did not advance'):
        recovery.process(frame)
    with pytest.raises(WheatingPrerequisite, match='unconfirmed'):
        recovery.process(frame)
    assert run.tap.call_count == 1


def test_unknown_tutorial_text_never_uses_the_cream_panel_as_a_button():
    run, recovery = handler()
    run.vision.text = Mock(read=Mock(return_value=AdTextObservation(lines=(
        AdTextLine('Confirm diamond purchase', (900, 400, 600, 40)),))))
    frame = shot('complete')
    assert recovery.process(frame) is frame
    run.tap.assert_not_called()


def test_launch_accepts_known_event_tutorial_only_in_hay_day(monkeypatch):
    run, event_board = handler()
    run._recovery = SimpleNamespace(event_board=event_board)
    launch = WheatLaunchRecovery(run, capture=Mock())
    monkeypatch.setattr('hayday.launcher.LaunchRecovery._returned_game', lambda *args: False)
    assert launch._returned_game(shot('unlocked'))
    assert not launch._returned_game(shot('growing'))
    run.client.foreground_package.return_value = 'com.google.android.gms'
    assert not launch._returned_game(shot('unlocked'))
    run.tap.assert_not_called()


def test_warm_hud_search_rechecks_pixels_and_finds_moved_hud():
    vision = FarmSceneVision()
    frame = shot('growing')
    assert vision.ready(frame.png)
    image = _decode(frame.png)
    altered = image.copy()
    altered[:250, :330] = 0
    assert not vision.ready(cv2.imencode('.png', altered)[1].tobytes())
    assert vision.ready(frame.png)
    altered = image.copy()
    altered[:260, 1344:] = 0
    assert not vision.ready(cv2.imencode('.png', altered)[1].tobytes())
    assert vision.ready(frame.png)
    altered = image.copy()
    altered[8:238, :326] = image[:230, :326]
    altered[:8, :326] = 0
    assert vision.ready(cv2.imencode('.png', altered)[1].tobytes())


@pytest.mark.parametrize('during_launch', [False, True])
def test_transient_android_rotation_waits_only_during_a_known_launch(tmp_path, monkeypatch, during_launch):
    landscape = Screenshot(b'landscape', 1920, 1080, 'restored')
    portrait = Screenshot(b'portrait', 1080, 1920, 'rotating')
    client = Mock(serial='test', capture_fast=Mock(side_effect=[portrait, landscape]))
    run = WheatingRunner(client, tmp_path)
    run._size = 1920, 1080
    run._launch_rotation_until = 115 if during_launch else 0
    monkeypatch.setattr('hayday.wheating.time.monotonic', lambda: 100.)
    run.wait = Mock()
    if during_launch:
        assert run._capture_raw() is landscape
        assert client.capture_fast.call_count == 2
    else:
        with pytest.raises(WheatingBlocked, match='resolution changed'):
            run._capture_raw()
        assert client.capture_fast.call_count == 1
    assert run._size == (1920, 1080)
    client.tap.assert_not_called()
    client.swipe.assert_not_called()


def test_rotation_wait_times_out_and_honors_cancel(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hayday.wheating.time.monotonic', lambda: clock[0])
    client = Mock(serial='test', capture_fast=Mock(return_value=Screenshot(b'p', 1080, 1920, 'p')))
    run = WheatingRunner(client, tmp_path)
    run._size = 1920, 1080
    run._launch_rotation_until = 115.
    run.wait = lambda _: clock.__setitem__(0, clock[0]+2)
    with pytest.raises(WheatingBlocked, match='resolution changed'):
        run._capture_raw()
    assert client.capture_fast.call_count == 4
    clock[0] = 100.
    run.wait = Mock(side_effect=WheatingCancelled('stopped'))
    with pytest.raises(WheatingCancelled):
        run._capture_raw()
    client.tap.assert_not_called()


def test_recorded_growing_workspace_does_not_pan_or_zoom(tmp_path, monkeypatch):
    client = Mock(serial='127.0.0.1:16416')
    run = WheatingRunner(client, tmp_path, vision=WheatingVision())
    shutil.copytree(ROOT/'farming_evidence', run.device_root/'farming_evidence')
    shutil.copy2(ROOT/'fields.json', run.device_root/'fields.json')
    run.state = json.loads((ROOT/'state.json').read_text())
    fields = WheatFields(run)
    frame = shot('growing')
    monkeypatch.setattr('hayday.wheating_restart_camera._settled', lambda _: frame)
    view = fields._group.observe(frame)
    assert view is not None and view.count('growing') == len(view.cells) == 9
    assert restore_workspace(run, fields) is frame
    client.swipe.assert_not_called()
    client.pinch_zoom_out.assert_not_called()
    # Partial evidence cannot claim the saved nine-tile workspace is complete.
    fields._group.observe = Mock(return_value=replace(view, cells=view.cells[:5]))
    monkeypatch.setattr('hayday.wheating_restart_camera._workspace_drag', lambda *args: None)
    with pytest.raises(WheatingBlocked, match='no clear grass'):
        restore_workspace(run, fields)


def test_camera_finds_top_edge_wheat_and_rejects_yellow_trees():
    vision = WheatingVision()
    frame = shot('top_edge')
    assert vision.field_bounds(frame) is None
    bounds = vision.edge_field_bounds(frame)
    assert bounds is not None and bounds.y < frame.height*.20
    from hayday.wheating_restart_camera import _workspace_drag
    x, y, _, end_y = _workspace_drag(frame, bounds.center, None)
    assert x > 0 and end_y > y
    assert vision.edge_field_bounds(shot('trees')) is None
    assert vision.edge_field_bounds(shot('growing')) is None

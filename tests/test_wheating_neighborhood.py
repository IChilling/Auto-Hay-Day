"""Recorded level-ten popup, freshness, and recovery entry-point regressions."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from hayday.ad_text import AdTextLine, AdTextObservation
from hayday.adb import Screenshot
from hayday.resource_vision import _decode
from hayday.wheating import WheatingCancelled, WheatingPrerequisite, WheatingRunner
from hayday.wheating_neighborhood import WheatNeighborhood
from hayday.wheating_recovery import WheatLaunchRecovery, WheatRecovery
from hayday.wheating_vision import WheatingVision

ROOT = Path(__file__).parent/'fixtures/neighborhood'


def shot(name, at='initial'):
    return Screenshot((ROOT/f'{name}.png').read_bytes(), 1920, 1080, at)


def handler():
    catalog = json.loads((ROOT/'text.json').read_text())
    readings = {shot(name).png: AdTextObservation(lines=tuple(AdTextLine(**line) for line in lines))
                for name, lines in catalog.items()}
    text = Mock(read=lambda png: readings.get(png, AdTextObservation()))
    run = SimpleNamespace(vision=WheatingVision(text_reader=text), diagnostics=None,
        client=SimpleNamespace(serial='test', foreground_package=Mock(return_value='com.supercell.hayday')),
        cancel_event=threading.Event(), check=Mock(), wait=Mock(), publish=Mock(), tap=Mock(),
        state={'pending': {'operation': 'collect'}, 'field_layout': {'version': 2}},
        _workspace_needs_restore=False, _workspace_zoom_attempted=True)
    return run, WheatNeighborhood(run)


def test_recorded_neighborhood_message_and_clear_farm():
    run, popup = handler()
    target = popup.observe(shot('introduction'))
    assert target is not None and target.box == (684, 136, 1112, 660)
    run.vision.text.read = Mock(side_effect=AssertionError('Ordinary farm must not request OCR'))
    assert popup.observe(shot('farm')) is None
    run.tap.assert_not_called()


@pytest.mark.parametrize('scale', [.5, .75, 1.25])
def test_neighborhood_geometry_and_text_scale_together(scale):
    run, popup = handler()
    original = shot('introduction')
    width, height = round(original.width*scale), round(original.height*scale)
    image = cv2.resize(_decode(original.png), (width, height))
    scaled = replace(original, png=cv2.imencode('.png', image)[1].tobytes(), width=width, height=height)
    text = run.vision.text.read(original.png)
    run.vision.text.read = Mock(return_value=replace(text, lines=tuple(
        replace(line, bounds=tuple(round(v*scale) for v in line.bounds)) for line in text.lines)))
    assert popup.observe(scaled) is not None
    run.tap.assert_not_called()


@pytest.mark.parametrize('text,error', [
    ('Confirm diamond purchase', None),
    ('Repair your neighborhood house', None),
    ('My, my what do we have here? Repair your neighborhood house to chat and play with your closest friends!', 'OCR failed'),
])
def test_unknown_partial_or_failed_ocr_does_not_authorize_input(text, error):
    run, popup = handler()
    run.vision.text.read = Mock(return_value=AdTextObservation(
        lines=(AdTextLine(text, (900, 400, 600, 40)),), error=error))
    frame = shot('introduction')
    assert popup.process(frame) is frame
    run.tap.assert_not_called()


def test_popup_waits_for_two_clear_frames_and_preserves_pending_work():
    run, popup = handler()
    pending = json.loads(json.dumps(run.state))
    run._capture_raw = Mock(side_effect=[shot('introduction', 'confirmed'),
        shot('farm', 'transition'), shot('introduction', 'animated'),
        shot('farm', 'clear1'), shot('farm', 'clear2')])
    result = popup.process(shot('introduction'))
    assert result.captured_at == 'clear2'
    run.tap.assert_called_once()
    assert run.tap.call_args.args[1].captured_at == 'confirmed'
    assert run.state == pending and not popup.uncertain
    assert run._workspace_needs_restore and not run._workspace_zoom_attempted
    assert popup.process(result) is result
    assert run.tap.call_count == 1


@pytest.mark.parametrize('problem', ['stale', 'changed', 'foreground'])
def test_changed_stale_or_wrong_app_before_tap_stops_input(problem):
    run, popup = handler()
    frame = shot('introduction')
    run._capture_raw = Mock(return_value=frame if problem == 'stale' else shot('farm', 'new'))
    if problem == 'foreground':
        run.client.foreground_package.return_value = 'com.google.android.gms'
        assert popup.process(frame) is frame
    else:
        with pytest.raises(WheatingPrerequisite, match='changed before'):
            popup.process(frame)
    run.tap.assert_not_called()


def test_foreground_switch_during_confirmation_does_not_tap():
    run, popup = handler()
    run.client.foreground_package.side_effect = ['com.supercell.hayday']+['com.google.android.gms']*4
    run._capture_raw = Mock(side_effect=[shot('introduction', f'fresh{i}') for i in range(4)])
    with pytest.raises(WheatingPrerequisite, match='changed before'):
        popup.process(shot('introduction'))
    run.tap.assert_not_called()


@pytest.mark.parametrize('next_frame', ['introduction', 'unknown', 'stale_farm'])
def test_unconfirmed_tap_or_unknown_followup_is_not_repeated(next_frame):
    run, popup = handler()
    frame = shot('introduction')
    if next_frame == 'unknown':
        pixels = _decode(frame.png).copy()
        pixels[0, 0] = 0
        unknown = replace(frame, png=cv2.imencode('.png', pixels)[1].tobytes())
        # Model a subsequent unrecognized dialog that still has farm HUD.
        observe = popup.observe
        popup.observe = lambda value: None if value.png == unknown.png else observe(value)
        run.vision.farm = Mock(return_value=True)
        following = [replace(unknown, captured_at=f'after{i}') for i in range(20)]
    elif next_frame == 'stale_farm':
        following = [shot('farm', 'same')]*20
    else:
        following = [replace(frame, captured_at=f'after{i}') for i in range(20)]
    run._capture_raw = Mock(side_effect=[replace(frame, captured_at='confirmed')]+following)
    with pytest.raises(WheatingPrerequisite, match='did not return'):
        popup.process(frame)
    with pytest.raises(WheatingPrerequisite, match='unconfirmed'):
        popup.process(frame)
    assert run.tap.call_count == 1 and popup.uncertain


def test_cancellation_during_confirmation_stops_before_input():
    run, popup = handler()
    run.wait.side_effect = WheatingCancelled('Stopped')
    with pytest.raises(WheatingCancelled):
        popup.process(shot('introduction'))
    run.tap.assert_not_called()


def test_launch_and_reconnect_accept_neighborhood_as_known_recovery(tmp_path, monkeypatch):
    run, popup = handler()
    full_run = WheatingRunner(run.client, tmp_path, vision=run.vision)
    recovery = WheatRecovery(full_run)
    full_run._recovery = recovery
    assert recovery.ready(shot('introduction'))
    launch = WheatLaunchRecovery(full_run, capture=Mock())
    monkeypatch.setattr('hayday.launcher.LaunchRecovery._returned_game', lambda *args: False)
    assert launch._returned_game(shot('introduction'))
    run.client.foreground_package.return_value = 'com.google.android.gms'
    assert not launch._returned_game(shot('introduction'))
    run.tap.assert_not_called()


@pytest.mark.parametrize('entry', ['visible', 'level_up', 'launch', 'reconnect'])
def test_each_recovery_entry_drains_introduction_before_crop_resume(entry):
    run, popup = handler()
    run.state = {'pending': None}
    recovery = WheatRecovery.__new__(WheatRecovery)
    recovery.run = run
    recovery.neighborhood = popup
    def passthrough(frame):
        return frame
    for name in ('achievement', 'silo_full', 'level_up', 'event_board', 'launch', 'reconnect', 'tutorial'):
        setattr(recovery, name, SimpleNamespace(process=Mock(side_effect=passthrough)))
    initial = shot('introduction') if entry == 'visible' else shot('farm')
    if entry != 'visible':
        process = getattr(recovery, entry).process
        process.side_effect = lambda frame: shot('introduction') if frame is initial else frame
    recovery.possible = Mock(return_value=(entry == 'launch', entry == 'reconnect'))
    run._capture_raw = Mock(side_effect=[shot('introduction', 'confirmed'), shot('farm', 'clear1'), shot('farm', 'clear2')])
    result = recovery.process(initial)
    assert result.captured_at == 'clear2'
    assert run.tap.call_count == 1 and not popup.uncertain
    assert run._workspace_needs_restore

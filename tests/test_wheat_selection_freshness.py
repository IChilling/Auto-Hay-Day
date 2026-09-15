"""Expired pre-harvest calculations must not turn a live farm into a hard stop."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import AdbError, Screenshot
from hayday.farming_vision import HarvestTarget
from hayday.resource_vision import VisualTarget, _decode
from hayday.wheating import (
    WheatingCancelled,
    WheatingPrerequisite,
    WheatingRunner,
    WheatingStaleControl,
)
from hayday.wheating_selection import _tap_fresh_wheat, _wheat_at, select_wheat

ROOT = Path(__file__).parent/'fixtures/stale_selection'


@pytest.fixture
def selection(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hayday.wheating.time.monotonic', lambda: clock[0])
    before = Screenshot((ROOT/'ripe.png').read_bytes(), 1920, 1080, 'before')
    client = SimpleNamespace(serial='test', tap=Mock(), foreground_package=Mock(return_value='com.supercell.hayday'))
    run = WheatingRunner(client, tmp_path)
    run.vision = SimpleNamespace(farm=Mock(return_value=True), _image_for=lambda _: _decode(before.png))
    run.wait = Mock()
    run.publish = Mock()
    run.diagnostics = tmp_path
    run.frame, run._captured = before, 90.
    counter = [0]
    def capture(**kwargs):
        counter[0] += 1
        clock[0] += .1
        frame = replace(before, captured_at=f'fresh-{counter[0]}')
        run.frame, run._captured = frame, clock[0]
        return frame
    run.capture = Mock(side_effect=capture)
    target = VisualTarget(678, 716, 30, 30, 1.)
    worker = SimpleNamespace(serial='test', _translated_plot=Mock(return_value=target.center),
        harvest_plan=(before, [(693, 731), (750, 702)]))
    return run, worker, before, target, clock


def test_expired_route_gets_a_new_capture_and_keeps_its_original_proof(selection):
    run, worker, before, target, _ = selection
    plan = worker.harvest_plan
    with pytest.raises(WheatingStaleControl):
        run.tap(target.center, before)
    fresh, checked = _tap_fresh_wheat(run, worker, before, target)
    assert fresh is run.frame and fresh is not before and checked.center == target.center
    run.client.tap.assert_called_once_with(693, 731, width=1920, height=1080)
    assert worker.harvest_plan is plan


def test_slow_revalidation_retries_only_the_unsent_selection(selection):
    run, worker, before, target, clock = selection
    delays = iter([6., .2])
    def translate(*args):
        clock[0] += next(delays)
        return target.center
    worker._translated_plot.side_effect = translate
    fresh, _ = _tap_fresh_wheat(run, worker, before, target)
    assert fresh.captured_at == 'fresh-2'
    assert run.capture.call_count == 2
    run.client.tap.assert_called_once()
    diagnostic = json.loads((run.diagnostics/'stale_control.json').read_text())
    assert diagnostic['same_frame'] and diagnostic['age_seconds'] == 6.


def test_repeated_delays_stop_after_three_captures_without_input(selection):
    run, worker, before, target, clock = selection
    def translate(*args):
        clock[0] += 6.
        return target.center
    worker._translated_plot.side_effect = translate
    with pytest.raises(WheatingPrerequisite, match='three observations'):
        _tap_fresh_wheat(run, worker, before, target)
    assert run.capture.call_count == 3
    run.client.tap.assert_not_called()


def test_superseded_frame_is_reobserved_instead_of_tapped(selection):
    run, worker, before, target, _ = selection
    def translate(*args):
        if worker._translated_plot.call_count == 1:
            run.frame = before
        return target.center
    worker._translated_plot.side_effect = translate
    fresh, _ = _tap_fresh_wheat(run, worker, before, target)
    assert fresh.captured_at == 'fresh-2'
    run.client.tap.assert_called_once()
    assert not json.loads((run.diagnostics/'stale_control.json').read_text())['same_frame']


@pytest.mark.parametrize('problem', ['stale_timestamp', 'foreign_app', 'foreground_switch', 'no_farm', 'unmapped', 'no_wheat', 'edge', 'resolution'])
def test_unconfirmed_target_never_sends_input(selection, problem):
    run, worker, before, target, _ = selection
    if problem == 'stale_timestamp':
        run.capture.side_effect = None
        run.capture.return_value = before
    elif problem == 'foreign_app':
        run.client.foreground_package.return_value = 'com.google.android.gms'
    elif problem == 'foreground_switch':
        run.client.foreground_package.side_effect = ['com.supercell.hayday', 'other']*3
    elif problem == 'no_farm':
        run.vision.farm.return_value = False
    elif problem == 'unmapped':
        worker._translated_plot.return_value = None
    elif problem == 'no_wheat':
        worker._translated_plot.return_value = (500, 600)
    elif problem == 'edge':
        worker._translated_plot.return_value = (20, 40)
    else:
        run.capture.side_effect = [replace(before, width=1080, height=1920, captured_at=str(i)) for i in range(3)]
    with pytest.raises(WheatingPrerequisite, match='no selection tap'):
        _tap_fresh_wheat(run, worker, before, target)
    run.client.tap.assert_not_called()


def test_recorded_wheat_and_nearby_grass_have_different_evidence(selection):
    run, _, before, target, _ = selection
    assert _wheat_at(run, before, target.center)
    assert not _wheat_at(run, before, (500, 600))


@pytest.mark.parametrize('error', [AdbError('input outcome unknown'), WheatingCancelled('cancelled')])
def test_error_during_or_after_input_cannot_repeat_the_tap(selection, error):
    run, worker, before, target, _ = selection
    if isinstance(error, AdbError):
        run.client.tap.side_effect = error
    else:
        run.wait.side_effect = error
    with pytest.raises(type(error)):
        _tap_fresh_wheat(run, worker, before, target)
    assert run.capture.call_count == 1
    assert run.client.tap.call_count == 1


def test_cancellation_before_confirmation_prevents_capture_and_input(selection):
    run, worker, before, target, _ = selection
    run.cancel_event.set()
    with pytest.raises(WheatingCancelled):
        _tap_fresh_wheat(run, worker, before, target)
    run.capture.assert_not_called()
    run.client.tap.assert_not_called()


def test_fresh_selection_still_requires_two_tool_observations(selection):
    run, worker, before, target, _ = selection
    plan = worker.harvest_plan
    harvest = HarvestTarget(VisualTarget(600, 600, 40, 40, 1.), target, 1., 1., highlight=target)
    worker.vision = SimpleNamespace(harvest=Mock(return_value=harvest))
    worker._planned_harvest = Mock(return_value=[target.center, (750, 702)])
    selected, _, _, ready = select_wheat(run, worker, before, target)
    assert ready and selected.captured_at == 'fresh-3'
    assert run.capture.call_count == 3
    run.client.tap.assert_called_once()
    assert worker._prepared_harvest.frame is selected
    assert worker._prepared_harvest.plan is plan

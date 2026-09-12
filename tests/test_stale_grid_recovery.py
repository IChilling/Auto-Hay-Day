"""Replay the bare farm and polluted saved grid that caused the restart loop."""
import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hayday.adb import Screenshot
from hayday.wheating import WheatingBlocked, WheatingCancelled, WheatingRunner
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_crop_resume import bare_grid, rediscover_bare_field, saved_soil_grid
from hayday.wheating_fields import WheatFields
from hayday.wheating_restart import WheatRestart
from hayday.wheating_restart_camera import restore_workspace
from hayday.wheating_vision import WheatingVision

FIXTURES = Path(__file__).parent/'fixtures/restart_bare'


@pytest.fixture
def recorded(tmp_path):
    frame = Screenshot((FIXTURES/'current.png').read_bytes(), 1920, 1080, 'first')
    worker = WheatCropWorker(SimpleNamespace(serial='test', drag_path=Mock()), Mock(),
        threading.Event(), Mock(), tmp_path/'fields.json', vision=WheatFarmingVision())
    worker._bind_size(frame)
    evidence = tmp_path/'farming_evidence'
    evidence.mkdir(exist_ok=True)
    for path in FIXTURES.glob('*.png'):
        shutil.copyfile(path, evidence/path.name)
    entry = json.loads((FIXTURES/'entry.json').read_text())
    worker.state['items']['field'] = entry
    worker.soil_frame = worker._saved_frame(entry['harvest_after'][-1])
    return worker, frame, entry


def test_recorded_stale_grid_fails_but_visible_bare_farm_restores_without_camera_input(recorded):
    worker, frame, entry = recorded
    grid = saved_soil_grid(worker, entry)
    assert grid is not None
    assert bare_grid(worker, grid[0], frame, grid[1], grid[2]) is None
    run = SimpleNamespace(client=Mock(), diagnostics=None, vision=WheatingVision(), check=Mock())
    fields = SimpleNamespace(worker=worker, _known_before=None, _known_points=[])
    with patch('hayday.wheating_restart_camera._settled', return_value=frame):
        assert restore_workspace(run, fields) is frame
    assert not run.client.mock_calls


def test_fresh_soil_and_picker_replace_stale_grid_before_planting(recorded):
    worker, frame, entry = recorded
    fresh = replace(frame, captured_at='second')
    picker = worker._saved_frame(entry['replant_evidence'])
    selected = worker.vision.empty_plot(picker.png)
    assert selected is not None
    worker._frame = Mock(return_value=fresh)
    worker._tap = Mock(return_value=picker)
    worker._await_seed_picker = Mock(return_value=(picker, selected))
    worker._plant_selected = Mock(return_value='planted')
    assert rediscover_bare_field(worker, frame, b'wheat', 'field', entry) == 'planted'
    worker._tap.assert_called_once()
    assert worker._tap.call_args.args[1] is fresh
    current = worker.state['items']['field']
    assert current['replant_grid'] is None
    assert worker._saved_frame(current['harvest_after'][-1]).png == fresh.png
    assert list(worker.state_path.parent.glob('farming_evidence/*rediscovered*.json'))
    worker._plant_selected.assert_called_once()


@pytest.mark.parametrize('change', [{'stage': 'plant_attempted'}, {'replanted': True},
    {'plant_before': {'file': 'attempt.png'}}, {'points': [[600,450]]}])
def test_rediscovery_never_replays_an_attempted_plant(recorded, change):
    worker, frame, entry = recorded
    worker._tap = Mock()
    assert rediscover_bare_field(worker, frame, b'wheat', 'field', {**entry, **change}) is None
    worker._tap.assert_not_called()


def test_unconfirmed_picker_keeps_previous_checkpoint(recorded):
    worker, frame, entry = recorded
    original = json.loads(json.dumps(entry))
    worker._frame = Mock(return_value=replace(frame, captured_at='second'))
    worker._tap = Mock(return_value=frame)
    worker._await_seed_picker = Mock(return_value=(frame, None))
    worker._plant_selected = Mock()
    assert rediscover_bare_field(worker, frame, b'wheat', 'field', entry).status == 'changed'
    assert worker.state['items']['field'] == original
    worker._plant_selected.assert_not_called()


def test_repeated_visible_soil_recognition_errors_do_not_force_stop(recorded, tmp_path):
    worker, frame, entry = recorded
    client = SimpleNamespace(serial='test', capture=Mock(return_value=frame), force_stop_hay_day=Mock())
    run = WheatingRunner(client, tmp_path/'run')
    run._lock_file = Mock()
    run.vision = WheatingVision()
    run.wait = Mock()
    restart = WheatRestart(run)
    fields = WheatFields.__new__(WheatFields)
    fields.run, fields.worker = run, worker
    worker._close = Mock()
    for _ in range(3):
        restart.recover(WheatingBlocked('Saved grid cannot match'), fields)
    client.force_stop_hay_day.assert_not_called()
    assert restart.stage == 'live_field_retry'
    assert worker.state['items']['field'] is entry
    run.cancel_event.set()
    with pytest.raises(WheatingCancelled):
        restart.recover(WheatingBlocked('Saved grid cannot match'), fields)
    assert not restart.active

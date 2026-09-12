"""Real timer colors and one-time, session-scoped growth calibration."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingRunner
from hayday.wheating_crop import WheatFarmingVision
from hayday.wheating_fields import WheatFields
from hayday.wheating_growth import WheatGrowthTimer, parse_countdown, read_countdown

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path,expected', [
    ('tests/fixtures/wheat_growth_purple.png', 58),
    ('images/reference_captures/farming_corn_planted.png', 299),
])
@pytest.mark.parametrize('scale', [1., .75])
def test_real_white_and_purple_timers(path, expected, scale):
    image = cv2.imread(str(ROOT/path))
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    frame = Screenshot(cv2.imencode('.png', image)[1].tobytes(), image.shape[1], image.shape[0], 'fresh')
    reading = read_countdown(frame, WheatFarmingVision())
    assert reading and reading[0] == expected


@pytest.mark.parametrize('text,expected', [('58SEC', 58), ('1MIN', 60), ('2MIN00SEC', 120),
    ('1MIN30SEC', 90), ('0SEC', None), ('58', None), ('1MIN75SEC', None),
    ('99MIN', None), ('', None), ('WHEAT58SEC', None)])
def test_countdown_requires_complete_valid_units(text, expected):
    assert parse_countdown(text) == expected


def test_measure_accounts_for_elapsed_time_and_skips_all_future_reads(tmp_path, monkeypatch):
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    timer = WheatGrowthTimer(run)
    bar = VisualTarget(300, 700, 616, 107, 1.)
    frame = Screenshot(b'first', 1920, 1080, 'first')
    following = replace(frame, captured_at='second')
    worker = SimpleNamespace(vision=SimpleNamespace(growing=Mock(return_value=bar)),
        _check=Mock(), _pause=Mock(), _quick_frame=Mock(return_value=following),
        _same_target=lambda *a: True, growth_ready_at=220.)
    clock = iter([101.2, 102.1])
    monkeypatch.setattr('hayday.wheating_growth.time.monotonic', lambda: next(clock))
    reader = Mock(side_effect=[(58, '58SEC', bar), (57, '57SEC', bar)])
    monkeypatch.setattr('hayday.wheating_growth.read_countdown', reader)
    assert timer.measure(worker, frame, frame, (900, 600), 100.) is following
    assert timer.duration == 60 and worker.growth_ready_at == 160
    assert run.state['wheat_growth']['seconds'] == 60
    worker.vision.growing.reset_mock()
    assert timer.measure(worker, frame, frame, (900, 600), 200.) is frame
    worker.vision.growing.assert_not_called()
    assert reader.call_count == 2


def test_unclear_timer_does_not_cache_a_guess(tmp_path, monkeypatch):
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    timer = WheatGrowthTimer(run)
    frame = Screenshot(b'first', 1920, 1080, 'first')
    worker = SimpleNamespace(vision=SimpleNamespace(growing=lambda png: True),
        _check=Mock(), _pause=Mock(), _quick_frame=Mock(return_value=frame))
    monkeypatch.setattr('hayday.wheating_growth.read_countdown', lambda *a: None)
    timer.measure(worker, frame, frame, (900, 600), 100.)
    assert timer.seconds is None and 'wheat_growth' not in run.state


def test_recovery_reuses_session_cache_but_new_start_rechecks_boosts(tmp_path):
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    first = WheatFields(run)
    first.worker.growth_timer.seconds = 60.
    restored = WheatFields(run)
    assert restored.worker.growth_timer is first.worker.growth_timer
    assert restored.worker.growth_duration == 60.
    new_run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    new_run.state['wheat_growth'] = {'seconds': 60.}
    assert WheatFields(new_run).worker.growth_timer.seconds is None

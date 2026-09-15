"""Recorded Play Games controls and startup ordering across independent farms."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.ad_text import AdTextLine, AdTextObservation
from hayday.adb import Screenshot
from hayday.wheating import WheatingPrerequisite
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_fields import WheatFields
from hayday.wheating_play_games import WheatPlayGames, profile_cancel_target
from hayday.wheating_vision import WheatingVision

ROOT = Path(__file__).parent/'fixtures/startup'
LABELS = tuple(AdTextLine(text, box) for text, box in (
    ('Cancel', (521, 472, 68, 17)),
    ('Google Play Games', (878, 477, 214, 26)),
    ('Next', (1349, 473, 45, 16)),
    ('Create a Play Games profile', (568, 808, 302, 23)),
))


def shot(name, at='first'):
    return Screenshot((ROOT/name).read_bytes(), 1920, 1080, at)


def handler():
    text = Mock(read=Mock(return_value=AdTextObservation(lines=LABELS)))
    run = SimpleNamespace(vision=WheatingVision(text_reader=text), diagnostics=None,
        client=SimpleNamespace(foreground_package=Mock(return_value='com.google.android.gms')),
        wait=Mock(), publish=Mock(), tap=Mock())
    return run, WheatPlayGames(run)


def test_recorded_profile_has_one_cancel_and_never_selects_next():
    run, recovery = handler()
    frame = shot('play_games.png')
    target = recovery.observe(frame)
    assert target is not None and target.center == (555, 480)
    assert recovery.possible(shot('play_games_launcher.png'))
    assert profile_cancel_target(frame, LABELS[:-1]) is None
    assert profile_cancel_target(frame, (*LABELS, LABELS[0])) is None
    wrong = (replace(LABELS[0], bounds=(1400, 472, 68, 17)), *LABELS[1:])
    assert profile_cancel_target(frame, wrong) is None
    assert not recovery.possible(shot('zoomed-16384.png'))
    assert not recovery.possible(shot('zoomed-16416.png'))
    assert not run.tap.called


def test_profile_dismissal_requires_two_fresh_controls_and_returns_to_hay_day():
    run, recovery = handler()
    frame, fresh = shot('play_games.png'), shot('play_games.png', 'second')
    clear = [shot('zoomed-16416.png', at) for at in ('third', 'fourth')]
    run._capture_raw = Mock(side_effect=[fresh, *clear])
    run.client.foreground_package.side_effect = ['com.google.android.gms']*2+['com.supercell.hayday']*2
    assert recovery.process(frame) is clear[-1]
    run.tap.assert_called_once_with((555, 480), fresh)
    assert all(call.kwargs == {'handle_notifications': False} for call in run._capture_raw.call_args_list)
    assert not recovery.uncertain


def test_profile_artwork_in_the_wrong_foreground_app_does_not_get_input():
    run, recovery = handler()
    run.client.foreground_package.return_value = 'com.supercell.hayday'
    frame = shot('play_games.png')
    assert recovery.process(frame) is frame
    run.vision.text.read.assert_not_called()
    run.tap.assert_not_called()


def test_profile_left_over_a_stopped_game_can_return_to_the_verified_launcher():
    run, recovery = handler()
    frame = shot('play_games_launcher.png')
    fresh = replace(frame, captured_at='second')
    clear = [shot('home-16448.png', at) for at in ('third', 'fourth')]
    run._capture_raw = Mock(side_effect=[fresh, *clear])
    run.client.launcher_package = Mock(return_value='app.lawnchair')
    run.client.foreground_package.side_effect = ['com.google.android.gms']*2+['app.lawnchair']*2
    assert recovery.process(frame) is clear[-1]
    run.tap.assert_called_once_with((555, 480), fresh)


def test_stale_profile_capture_prevents_cancel():
    run, recovery = handler()
    frame = shot('play_games.png')
    run._capture_raw = Mock(return_value=frame)
    with pytest.raises(WheatingPrerequisite, match='changed before'):
        recovery.process(frame)
    run.tap.assert_not_called()


def test_uncleared_profile_is_not_repeatedly_cancelled():
    run, recovery = handler()
    frame = shot('play_games.png')
    run._capture_raw = Mock(side_effect=[replace(frame, captured_at=str(i)) for i in range(11)])
    with pytest.raises(WheatingPrerequisite, match='did not return to Hay Day'):
        recovery.process(frame)
    assert run.tap.call_count == 1
    with pytest.raises(WheatingPrerequisite, match='no repeated Cancel'):
        recovery.process(frame)
    assert run.tap.call_count == 1


def test_startup_prepares_camera_once_and_keeps_open_shop_for_reconciliation(monkeypatch):
    fields = WheatFields.__new__(WheatFields)
    frame = shot('zoomed-16416.png')
    restored = shot('before-16416.png')
    run = SimpleNamespace(_workspace_needs_restore=True, publish=Mock(), frame=frame,
                          vision=SimpleNamespace(shop=Mock(return_value=SimpleNamespace(kind='overview'))))
    fields.run = run
    prepare = Mock(return_value=restored)
    monkeypatch.setattr('hayday.wheating_restart_camera.restore_workspace', prepare)
    assert fields.prepare_workspace(frame) is frame
    assert run._workspace_needs_restore
    prepare.assert_not_called()
    run.vision.shop.return_value.kind = 'unknown'
    assert fields.prepare_workspace(frame) is restored
    assert not run._workspace_needs_restore
    assert fields.prepare_workspace(restored) is restored
    prepare.assert_called_once_with(run, fields)


def test_startup_zoom_reuses_all_fifteen_plots_only_with_fresh_crop_evidence(tmp_path, monkeypatch):
    fields = WheatFields.__new__(WheatFields)
    fields.worker = WheatCropWorker(SimpleNamespace(serial='test'), Mock(), threading.Event(), Mock(),
                                   tmp_path/'fields.json', vision=WheatFarmingVision())
    fields.run = SimpleNamespace(vision=WheatingVision())
    fields._known_before = shot('zoom-layout-16448.png')
    fields._known_points = json.loads((ROOT/'zoom-layout-16448.json').read_text())
    current = shot('zoomed-ready-16448.png')
    # This real startup frame changed zoom by roughly 4%, so translation fails.
    assert fields.worker._translated_plot(fields._known_before, current,
                                         fields._known_points[0], require_visible=False) is None
    route, observed = fields._known_wheat_points(current, minimum=.8)
    assert len(route) == len(observed) == 15
    # The same historical layout cannot authorize a changed/partial field.
    assert fields._known_wheat_points(shot('partial-16448.png'), minimum=.8) is None
    assert fields._known_wheat_points(shot('zoomed-16384.png'), minimum=.8) is None
    monkeypatch.setattr(fields.run.vision, 'wheat_outside', lambda *args: True)
    assert fields._known_wheat_points(current, minimum=.8) is None

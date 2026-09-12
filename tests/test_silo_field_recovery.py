"""Regression coverage for healthy full shops and disconnected bare fields."""
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingRunner
from hayday.wheating_fields import WheatFields
from hayday.wheating_restart import WheatRestart
from hayday.wheating_shop import WheatShop
from hayday.wheating_vision import SaleSlot, ShopView


def test_missed_composer_tap_returns_to_fresh_overview(tmp_path):
    frame = Screenshot(b'frame', 1920, 1080, 'fresh')
    slot = SaleSlot('empty', VisualTarget(300, 300, 200, 240, 1))
    overview = ShopView('overview', slots=(slot,))
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    run.wait = Mock()
    shop = WheatShop(run)
    shop.current = lambda *args: (frame, overview)
    shop.observe = Mock(return_value=(frame, overview))
    shop._tap = Mock()
    assert shop._open_slot(frame, overview, slot, 'composer') is None
    assert shop._ready_view == (frame, overview)
    assert run.state['pending'] is None
    shop._tap.assert_called_once()


def test_observed_full_shop_does_not_trip_stall_watchdog(tmp_path):
    clock = [0.]
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    run.state['silo_recovery'] = {'released': 20}
    run.state['ad_after'] = time.time()+600
    watchdog = WheatRestart(run, clock=lambda: clock[0])
    run._failsafe = watchdog
    watchdog.monitoring = True
    frame = Screenshot(b'shop', 1920, 1080, 'fresh')
    overview = ShopView('overview', slots=(SaleSlot('wheat', VisualTarget(300, 300, 200, 240, 1)),))
    shop = WheatShop(run)
    shop.observe = Mock(return_value=(frame, overview))
    shop.current = lambda *args: (frame, overview)
    for _ in range(12):
        clock[0] += 60
        assert shop.service()
        watchdog.check()
    assert watchdog.history == []
    assert run.state['silo_recovery']['released'] == 20


def test_partial_ripe_field_cannot_erase_silo_checkpoint():
    fields = WheatFields.__new__(WheatFields)
    fields.worker = SimpleNamespace(state={'items': {}}, _PENDING={'harvest_attempted'})
    fields.run = SimpleNamespace(state={'silo_recovery': {'released': 20}})
    fields._mature_field_confirmed = Mock(return_value=True)
    assert not fields.reconcile_stale_state(None)
    assert fields.run.state['silo_recovery']['released'] == 20
    fields._mature_field_confirmed.assert_not_called()


@pytest.mark.parametrize('remaining_kind', ['empty', 'ripe'])
def test_planting_first_section_repairs_soil_without_reharvesting(tmp_path, remaining_kind):
    frame = Screenshot(b'farm', 1920, 1080, 'fresh')
    targets = [VisualTarget(600, 450, 20, 20, 1), VisualTarget(900, 450, 20, 20, 1)]
    remaining = targets.copy()
    run = SimpleNamespace(client=SimpleNamespace(serial='test'), capture=lambda **kw: frame,
        cancel_event=threading.Event(), publish=Mock(), device_root=tmp_path,
        state={}, planted=0, check=Mock(), save_json=Mock(), persist=Mock(), diagnostics=None)
    run.vision = SimpleNamespace(shop_building=lambda f: None, wheat_icon=b'wheat',
        harvest_sweep=lambda f: [t.center for t in remaining],
        plots=lambda f, kind: tuple(remaining) if kind == remaining_kind else ())
    fields = WheatFields(run)
    fields._center_field = lambda f: f
    fields._relocate = lambda f: None
    fields._clear = lambda: frame
    fields._select = lambda f, t, k: (f, t.center, t, True)
    fields._remember_layout = Mock()
    fields._known_wheat_points = lambda *a, **kw: None
    fields.next_harvest = math.inf
    worker = fields.worker
    worker._translated_plot = lambda before, after, origin, **kw: origin
    selected_points = []
    def plant(f, icon, key):
        target = remaining.pop(0)
        selected_points.append(target.center)
        worker.planted_points = [target.center]
        worker.plant_before = frame
        worker.remaining_seed_stock = 20
        worker.growth_ready_at = time.monotonic()+120
        worker.field_size = 1
        worker.state['items'][key] = {'stage': 'growing', 'replanted': True}
        return SimpleNamespace(status='waiting')
    worker.work_if_recognized = plant
    count, _ = fields.work_view(frame)
    expected = 2 if remaining_kind == 'empty' else 1
    assert count == run.planted == expected
    assert selected_points == [t.center for t in targets[:expected]]
    assert remaining == targets[expected:]


def test_delayed_login_overlay_does_not_abort_camera_restoration():
    from hayday.wheating_restart_camera import _settled
    frames = [Screenshot(b'overlay', 1920, 1080, '1'),
              Screenshot(b'farm', 1920, 1080, '2'),
              Screenshot(b'farm', 1920, 1080, '3')]
    run = SimpleNamespace(wait=Mock(), capture=Mock(side_effect=frames),
                          vision=SimpleNamespace(farm=lambda f: f.png == b'farm'))
    with patch('hayday.wheating_restart_camera.CameraNavigator._modal_visible', return_value=False), \
         patch('hayday.wheating_restart_camera.CameraNavigator._view', return_value='same'), \
         patch('hayday.wheating_restart_camera.CameraNavigator._same_view', return_value=True):
        assert _settled(run) is frames[-1]


def test_partial_repair_does_not_inflate_the_saved_layout(tmp_path):
    frame = Screenshot(b'farm', 1920, 1080, 'fresh')
    fields = WheatFields.__new__(WheatFields)
    fields._known_before = frame
    fields._known_points = [(600, 450), (650, 475)]
    fields.run = SimpleNamespace(state={'field_layout': {'points': [[600, 450], [650, 475]]}})
    fields.worker = SimpleNamespace(plant_before=frame, planted_points=[(653, 478)],
        _translated_plot=lambda *a, **kw: (600, 450),
        _saved_frame=lambda proof: frame, _evidence=Mock(return_value={'file': 'verified.png'}))
    fields._remember_layout({'replanted': True})
    assert fields.run.state['field_layout']['points'] == [[653, 478]]
    assert len(fields._known_points) == 1


def test_recorded_camera_shift_uses_farm_landmarks_despite_stationary_hud():
    from hayday.wheating_restart_camera import farm_transform
    root = Path(__file__).parent/'fixtures'
    before = Screenshot((root/'workspace_before.png').read_bytes(), 1920, 1080, 'before')
    after = Screenshot((root/'workspace_shifted.png').read_bytes(), 1920, 1080, 'after')
    matrix = farm_transform(before, after)
    assert matrix is not None
    assert .9 < matrix[0, 0] < 1.1
    assert abs(matrix[1, 2]) > 100  # HUD identity must not be accepted as farm position.


def test_interrupted_harvest_recognizes_recorded_soil_after_zoom_change():
    from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
    root = Path(__file__).parent/'fixtures'
    before = Screenshot((root/'harvest_before_zoom.png').read_bytes(), 1920, 1080, 'before')
    after = Screenshot((root/'harvest_after_zoom.png').read_bytes(), 1920, 1080, 'after')
    harvest = WheatFarmingVision().harvest(before.png)
    assert harvest is not None
    assert WheatCropWorker._translated_plot(before, after, harvest.highlight.center) is None
    assert WheatCropWorker._exposed_soil(before, after, harvest) is not None


def test_stale_work_retirement_preserves_a_verified_complete_grid(tmp_path):
    frame = Screenshot(b'farm', 1920, 1080, 'fresh')
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = frame, [(700, 450), (750, 475), (800, 500)]
    layout = {'points': fields._known_points}
    fields.run = SimpleNamespace(state={'field_layout': layout, 'seed_reserve': 3},
        device_root=tmp_path, save_json=Mock(), persist=Mock(), publish=Mock())
    fields.worker = SimpleNamespace(state={'items': {'old': {'stage': 'harvest_attempted'}}},
        _PENDING={'harvest_attempted'}, state_path=tmp_path/'fields.json',
        _translated_plot=Mock(return_value=(700, 450)))
    fields._mature_field_confirmed = Mock(return_value=True)
    assert fields.reconcile_stale_state(frame)
    assert fields.run.state['field_layout'] is layout
    assert len(fields._known_points) == 3

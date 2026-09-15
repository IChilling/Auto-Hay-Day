"""Coverage, diagonal geometry and recovery from actual missed tiles."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hayday.adb import AdbClient, AdbError, Screenshot, parse_touch_device
from hayday.resource_vision import VisualTarget
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_fields import WheatFields
from hayday.wheating_routes import diagonal_mask_sweep, diagonal_order, diagonal_trace

ROOT = Path(__file__).parent/'fixtures'


def shot(name):
    return Screenshot((ROOT/name).read_bytes(), 1920, 1080, name)


@pytest.mark.parametrize('side,pitch', [(3, 54), (11, 27), (22, 20)])
def test_diagonal_route_covers_irregular_and_large_layouts(side, pitch):
    points = [(960+(i-j)*pitch, 230+(i+j)*pitch//2)
              for i in range(side) for j in range(side)
              if i == 0 or j == 0 or (i+j) % 7 != 0]
    ordered = diagonal_order(points)
    assert ordered[0] == points[0]
    assert len(ordered) == len(set(ordered)) == len(points)
    assert set(ordered) == set(points)
    path = diagonal_trace(ordered, bounds=(200, 180, 1720, 960))
    assert set(points) <= set(path)
    assert len(path) <= 2*len(points)
    for a, b in zip(path, path[1:], strict=False):
        dx, dy = abs(b[0]-a[0]), abs(b[1]-a[1])
        assert abs(dy-.5*dx) <= 2
        assert dx and dy  # No side-to-side row or vertical shortcut.


def test_diagonal_lanes_cover_every_pixel_including_islands_and_thin_necks():
    mask = np.zeros((700, 1200), np.uint8)
    cv2.fillPoly(mask, [np.array([(300,200),(700,250),(880,460),(590,490),(300,310)])], 1)
    mask[340, 875:925] = 1
    mask[110:120, 230:239] = 1
    mask[280:330, 510:560] = 0
    path = diagonal_mask_sweep(mask, step=16, margin=10)
    cover = np.zeros_like(mask)
    for a, b in zip(path[::2], path[1::2], strict=True):
        assert abs(abs(b[1]-a[1])-.5*abs(b[0]-a[0])) <= 1
        cv2.line(cover, a, b, 1, 18)
    assert not np.any(mask & (cover == 0))


@pytest.mark.parametrize('end', [(950, 20), (50, 950)])
def test_diagonal_turns_stay_inside_wide_or_tall_narrow_viewports(end):
    bounds = (0, 0, 1000, 40) if end[0] == 950 else (0, 0, 100, 1000)
    path = diagonal_trace([(50, 20), end], slope=.5, bounds=bounds)
    assert path[0] == (50, 20) and path[-1] == end
    assert all(bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3] for x, y in path)
    for a, b in zip(path, path[1:], strict=False):
        dx, dy = abs(b[0]-a[0]), abs(b[1]-a[1])
        assert dx and dy and abs(dy-.5*dx) <= 2


def test_live_sprouts_obscure_border_but_not_selected_soil_identity():
    vision = WheatFarmingVision()
    frame = shot('reliability/obscured_picker.png')
    plot = vision.empty_plot(frame.png)
    assert plot is not None and vision.full_outline
    assert (plot.width, plot.height) == (117, 65)
    assert np.linalg.norm(np.subtract(plot.center, (852,632))) < 3
    assert vision.empty_tiles(frame, plot, 512)


def test_a_missing_cell_cannot_cut_off_a_visible_bare_section(monkeypatch):
    vision = WheatFarmingVision()
    frame = shot('mumu_soil_picker.png')
    plot = vision.empty_plot(frame.png)
    expected = vision.empty_tiles(frame, plot, 512)
    # Make the selected cell temporarily unreadable. Independent inspection
    # must still find every other visible tile, regardless of connectivity.
    original = vision.tile_pixels
    monkeypatch.setattr(vision, 'tile_pixels', lambda image, p, dx, dy:
        None if np.linalg.norm(np.subtract(p, plot.center)) < 5 else original(image, p, dx, dy))
    actual = vision.empty_tiles(frame, plot, 512)
    assert set(actual) == set(expected)-{plot.center}


@pytest.mark.parametrize('scale', [.5, .67, .83, 1., 1.25])
def test_live_complete_field_keeps_all_51_tiles_at_different_resolutions(scale):
    grid = json.loads((ROOT/'reliability/complete_grid.json').read_text())
    image = cv2.imread(str(ROOT/'reliability/complete_bare.png'))
    resized = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    frame = Screenshot(cv2.imencode('.png', resized)[1].tobytes(), resized.shape[1], resized.shape[0], 'scaled')
    plot = VisualTarget(*(round(v*scale) for v in grid['plot']), 1.)
    vision = WheatFarmingVision()
    vision.full_outline = True  # The supplied selected outline anchors this clear frame.
    points = vision.empty_tiles(frame, plot, 512)
    assert len(points) == len(set(points)) == 51
    assert all(min(np.linalg.norm(np.subtract(p, np.multiply(want, scale))) for p in points) < 12*scale
               for want in grid['points'])


def test_seed_reserve_failure_preserves_seeds_without_stopping(tmp_path):
    fields = WheatFields.__new__(WheatFields)
    fields.run = SimpleNamespace(state={'seed_reserve': 3}, publish=Mock())
    fields.worker = SimpleNamespace(field_size=51)
    fields._known_points = [(600,500)]*46
    fields._group = SimpleNamespace(points=[])
    assert fields._conservative_seed_reserve() == 51
    fields._known_points = []
    fields.worker.field_size = 0
    assert fields._conservative_seed_reserve() == 512


def test_large_diagonal_drag_encodes_every_waypoint_and_releases(tmp_path):
    binary = tmp_path/'adb.exe'
    binary.touch()
    client = AdbClient(binary, 'test')
    client._touch_configuration = Mock(return_value=(parse_touch_device((ROOT/'mumu_touch.txt').read_text()), 1))
    calls = []

    def run(args, **kw):
        calls.append((args, kw))
        return (b'\x7fELF\x02\x01' if args[:2] == ['exec-out','dd'] else b''), b''
    client._run = run
    points = diagonal_trace(diagonal_order([(960+(i-j)*24,260+(i+j)*12)
                                            for i in range(16) for j in range(16)]))
    assert len(points) > 100
    client.drag_path([(800,200), *points], width=1920, height=1080,
                     duration_ms=4000, min_waypoint_ms=85, max_step_px=20)
    script = next(kw['_stdin'] for _, kw in calls if '_stdin' in kw)
    assert script.count(b"printf '%b'") >= len(points)
    assert calls[-1][1]['_release'] is True
    with pytest.raises(AdbError, match='2049'):
        client.drag_path([(500,500)]*2050, width=1920, height=1080)


@pytest.mark.parametrize('flickering', [False, True])
def test_partial_planting_reconciles_bare_and_unknown_without_replaying(tmp_path, flickering):
    before = shot('multi/interrupted_before.png')
    after = shot('multi/interrupted_bare.png')
    worker = WheatCropWorker(SimpleNamespace(serial='test', drag_path=Mock()), Mock(),
                             threading.Event(), Mock(), tmp_path/'fields.json', vision=WheatFarmingVision())
    worker._size = (1920,1080)
    points = json.loads((ROOT/'multi/interrupted_points.json').read_text())
    worker._saved_frame = Mock(return_value=before)
    worker._count = Mock(return_value=100)
    worker._seed = Mock(return_value=object())
    states = ('growing',)*(len(points)-2)+('bare','unknown')
    worker._observed_planting = Mock(side_effect=[states,
        states[:-1]+('growing',) if flickering else states])
    worker._frame = Mock(return_value=replace(after,captured_at='second'))
    worker._pause = Mock()
    entry = {'stage':'plant_attempted', 'plant_before':{}, 'points':points,
             'available_before':100, 'operation':'partial'}
    worker.state['items']['key'] = entry
    result = worker._resume_planted(after, b'wheat', 'key', entry)
    assert result.status == 'waiting'
    assert len(worker.planted_points) == len(points)-2
    assert worker.field_points == [tuple(p) for p in points]
    updated = worker.state['items']['key']
    assert updated['recovered_bare'] == updated['unverified_count'] == 1
    assert updated['deferred_points'] == points[-2:]
    worker.client.drag_path.assert_not_called()


@pytest.mark.parametrize('miss_selected', [False, True])
def test_newly_visible_soil_expands_coverage_without_exceeding_confirmed_stock(tmp_path, miss_selected):
    frame = shot('mumu_soil_picker.png')
    vision = WheatFarmingVision()
    plot = vision.empty_plot(frame.png)
    points = vision.empty_tiles(frame, plot, 512)[:3]
    assert len(points) == 3 and points[0] == plot.center
    client = SimpleNamespace(serial='test', drag_path=Mock(side_effect=RuntimeError('captured gesture')))
    worker = WheatCropWorker(client, Mock(), threading.Event(), Mock(), tmp_path/'fields.json', vision=vision)
    worker.soil_frame = frame
    worker._quick_frame = Mock(return_value=replace(frame, captured_at='second'))
    worker._seed = Mock(return_value=plot)
    worker._count = Mock(return_value=2)
    worker._fresh = Mock()
    worker.vision.empty_tiles = Mock(side_effect=[points[1:] if miss_selected else points[:2], points])
    with pytest.raises(RuntimeError, match='captured gesture'):
        worker._plant_selected(frame, b'wheat', 'key', plot)
    entry = worker.state['items']['key']
    assert entry['available_before'] == len(entry['points']) == 2
    assert entry['points'][0] == list(plot.center)
    assert worker.field_size == 3 and worker.field_points == points


def test_repair_keeps_verified_crops_when_the_sickle_menu_covers_them(monkeypatch):
    from hayday import wheating_harvest_repair as repair
    frame = shot('multi/residual_ripe.png')
    captures = [replace(frame, captured_at=str(i)) for i in range(5)]
    crops = {0: (500, 400), 1: (554, 427), 2: (608, 454)}
    seen = []

    def observe(worker, current, grid):
        seen.append(current.captured_at)
        # No crop pixels are available behind the open fan.
        return {} if current in captures[2:] else crops

    monkeypatch.setattr(repair, 'ripe_tiles', observe)
    monkeypatch.setattr(repair, 'same_harvest', lambda *args: True)
    target = VisualTarget(485, 385, 30, 30, 1.)
    harvest = SimpleNamespace(highlight=target, target=target, tool=target)
    worker = SimpleNamespace(_pause=Mock(), _check=Mock(), _fresh=Mock(), progress=Mock(),
        _quick_frame=Mock(side_effect=captures), _translated_plot=lambda a, b, p: p,
        vision=SimpleNamespace(harvest=Mock(return_value=harvest)), client=Mock(),
        state={'items': {'key': {'operation': 'repair', 'replant_grid': {}}}},
        _evidence=Mock(return_value={}), _record=Mock(), _sweep_duration=Mock(return_value=1000),
        cancel_event=threading.Event())
    assert repair.repair_remaining(worker, frame, 'key', object()) is captures[-1]
    route = worker.client.drag_path.call_args.args[0]
    assert set(crops.values()) <= set(route)
    assert '2' not in seen and '3' not in seen


def test_recorded_truncated_seed_drag_and_its_completed_recovery():
    before = shot('reliability/seed_fault_before.png')
    partial = shot('reliability/seed_fault_after.png')
    repaired = shot('reliability/seed_fault_repaired.png')
    points = json.loads((ROOT/'reliability/seed_fault_points.json').read_text())
    worker = WheatCropWorker.__new__(WheatCropWorker)
    worker.vision = WheatFarmingVision()
    plot = worker.vision.empty_plot(before.png)
    states = worker._observed_planting(before, partial, points, plot, allow_bare=True)
    assert states == ('growing',)*15+('bare',)*7
    assert worker._observed_planting(before, repaired, points, plot) == ('growing',)*22

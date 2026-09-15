"""Regression evidence from both live MuMu farms and expanded layout cases."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.wheating import WheatingPrerequisite
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_field_group import lattice_pitch, observe_group
from hayday.wheating_transient import WheatAchievement
from hayday.wheating_vision import WheatingVision

ROOT = Path(__file__).parent/'fixtures/multi'


def shot(name):
    return Screenshot((ROOT/name).read_bytes(), 1920, 1080, name)


@pytest.mark.parametrize('farm,count', [('large', 51), ('small', 9), ('upper_row', 51)])
def test_mixed_rotations_and_overlapping_objects_have_exact_bare_coverage(farm, count):
    vision = WheatFarmingVision()
    seed, bare = shot(f'{farm}_seed.png'), shot(f'{farm}_bare.png')
    plot = vision.empty_plot(seed.png)
    assert plot is not None and vision.full_outline
    origin = WheatCropWorker._translated_plot(seed, bare, plot.center, require_visible=False)
    assert origin is not None
    shifted = replace(plot, x=origin[0]-plot.width//2, y=origin[1]-plot.height//2)
    points = vision.empty_tiles(bare, shifted, 512)
    assert len(points) == len(set(points)) == count


def test_ripe_overhang_does_not_invent_extra_growing_plots():
    seed, mature = shot('large_seed.png'), shot('large_mature.png')
    points = json.loads((ROOT/'large_layout.json').read_text())['points']
    moved = WheatCropWorker._translated_plot(seed, mature, points[0], require_visible=False)
    assert moved is not None
    projected = np.asarray(points)+np.subtract(moved, points[0])
    result = observe_group(mature, projected, lattice_pitch(points),
                           WheatFarmingVision()._soil_textures, bounded=True)
    assert result is not None
    assert (result.count('ripe'), result.count('growing'), result.count('bare')) == (51, 0, 0)


@pytest.mark.parametrize('pitch,side', [(20, 2), (24, 11), (27, 15)])
def test_irregular_lattice_with_holes_scales_beyond_98_plots(pitch, side):
    image = np.full((1080, 1920, 3), (25, 175, 25), np.uint8)
    dx, dy = pitch, pitch/2
    points = []
    for i in range(side):
        for j in range(side):
            if 0 < i < side-1 and 0 < j < side-1 and (i+j) % 7 == 0:
                continue
            x, y = round(960+(i-j)*dx), round(290+(i+j)*dy)
            points.append((x, y))
            cv2.fillConvexPoly(image, np.int32([(x-dx,y),(x,y-dy),(x+dx,y),(x,y+dy)]), (0,220,255))
    frame = Screenshot(cv2.imencode('.png', image)[1].tobytes(), 1920, 1080, 'grid')
    group = observe_group(frame, points, (dx,dy), WheatFarmingVision()._soil_textures, bounded=True)
    assert group is not None and group.count('ripe') == len(points)
    if side >= 11:
        assert len(points) > 98


def test_clipped_field_is_found_for_camera_recovery():
    vision = WheatingVision()
    frame = shot('small_edge.png')
    assert vision.field_bounds(frame) is None
    bounds = vision.edge_field_bounds(frame)
    assert bounds is not None and bounds.x > 1600 and bounds.x+bounds.width == frame.width
    assert vision.edge_field_bounds(shot('large_bare.png')) is None


def test_achievement_waits_for_two_clear_frames_and_never_taps():
    run = SimpleNamespace(vision=WheatingVision(), publish=Mock(), wait=Mock(), client=Mock())
    recovery = WheatAchievement(run)
    blocked, clear = shot('achievement.png'), shot('small_bare.png')
    assert recovery.visible(blocked) and not recovery.visible(clear)
    frames = [replace(blocked, captured_at='2'), replace(clear, captured_at='3'),
              replace(clear, captured_at='4')]
    run._capture_raw = Mock(side_effect=frames)
    assert recovery.process(blocked) is frames[-1]
    assert run._capture_raw.call_count == 3
    assert not run.client.mock_calls


def test_stuck_achievement_does_not_restart_the_game():
    frame = shot('achievement.png')
    run = SimpleNamespace(vision=WheatingVision(), publish=Mock(), wait=Mock(),
                          _capture_raw=Mock(return_value=frame))
    with pytest.raises(WheatingPrerequisite, match='achievement overlay'):
        WheatAchievement(run).process(frame)
    assert run._capture_raw.call_count == 16


@pytest.mark.parametrize('name', ['interrupted_bare.png', 'interrupted_shifted.png'])
def test_cancel_before_seed_contact_is_reconciled_as_bare_without_counting_planting(name):
    worker = WheatCropWorker.__new__(WheatCropWorker)
    worker.vision = WheatFarmingVision()
    before, bare = shot('interrupted_before.png'), shot(name)
    points = json.loads((ROOT/'interrupted_points.json').read_text())
    plot = worker.vision.empty_plot(before.png)
    assert worker._planted_state(before, bare, points, plot) is None
    assert worker._observed_planting(before, bare, points, plot, allow_bare=True) == ('bare',)*51


def test_bare_seed_reserve_is_kept_before_a_partial_field_visits_the_shop():
    from hayday.wheating_fields import WheatFields
    fields = WheatFields.__new__(WheatFields)
    fields.run = SimpleNamespace(state={}, persist=Mock(), vision=SimpleNamespace(plots=Mock(return_value=[object()])))
    fields._group = SimpleNamespace(seed_reserve=1)
    fields._clear = Mock(return_value=shot('small_bare.png'))
    fields.bare_seed_reserve = Mock(return_value=8)
    fields.protect_bare_seeds()
    assert fields.run.state['seed_reserve'] == 8


def test_three_digit_stock_bubble_does_not_hide_the_wheat_seed(tmp_path):
    frame = shot('seed_stock_209.png')
    worker = WheatCropWorker(SimpleNamespace(serial='test'), lambda: frame, threading.Event(),
                             Mock(), tmp_path/'fields.json', vision=WheatFarmingVision())
    plot = worker.vision.empty_plot(frame.png)
    seed = worker._seed(frame, None, plot)
    assert seed is not None and worker._count(frame, seed) == 209


def test_brown_pig_pen_is_not_an_extra_plot_and_dense_sprouts_verify():
    vision = WheatFarmingVision()
    seed, bare, growing = shot('pen_seed.png'), shot('pen_bare.png'), shot('pen_growing.png')
    plot = vision.empty_plot(seed.png)
    origin = WheatCropWorker._translated_plot(seed, bare, plot.center, require_visible=False)
    shifted = replace(plot, x=origin[0]-plot.width//2, y=origin[1]-plot.height//2)
    points = vision.empty_tiles(bare, shifted, 512)
    assert len(points) == 51
    projected = [tuple(map(int, np.add(p, np.subtract(plot.center, origin)))) for p in points]
    assert (1109, 512) not in projected  # Brown pen floor aligned with the lattice.
    worker = WheatCropWorker.__new__(WheatCropWorker)
    worker.vision = vision
    assert worker._observed_planting(seed, growing, projected, plot) == ('growing',)*51


def test_right_clipped_wheat_is_centered_before_a_harvest_route_is_built():
    from hayday.resource_vision import VisualTarget
    from hayday.wheating_fields import WheatFields
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = None, []
    clipped, clear = shot('third_clipped.png'), shot('third_ripe.png')
    vision = WheatingVision()
    # The ripe mask stops at 88% of screen width. A bounding box touching
    # that cutoff cannot establish that the rest of the field is visible.
    vision.field_bounds = Mock(side_effect=[VisualTarget(1400, 300, 289, 200, 1.),
                                           VisualTarget(700, 300, 500, 200, 1.)])
    fields.run = SimpleNamespace(vision=vision, check=Mock(), client=Mock(),
                                 wait=Mock(), capture=Mock(return_value=clear))
    assert fields._center_field(clipped) is clear
    fields.run.client.swipe.assert_called_once()


def test_complete_saved_field_takes_precedence_over_clipped_foliage():
    from hayday.wheating_fields import WheatFields
    from hayday.wheating_restart_camera import farm_transform
    fields = WheatFields.__new__(WheatFields)
    fields._known_before = shot('shop_return_seed.png')
    fields._known_points = json.loads((ROOT/'shop_return_points.json').read_text())
    edge = shot('shop_return_edge.png')
    matrix = farm_transform(fields._known_before, edge)
    assert matrix is not None
    projected = np.c_[fields._known_points, np.ones(9)] @ matrix.T
    assert projected[:, 0].max() > edge.width*.90
    from hayday.resource_vision import VisualTarget
    vision = SimpleNamespace(field_bounds=Mock(return_value=VisualTarget(600, 300, 300, 200, 1.)),
                              farm=lambda f: True)
    fields.run = SimpleNamespace(vision=vision, check=Mock(), client=Mock(), wait=Mock(),
                                 capture=Mock(return_value=fields._known_before))
    fields._center_field(edge)
    fields.run.client.swipe.assert_called_once()
    vision.field_bounds.assert_not_called()


def test_translation_rejects_competing_matches_and_local_clusters():
    from hayday.wheating_restart_camera import _distributed_translation
    rng = np.random.default_rng(821)
    old = rng.uniform((250, 250), (1500, 850), (120, 2))
    shifted = old[:60]+(400, -160)
    # Unrelated descriptor matches cannot suppress a strong, distributed shift.
    mixed = np.concatenate([shifted, rng.uniform((200, 150), (1700, 950), (60, 2))])
    matrix = _distributed_translation(old, mixed, 1920, 1080)
    np.testing.assert_allclose(matrix, [[1, 0, 400], [0, 1, -160]])
    tied = np.concatenate([shifted, old[60:]+(-400, 160)])
    assert _distributed_translation(old, tied, 1920, 1080) is None
    local = rng.uniform((500, 400), (550, 450), (120, 2))
    assert _distributed_translation(local, local+(400, -160), 1920, 1080) is None


def test_ripe_stalks_over_bare_soil_do_not_start_a_false_growth_wait():
    evidence = json.loads((ROOT/'silo_overhang.json').read_text())
    frame = shot('silo_overhang.png')
    points = [cell[:2] for cell in evidence['cells']]
    group = observe_group(frame, points, evidence['pitch'], WheatFarmingVision()._soil_textures,
                          bounded=True)
    assert group is not None and group.count('growing') == 0
    assert group.count('ripe') >= 10 and group.count('bare') >= 30


def test_isolated_bare_tile_is_found_between_growing_wheat_and_buildings():
    frame = shot('last_bare_tile.png')
    candidates = WheatingVision().plots(frame, 'empty')
    assert len(candidates) == 1
    assert np.linalg.norm(np.subtract(candidates[0].center, (706, 443))) < 12


def test_picker_translation_can_follow_a_large_verified_camera_shift():
    seed, edge = shot('shop_return_seed.png'), shot('shop_return_edge.png')
    assert WheatCropWorker._orb_translated_plot(seed, edge, (800, 700), require_visible=False) is None
    moved = WheatCropWorker._translated_plot(seed, edge, (800, 700), require_visible=False)
    assert moved is not None and np.linalg.norm(np.subtract(moved, (1594, 477))) < 3
    assert WheatCropWorker._translated_plot(seed, edge, (1064, 632)) is None


def test_smoke_does_not_split_a_complete_soil_outline():
    vision = WheatFarmingVision()
    plot = vision.empty_plot(shot('smoky_outline.png').png)
    assert plot is not None and vision.full_outline
    assert (plot.width, plot.height) == (117, 65)
    assert np.linalg.norm(np.subtract(plot.center, (933, 675))) < 4


def test_residual_harvest_finds_only_the_one_missed_tile():
    from hayday.wheating_harvest_repair import ripe_tiles
    evidence = json.loads((ROOT/'residual_grid.json').read_text())
    grid = shot('residual_grid.png'), evidence['points'], evidence['pitch']
    worker = WheatCropWorker.__new__(WheatCropWorker)
    remaining = ripe_tiles(worker, shot('residual_ripe.png'), grid)
    assert remaining == {41: (598, 323)}


def test_moving_harvest_rewards_cannot_authorize_a_repair(monkeypatch):
    from hayday import wheating_harvest_repair as repair
    frames = [shot('residual_rewards.png'), shot('residual_ripe.png')]
    worker = SimpleNamespace(_pause=Mock(), _quick_frame=Mock(side_effect=frames), client=Mock())
    # Both frames contain yellow, but on different cells: no stable ripe crop.
    monkeypatch.setattr(repair, 'ripe_tiles', Mock(side_effect=[{0: (500, 400)},
                        {0: (500, 400)}, {1: (560, 428)}]))
    assert repair.repair_remaining(worker, frames[0], 'key', object()) is frames[1]
    assert not worker.client.mock_calls


def test_exhausted_repairs_defer_remaining_wheat_without_blocking_soil(monkeypatch):
    from hayday import wheating_harvest_repair as repair
    frames = [shot('residual_rewards.png'), shot('residual_ripe.png')]
    worker = SimpleNamespace(_pause=Mock(), _quick_frame=Mock(side_effect=frames), client=Mock(), progress=Mock(),
                             state={'items': {'key': {'harvest_repairs': [{}, {}, {}]}}})
    monkeypatch.setattr(repair, 'ripe_tiles', Mock(return_value={0: (500, 400)}))
    assert repair.repair_remaining(worker, frames[0], 'key', object()) is frames[-1]
    worker.progress.assert_called_once()
    assert not worker.client.mock_calls

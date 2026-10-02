"""Reconcile failed single-plot planting without replaying uncertain seed use."""

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.farming import FarmingWorker
from hayday.resource_vision import VisualTarget
from hayday.resources import ResourceResult

FIXTURES = Path(__file__).parent/'fixtures/orders'


def pending_worker(tmp_path):
    frame = Screenshot((FIXTURES/'cotton_failed_planting.png').read_bytes(), 1920, 1080, 'first')
    client = Mock(serial='test')
    worker = FarmingWorker(client, Mock(return_value=replace(frame, captured_at='second')),
                            threading.Event(), Mock(), tmp_path/'farming.json')
    worker._record('cotton', 'plant_attempted', available_before=1, operation='earlier-plant')
    worker._close = Mock()
    return worker, frame, (FIXTURES/'cotton_seed_icon.png').read_bytes()


def test_native_unchanged_picker_reconciles_no_effect_without_replanting(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    result = worker.work_if_recognized(frame, icon, 'cotton')
    assert result.status == 'waiting'
    saved = json.loads(worker.state_path.read_text())['items']['cotton']
    assert saved['stage'] == 'plant_no_effect'
    receipt = saved['plant_receipt']
    assert receipt['operation'] == 'earlier-plant' and receipt['before'] == receipt['after'] == 1
    assert len(receipt['evidence']) == 2
    worker.client.swipe.assert_not_called()
    worker.client.drag_path.assert_not_called()
    restored = FarmingWorker(worker.client, worker.capture, threading.Event(), Mock(), worker.state_path)
    assert restored.state['items']['cotton']['plant_receipt'] == receipt


@pytest.mark.parametrize('counts', [(0, 0), (1, 0), (1, None), (1, 2)])
def test_changed_or_unreadable_stock_keeps_planting_intent(tmp_path, counts):
    worker, frame, icon = pending_worker(tmp_path)
    worker._count = Mock(side_effect=counts)
    result = worker.work_if_recognized(frame, icon, 'cotton')
    assert result.status == 'unsupported'
    assert worker.state['items']['cotton']['stage'] == 'plant_attempted'
    worker.client.drag_path.assert_not_called()
    worker._close.assert_not_called()


def test_reused_capture_cannot_prove_planting_had_no_effect(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker.capture.return_value = frame
    assert worker.work_if_recognized(frame, icon, 'cotton').status == 'unsupported'
    assert worker.state['items']['cotton']['stage'] == 'plant_attempted'


def test_disconnected_planting_saves_original_plot_and_seed_evidence(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker._size = (1920, 1080)
    worker.client.drag_path.side_effect = RuntimeError('disconnected')
    with pytest.raises(RuntimeError, match='disconnected'):
        worker._plant_selected(frame, icon, 'cotton')
    saved = json.loads(worker.state_path.read_text())['items']['cotton']
    assert saved['stage'] == 'plant_attempted'
    assert saved['plant_before']['plot'] and saved['plant_before']['seed']
    assert (tmp_path/'farming_evidence'/saved['plant_before']['file']).is_file()
    worker.client.drag_path.assert_called_once()
    worker.client.swipe.assert_not_called()


@pytest.mark.parametrize('stock,required,growing,planned,shortfall', [
    (1, 3, 0, 1, 3),  # One seed cannot meet the recipe in its first cycle.
    (2, 3, 0, 2, 2),  # Reinvest both harvested cotton into two separate plots.
    (0, 3, 2, 0, 0),  # Four crops already growing cover recipe plus one seed.
    (2, 3, 1, 0, 0),  # Harvesting one of those plots must not start another cycle.
    (4, 3, 0, 0, 0),  # Three for production and a seed for future requirements.
    (3, 6, 0, 3, 4),
])
def test_crop_budget_accounts_for_seed_cost_and_existing_growth(stock, required, growing, planned, shortfall):
    assert FarmingWorker._plant_budget(stock, required, growing) == (planned, shortfall)


def test_recipe_requirement_and_seed_limited_plan_survive_restart(tmp_path):
    worker, _, _ = pending_worker(tmp_path)
    worker._record('cotton', 'growing')
    worker._requirement('cotton', {'available': 0, 'required': 3, 'captured_at': 'recipe'})
    plan = worker._plan('cotton', 0)
    assert plan['seed_limited'] and plan['shortfall'] == 2
    assert worker.state['items']['cotton']['growing_plots'] == 1
    restored = FarmingWorker(worker.client, worker.capture, threading.Event(), Mock(), worker.state_path)
    assert restored.state['items']['cotton']['demand']['required'] == 3
    result = restored._continue_planting(ResourceResult('waiting', ''), b'cotton', 'cotton')
    assert result.details['seed_limited']
    assert 'another harvest cycle' in result.message
    worker.client.drag_path.assert_not_called()


def test_confirmed_plant_keeps_second_plot_due_and_saves_growth_evidence(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker._record('cotton', 'plant_no_effect')
    worker._requirement('cotton', {'available': 2, 'required': 3})
    plot = VisualTarget(807, 599, 74, 54, .99)
    seed = VisualTarget(646, 182, 120, 116, .99)
    timer = VisualTarget(600, 500, 500, 50, .99)
    worker._size = (1920, 1080)
    worker._seed = Mock(return_value=seed)
    worker._count = Mock(return_value=2)
    worker._see = Mock(side_effect=lambda method, f: {'seed_menu': True, 'empty_plot': plot, 'growing': timer}[method])
    worker._pause = Mock()
    worker.capture = Mock(side_effect=[replace(frame, captured_at=stamp) for stamp in ('checked', 'grow1', 'grow2')])
    result = worker._plant_selected(frame, icon, 'cotton', plot)
    assert result.status == 'waiting'
    saved = worker.state['items']['cotton']
    assert saved['planting_plan']['planned'] == 2 and saved['planting_plan']['confirmed'] == 1
    assert saved['growing_plots'] == 1
    assert len(saved['plant_confirmed']) == 2
    restored = FarmingWorker(worker.client, worker.capture, threading.Event(), Mock(), worker.state_path)
    assert restored.state['items']['cotton']['planting_plan']['confirmed'] == 1


def test_existing_growth_does_not_strand_an_unfinished_multi_plot_plan(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker._record('cotton', 'growing', growing_plots=1, demand={'required': 3, 'available': 1},
                   planting_plan={'required': 3, 'stock_before': 2, 'planned': 2,
                                  'confirmed': 1, 'shortfall': 2, 'seed_limited': False, 'operations': ['first']})
    worker._size = (1920, 1080)
    plot = VisualTarget(800, 600, 100, 60, .99)
    worker._see = Mock(side_effect=lambda method, f: (plot,) if method == 'empty_plots' else None)
    worker._tap = Mock(return_value=frame)
    worker._translated_plot = Mock(return_value=plot.center)
    worker._await_seed_picker = Mock(return_value=(frame, plot))
    def finish(*args):
        entry = worker.state['items']['cotton']
        worker._record('cotton', 'growing', growing_plots=2,
                       planting_plan={**entry['planting_plan'], 'confirmed': 2, 'operations': ['first', 'second']})
        return ResourceResult('waiting', 'Planted')
    worker._plant_selected = Mock(side_effect=finish)
    result = worker._continue_planting(ResourceResult('waiting', ''), icon, 'cotton')
    worker._plant_selected.assert_called_once()
    assert result.details['plots_remaining'] == 0 and result.details['growing_plots'] == 2


def test_no_empty_plot_keeps_quantity_debt_without_input(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker._record('cotton', 'growing', growing_plots=1, demand={'required': 3, 'available': 1},
                   planting_plan={'required': 3, 'stock_before': 2, 'planned': 2,
                                  'confirmed': 1, 'shortfall': 2, 'seed_limited': False, 'operations': ['first']})
    worker._size = (1920, 1080)
    worker._see = Mock(return_value=None)
    result = worker._continue_planting(ResourceResult('waiting', ''), icon, 'cotton')
    assert result.details['plots_remaining'] == 1
    worker.client.tap.assert_not_called()
    worker.client.drag_path.assert_not_called()


def test_native_empty_soil_candidates_are_reused_for_extra_crop_selection():
    from hayday.farming_vision import FarmingVision
    targets = FarmingVision().empty_plots((FIXTURES/'home_farm.png').read_bytes())
    assert len(targets) >= 6
    assert all(300 < t.center[0] < 760 and 450 < t.center[1] < 690 for t in targets)


def test_recipe_quantity_reaches_the_crop_worker(tmp_path):
    from types import SimpleNamespace

    from hayday.resources import ResourceWorker
    fields = SimpleNamespace(work_if_recognized=Mock(return_value=ResourceResult('waiting', 'crop')))
    unrelated = SimpleNamespace(work_if_recognized=lambda *a, **kw: None)
    worker = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json', fields=fields,
        fruits=unrelated, fishing=unrelated,
        vision=SimpleNamespace(observe=lambda *a, **kw: SimpleNamespace(popups=[])))
    baseline = {'available': 1, 'required': 3}
    worker._state['items']['cotton'] = {'inventory_before': baseline}
    frame = Screenshot(b'frame', 1920, 1080, 'recipe')
    worker._at_source(frame, b'icon', 'cotton', None, (), 0)
    fields.work_if_recognized.assert_called_once_with(frame, b'icon', 'cotton', baseline=baseline)


def test_harvested_stock_can_satisfy_recipe_without_spending_it_on_replanting(tmp_path):
    worker, frame, icon = pending_worker(tmp_path)
    worker._size = (1920, 1080)
    worker._record('cotton', 'harvested_needs_replant', growing_plots=0)
    worker._requirement('cotton', {'available': 2, 'required': 3})
    worker._count = Mock(return_value=4)  # Fresh picker after harvesting the last plot.
    result = worker._plant_selected(frame, icon, 'cotton', harvested=True,
                                     plot=worker.vision.empty_plot(frame.png))
    assert result.status == 'waiting'
    assert worker.state['items']['cotton']['stage'] == 'stock_check'
    assert worker.state['items']['cotton']['planting_plan']['planned'] == 0
    worker.client.drag_path.assert_not_called()


def test_corrupt_crop_plan_cannot_authorize_more_planting(tmp_path):
    worker, _, _ = pending_worker(tmp_path)
    worker._record('cotton', 'growing', growing_plots=1, demand={'required': 3},
                   planting_plan={'required': 3, 'stock_before': 2, 'planned': 2,
                                  'confirmed': 3, 'shortfall': 2, 'seed_limited': False, 'operations': []})
    with pytest.raises(ValueError, match='planting plan'):
        FarmingWorker(worker.client, worker.capture, threading.Event(), Mock(), worker.state_path)
    worker.client.drag_path.assert_not_called()


@pytest.mark.parametrize('change', [None, 'cloud', 'duplicate', 'late', 'corrupt', 'missing', 'crop_removed'])
def test_native_ineffective_harvest_requires_two_intact_immediate_observations(tmp_path, change):
    worker, _, _ = pending_worker(tmp_path)
    proofs = []
    folder = tmp_path/'farming_evidence'
    folder.mkdir(exist_ok=True)
    for stage, stamp in [('before', '00'), ('after_0', '15'), ('after_1', '18')]:
        prefix = 'cotton_cloud' if change == 'cloud' else 'cotton'
        data = (FIXTURES/f'{prefix}_no_effect_{stage}.png').read_bytes()
        if change == 'crop_removed' and stage != 'before':
            from hayday.resource_vision import _decode, _png
            image = _decode(data)
            image[462:525, 456:570] = (60, 110, 155)
            data = _png(image)
        name = f'old_harvest_{stage}.png'
        (folder/name).write_bytes(data)
        proofs.append({'file': name, 'sha256': hashlib.sha256(data).hexdigest(),
                       'width': 1920, 'height': 1080, 'captured_at': f'2026-10-01T20:28:{stamp}+00:00'})
    if change == 'duplicate':
        proofs[2]['captured_at'] = proofs[1]['captured_at']
    elif change == 'late':
        proofs[2]['captured_at'] = '2026-10-01T20:35:00+00:00'
    elif change == 'corrupt':
        (folder/proofs[2]['file']).write_bytes(b'changed')
    elif change == 'missing':
        proofs.pop()
    worker._record('cotton', 'harvest_attempted', operation='old', growing_plots=1,
                   harvest_before=proofs[0], harvest_after=proofs[1:])
    unchanged = change in (None, 'cloud')
    assert worker._unchanged_harvest('cotton') is unchanged
    entry = worker.state['items']['cotton']
    assert entry['stage'] == ('harvest_no_effect' if unchanged else 'harvest_attempted')
    assert entry['growing_plots'] == 1
    worker.client.drag_path.assert_not_called()
    worker.client.swipe.assert_not_called()


def test_harvest_starts_on_the_opaque_sickle_handle_instead_of_its_empty_center():
    from hayday.farming_vision import FarmingVision
    vision = FarmingVision()
    frame = (FIXTURES/'cotton_no_effect_before.png').read_bytes()
    sickle = vision._matches(frame, 'sickle', .90)[0]
    harvest = vision.harvest(frame)
    assert harvest is not None
    x = round((harvest.tool.center[0]-sickle.x)/harvest.scale)
    y = round((harvest.tool.center[1]-sickle.y)/harvest.scale)
    assert vision.references['sickle'][y, x, 3] >= 200
    assert 300 < harvest.tool.center[0] < 330 and 480 < harvest.tool.center[1] < 510
    geometry = vision.manifest['harvest_geometry']
    old_x, old_y = geometry['tool_point']
    assert vision.references['sickle'][old_y-geometry['sickle_box'][1],
                                        old_x-geometry['sickle_box'][0], 3] == 0


@pytest.mark.parametrize('prefix', ['cotton', 'cotton_cloud'])
@pytest.mark.parametrize('stage', ['before', 'after_0', 'after_1'])
def test_cotton_harvest_includes_the_full_selection_outline_below_the_white_bolls(prefix, stage):
    from hayday.farming_vision import FarmingVision
    harvest = FarmingVision().harvest((FIXTURES/f'{prefix}_no_effect_{stage}.png').read_bytes())
    assert harvest is not None
    assert harvest.highlight.x < 460 and harvest.highlight.width > 100
    assert harvest.highlight.y+harvest.highlight.height >= 563
    assert 500 < harvest.drag_target.center[0] < 528
    assert 535 < harvest.drag_target.center[1] < 560


@pytest.mark.parametrize('change', [None, 'plant_remains', 'same_frame', 'cloud'])
def test_harvest_soil_uses_removed_crop_instead_of_disappearing_palette(change):
    from hayday.farming_vision import FarmingVision
    from hayday.resource_vision import _decode, _png
    before = Screenshot((FIXTURES/'cotton_harvested_before.png').read_bytes(), 1920, 1080, 'before')
    data = (FIXTURES/'cotton_harvested_after_1.png').read_bytes()
    if change == 'plant_remains':
        image = _decode(data)
        image[461:565, 455:569] = _decode(before.png)[461:565, 455:569]
        data = _png(image)  # Palette disappeared, but the selected cotton remains.
    elif change == 'same_frame':
        data = before.png
    elif change == 'cloud':
        before = replace(before, png=(FIXTURES/'cotton_cloud_no_effect_before.png').read_bytes())
        data = (FIXTURES/'cotton_cloud_no_effect_after_1.png').read_bytes()
    after = Screenshot(data, 1920, 1080, 'after')
    soil = FarmingWorker._exposed_soil(before, after, FarmingVision().harvest(before.png))
    if change is None:
        assert soil is not None
        assert 500 < soil.center[0] < 525 and 535 < soil.center[1] < 560
    else:
        assert soil is None


@pytest.mark.parametrize('change', [None, 'corrupt', 'duplicate_saved', 'duplicate_fresh', 'replanted'])
def test_saved_successful_harvest_reopens_only_verified_soil_without_repeating_drag(tmp_path, change):
    worker, _, _ = pending_worker(tmp_path)
    worker._size = (1920, 1080)
    worker._pause = Mock()
    frames = [Screenshot((FIXTURES/f'cotton_harvested_{label}.png').read_bytes(), 1920, 1080,
                          f'2026-10-01T20:50:{stamp}+00:00')
              for label, stamp in [('before', '00'), ('after_0', '15'), ('after_1', '18')]]
    proofs = [worker._evidence(f, 'saved', f'harvest_{label}')
              for f, label in zip(frames, ['before', 'after_0', 'after_1'], strict=True)]
    if change == 'corrupt':
        (tmp_path/'farming_evidence'/proofs[1]['file']).write_bytes(b'corrupt')
    if change == 'duplicate_saved':
        proofs[2]['captured_at'] = proofs[1]['captured_at']
    worker._record('cotton', 'harvest_attempted', operation='saved', growing_plots=2,
                   harvest_before=proofs[0], harvest_after=proofs[1:])
    frame = replace(frames[-1], captured_at='fresh1')
    if change == 'replanted':
        frame = replace(frame, png=frames[0].png)
        worker._close = Mock(return_value=frame)
    worker.capture = Mock(return_value=replace(frame, captured_at='fresh1' if change == 'duplicate_fresh' else 'fresh2'))
    worker._await_seed_picker = Mock(return_value=(frame, VisualTarget(477, 524, 72, 54, .99)))
    recovered = worker._resume_harvested_soil(frame, 'cotton')
    assert (recovered is not None) == (change is None)
    if change is None:
        worker.client.tap.assert_called_once()
        entry = worker.state['items']['cotton']
        assert entry['growing_plots'] == 1
        assert entry['stage'] == 'harvested_needs_replant'
        assert entry['harvest_receipt']['outcome'] == 'harvested'
        restored = FarmingWorker(worker.client, worker.capture, threading.Event(), Mock(), worker.state_path)
        restored._confirm_crop_harvest('cotton')
        assert restored.state['items']['cotton']['growing_plots'] == 1
    else:
        worker.client.tap.assert_not_called()
    worker.client.drag_path.assert_not_called()
    worker.client.swipe.assert_not_called()


def test_seed_picker_dismissal_avoids_grass_showing_through_the_palette(tmp_path):
    worker, _, _ = pending_worker(tmp_path)
    frame = Screenshot((FIXTURES/'cotton_seed_palette.png').read_bytes(), 1920, 1080, 'picker')
    worker._size = (1920, 1080)
    worker._pause = Mock()
    region = worker.vision.seed_menu_region(frame.png)
    assert region is not None
    FarmingWorker._close(worker, frame)
    x, y = worker.client.tap.call_args.args
    assert x > 1000
    assert not (region.x <= x < region.x+region.width and region.y <= y < region.y+region.height)


def test_snow_covering_one_soil_candidate_does_not_strand_remaining_planting(tmp_path):
    worker, _, icon = pending_worker(tmp_path)
    worker._size = (1920, 1080)
    frames = [Screenshot((FIXTURES/f'cotton_empty_plots_{n}.png').read_bytes(), 1920, 1080, str(n))
              for n in (11, 12)]
    before = worker.vision.empty_plots(frames[0].png)
    after = worker.vision.empty_plots(frames[1].png)
    highest = max(before, key=lambda t: t.score)
    assert not any(worker._same_target(highest, t, frames[1]) for t in after)
    worker.capture = Mock(side_effect=frames)
    worker._record('cotton', 'growing', growing_plots=1, demand={'required': 3},
        planting_plan={'required': 3, 'stock_before': 2, 'planned': 2, 'confirmed': 1,
                       'shortfall': 2, 'seed_limited': False, 'operations': ['first']})
    worker._tap = Mock(return_value=frames[1])
    worker._await_seed_picker = Mock(return_value=(frames[1], after[0]))
    worker._plant_selected = Mock(return_value=ResourceResult('changed', 'Reached planting validation'))
    result = worker._continue_planting(ResourceResult('waiting', ''), icon, 'cotton')
    assert result.message == 'Reached planting validation'
    selected = worker._tap.call_args.args[0]
    assert any(t.center == selected and worker._same_target(c, t, frames[1]) for c in before for t in after)
    worker._plant_selected.assert_called_once()


def test_harvest_tracks_selected_soil_when_opening_picker_pans_camera(tmp_path):
    worker, _, icon = pending_worker(tmp_path)
    worker._record('cotton', 'growing', growing_plots=2)
    worker._pause = Mock()
    frames = [Screenshot((FIXTURES/f'cotton_picker_pan_{n}.png').read_bytes(),
                         1920, 1080, f'2026-10-02T00:08:{n}+00:00')
              for n in range(21, 27)]
    worker.capture = Mock(side_effect=frames)
    worker._plant_selected = Mock(return_value=ResourceResult('changed', 'Planting handoff'))
    worker._continue_planting = Mock(side_effect=lambda result, *args: result)
    result = worker.work_if_recognized(replace(frames[0], captured_at='initial'), icon,
                                       'cotton', {'available': 0, 'required': 3})
    assert result.message == 'Planting handoff'
    frame, _, _, plot, harvested, tracked = worker._plant_selected.call_args.args
    assert harvested and plot is not None and frame.captured_at == frames[-1].captured_at
    assert tracked == plot.center and 820 < tracked[0] < 860
    worker.client.drag_path.assert_called_once()
    worker.client.tap.assert_called_once()
    assert worker.client.tap.call_args.args[0] < 540  # Before the automatic camera pan.
    entry = worker.state['items']['cotton']
    assert entry['growing_plots'] == 1
    assert entry['replant_evidence']['plot'] == list(plot.box)


@pytest.mark.parametrize('replanted', [False, True])
def test_saved_harvest_recovery_tracks_zoom_but_still_requires_empty_original_plot(tmp_path, replanted):
    import cv2
    import numpy as np

    from hayday.resource_vision import _decode, _png

    worker, _, _ = pending_worker(tmp_path)
    worker._size = (1920, 1080)
    worker._pause = Mock()
    saved = [Screenshot((FIXTURES/f'cotton_picker_pan_{n}.png').read_bytes(),
                        1920, 1080, f'2026-10-02T00:08:{n}+00:00') for n in (21, 22, 23)]
    proofs = [worker._evidence(f, 'saved', f'harvest_{n}') for n, f in enumerate(saved)]
    worker._record('cotton', 'harvested_needs_replant', operation='saved', growing_plots=1,
                   harvest_before=proofs[0], harvest_after=proofs[1:],
                   harvest_receipt={'operation': 'saved', 'outcome': 'harvested'})
    current = Screenshot((FIXTURES/'cotton_recovery_zoom.png').read_bytes(), 1920, 1080, 'fresh1')
    harvest = worker.vision.harvest(saved[0].png)
    assert worker._translated_plot(saved[0], current, harvest.highlight.center) is None
    if replanted:
        matrix = worker._plot_transform(saved[0], current, allow_scale=True)
        restored = cv2.warpAffine(_decode(saved[0].png), matrix, (1920, 1080))
        mask = np.zeros((1080, 1920), np.uint8)
        x, y, w, h = harvest.highlight.box
        mask[y:y+h, x:x+w] = 255
        mask = cv2.warpAffine(mask, matrix, (1920, 1080))
        image = _decode(current.png)
        image[mask > 0] = restored[mask > 0]
        current = replace(current, png=_png(image))
    worker.capture = Mock(return_value=replace(current, captured_at='fresh2'))
    picker = VisualTarget(804, 596, 78, 58, .99)
    worker._await_seed_picker = Mock(return_value=(current, picker))
    recovered = worker._resume_harvested_soil(current, 'cotton')
    if replanted:
        assert recovered is None
        worker.client.tap.assert_not_called()
    else:
        assert recovered == (current, picker)
        x, y = worker.client.tap.call_args.args
        assert 290 < x < 320 and 530 < y < 550
        assert worker.state['items']['cotton']['growing_plots'] == 1
    worker.client.drag_path.assert_not_called()


@pytest.mark.parametrize('corrupt', [False, True])
def test_confirmed_harvest_does_not_reopen_soil_when_remaining_crop_budget_is_sufficient(tmp_path, corrupt):
    worker, current, icon = pending_worker(tmp_path)
    frames = [Screenshot((FIXTURES/f'cotton_picker_pan_{n}.png').read_bytes(), 1920, 1080,
                          f'2026-10-02T00:08:{n}+00:00') for n in (21, 22, 23)]
    proofs = [worker._evidence(f, 'saved', f'harvest_{n}') for n, f in enumerate(frames)]
    worker._record('cotton', 'harvested_needs_replant', operation='saved', growing_plots=2,
                   harvest_before=proofs[0], harvest_after=proofs[1:])
    if corrupt:
        (tmp_path/'farming_evidence'/proofs[1]['file']).write_bytes(b'changed')
    worker._resume_harvested_soil = Mock(return_value=None)
    result = worker.work_if_recognized(current, icon, 'cotton', {'available': 2, 'required': 3})
    entry = worker.state['items']['cotton']
    if corrupt:
        assert result.status == 'unsupported' and entry['stage'] == 'harvested_needs_replant'
        assert entry['growing_plots'] == 2
    else:
        assert result.status == 'waiting' and entry['stage'] == 'stock_check'
        assert entry['growing_plots'] == 1
        assert entry['harvest_receipt']['outcome'] == 'harvested'
        worker._resume_harvested_soil.assert_not_called()
    worker.client.tap.assert_not_called()
    worker.client.drag_path.assert_not_called()

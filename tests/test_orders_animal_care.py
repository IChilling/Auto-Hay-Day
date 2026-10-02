"""Animal care uses feed states, fresh stock and durable production ownership."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.animal_care import AnimalCareWorker, request_feeding
from hayday.herd_vision import ANIMALS, HerdVision
from hayday.resource_vision import VisualTarget
from hayday.resources import ResourceChanged, ResourceResult

FIXTURES = Path(__file__).parent/'fixtures/orders'
REFERENCES = Path(__file__).resolve().parents[1]/'images/reference_captures'


@pytest.fixture(scope='module')
def vision():
    return HerdVision()


@pytest.mark.parametrize('animal,name,stock,enabled', [
    ('sheep', 'sheep_accessory_ready.png', 0, True),
    ('cow', 'cow_accessory_grey_bucket.png', 0, True),
    ('cow', 'cow_feed_after.png', 1, False),
    ('pig', 'herd_pig-menu.png', 6, True),
    ('goat', 'herd_goat-menu.png', 7, False),
])
def test_feed_stock_and_enabled_state_are_independent(vision, animal, name, stock, enabled):
    result = vision.menu((FIXTURES/name).read_bytes(), animal, lambda: False)
    assert result and result.available == stock and result.enabled is enabled


def test_chicken_feed_can_be_disabled_with_positive_stock(vision):
    result = vision.menu((REFERENCES/'resource_chicken_fed.png').read_bytes(), 'chicken', lambda: False)
    assert result and result.available == 9 and not result.enabled


@pytest.mark.parametrize('animal', ANIMALS)
def test_feed_mill_is_not_an_animal_menu(vision, animal):
    png = (FIXTURES/'herd_feed-mill.png').read_bytes()
    assert vision.menu(png, animal, lambda: False) is None


def test_mill_cow_feed_cannot_block_menu_dismissal(vision):
    frame = Screenshot((FIXTURES/'herd_feed-mill.png').read_bytes(), 1920, 1080, 'mill')
    assert not vision.cow.menu_visible(frame.png, lambda: False)
    assert vision.cow.menu_ground(frame, lambda: False) is None


def test_verified_clear_pen_survives_the_pig_tool_overlay(vision):
    before = Screenshot((FIXTURES/'herd_all-pens.png').read_bytes(), 1920, 1080, 'before')
    after = Screenshot((FIXTURES/'herd_pig-menu.png').read_bytes(), 1920, 1080, 'after')
    pens = vision.pens(before.png, 'pig', lambda: False)
    assert len(pens) == 2
    confirmed = [vision.confirm_pen(before, after, 'pig', pen, lambda: False) for pen in pens]
    assert any(confirmed)
    assert not vision.confirm_pen(before, after, 'goat', pens[0], lambda: False)


def request(path, **changes):
    defaults = dict(animal='sheep', frame=Screenshot((FIXTURES/'sheep_accessory_ready.png').read_bytes(),
        1920, 1080, 'harvest'), polygon=((1297, 601), (1108, 673), (963, 606), (1114, 521)),
        item_key='wool', harvest_operation='harvest-one')
    defaults.update(changes)
    return request_feeding(path, 'test', **defaults)


def test_new_harvest_requests_only_its_pen_and_preserves_uncertain_input(tmp_path):
    path = tmp_path/'animal_care.json'
    job = request(path)
    state = json.loads(path.read_text())
    assert list(state['jobs']) == [job]
    assert state['jobs'][job]['animal'] == 'sheep'
    state['pending'] = {'job': job, 'animal': 'sheep', 'stock_before': 3}
    state['jobs'][job].update(stage='waiting_feed', feed_key='feed', next_due=900)
    path.write_text(json.dumps(state))
    request(path)
    updated = json.loads(path.read_text())
    assert updated == state


def test_old_farm_wide_requests_are_retired_without_losing_uncertain_intent(tmp_path):
    path = tmp_path/'animal_care.json'
    old = {'version': 1, 'serial': 'test', 'animals': {a: {'stage': 'needed'} for a in ANIMALS},
           'pending': {'animal': 'pig', 'stock_before': 6}}
    path.write_text(json.dumps(old))
    request(path)
    state = json.loads(path.read_text())
    assert len(state['jobs']) == 1
    assert state['pending'] == old['pending']
    assert state['legacy'] == old


def test_replacement_harvest_updates_an_unexecuted_care_request(tmp_path):
    path = tmp_path/'animal_care.json'
    key = request(path, harvest_operation='cancelled-before-input')
    assert request(path, harvest_operation='fresh-harvest') == key
    state = json.loads(path.read_text())
    assert len(state['jobs']) == 1
    assert state['jobs'][key]['stage'] == 'awaiting_harvest'
    assert state['jobs'][key]['harvest_operation'] == 'fresh-harvest'


def test_unconfirmed_harvest_has_priority_over_feeding_navigation(tmp_path):
    request(tmp_path/'animal_care.json')
    owner = SimpleNamespace(state_path=tmp_path/'resources.json', serial='test',
        fruits=SimpleNamespace(state={'items': {'wool': {'stage': 'attempted', 'operation': 'harvest-one'}}}),
        cancel_event=threading.Event(), client=Mock())
    worker = AnimalCareWorker(owner)
    assert worker.step(Mock()) is None
    assert worker.vision is None
    owner.client.tap.assert_not_called()


def test_unconfirmed_harvest_also_precedes_older_unrelated_feeding_jobs(tmp_path):
    request(tmp_path/'animal_care.json', harvest_operation=None)
    owner = SimpleNamespace(state_path=tmp_path/'resources.json', serial='test',
        fruits=SimpleNamespace(state={'items': {'egg': {'stage': 'attempted', 'operation': 'new'}}}),
        cancel_event=threading.Event(), client=Mock())
    worker = AnimalCareWorker(owner)
    assert worker.step(Mock()) is None
    assert worker.vision is None
    owner.client.tap.assert_not_called()


def test_unscoped_uncertain_feed_blocks_further_actions(tmp_path):
    path = tmp_path/'animal_care.json'
    request(path)
    state = json.loads(path.read_text())
    state['pending'] = {'animal': 'pig', 'stock_before': 6}
    path.write_text(json.dumps(state))
    owner = SimpleNamespace(state_path=tmp_path/'resources.json', serial='test', client=Mock())
    result = AnimalCareWorker(owner).step(Mock())
    assert result.status == 'unsupported'
    owner.client.tap.assert_not_called()


@pytest.mark.parametrize('after,enabled,outcome,stage', [
    (1, False, 'fed', 'complete'), (0, True, 'fed', 'needed'),
    (3, True, 'no_effect', 'needed'), (3, False, 'already_fed', 'complete')])
def test_partial_or_failed_sweep_never_completes_a_hungry_pen(tmp_path, after, enabled, outcome, stage):
    worker = AnimalCareWorker(SimpleNamespace(state_path=tmp_path/'resources.json'))
    worker._evidence.mkdir()
    job = {'animal': 'sheep'}
    worker.state = {'pending': {'stock_before': 3, 'operation': 'feed-one'}, 'jobs': {'one': job}}
    worker._dismiss = Mock()
    result = worker._receipt(job, SimpleNamespace(png=b'recorded'), SimpleNamespace(available=after, enabled=enabled))
    assert job['stage'] == stage
    assert job['last_feed']['outcome'] == outcome
    assert result.details['feed_confirmed'] is (outcome == 'fed')
    assert worker.state['pending'] is None


def test_neighboring_pens_cannot_satisfy_a_specific_pen_request(tmp_path, vision):
    before = Screenshot((FIXTURES/'herd_all-pens.png').read_bytes(), 1920, 1080, 'before')
    pens = vision.pens(before.png, 'pig', lambda: False)
    assert len(pens) == 2
    job_id = request(tmp_path/'animal_care.json', animal='pig', frame=before,
                     polygon=pens[0][2], item_key='bacon')
    worker = AnimalCareWorker(SimpleNamespace(state_path=tmp_path/'resources.json', cancel_event=threading.Event()))
    worker.state = json.loads(worker.path.read_text())
    worker.vision = vision
    selected = worker._target_pens(worker.state['jobs'][job_id], before)
    assert len(selected) == 1
    assert selected[0][2] == pens[0][2]


def test_snowy_cliff_is_not_a_modal():
    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'herd_winter-cliff.png').read_bytes(), 1920, 1080, 'cliff')
    assert not CameraNavigator._modal_visible(frame)


def test_dimmed_popup_artwork_cannot_be_used_as_farm_ground():
    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'dimmed_illustrated_popup.png').read_bytes(), 1920, 1080, 'popup')
    assert CameraNavigator._modal_visible(frame)


def test_camera_swipe_cannot_end_on_the_boat():
    import cv2
    import numpy as np

    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'camera_boat_shore.png').read_bytes(), 1920, 1080, 'shore')
    hsv = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.png, np.uint8), 1), cv2.COLOR_BGR2HSV)
    candidates = [CameraNavigator._grass_start(frame, dx, 0) for dx in (.36, .18, .09)]
    assert any(candidates)
    for drag in filter(None, candidates):
        for x, y in (drag[:2], drag[2:]):
            hue, saturation, value = hsv[y, x]
            assert 24 <= hue <= 88 and saturation >= 75 and value >= 65
            assert not (750 <= x <= 1200 and 50 <= y <= 410)


def test_narrow_edge_keeps_both_swipe_endpoints_on_observed_ground():
    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'camera_narrow_edge.png').read_bytes(), 1920, 1080, 'edge')
    assert all(CameraNavigator._grass_start(frame, .36*fraction, 0) is None
               for fraction in (1., .5, .25))
    drag = CameraNavigator._search_drag(frame, .36, 0)
    assert drag is not None
    x, y, end_x, end_y = drag
    assert end_y == y and 50 < end_x-x < 100
    # The native frame's narrow clear patch is above the fence, away from
    # the nearby wishing well and cliff. Both contacts must fit inside it.
    assert 340 <= x < end_x <= 490 and 205 <= y <= 260


def test_thin_grass_between_forest_and_cliff_can_recover_without_a_speculative_tap():
    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'camera_thin_grass.png').read_bytes(), 1920, 1080, 'edge')
    assert CameraNavigator._grass_start(frame, .045, 0) is None
    drag = CameraNavigator._search_drag(frame, .36, 0)
    assert drag is not None and drag[2]-drag[0] == 43 and drag[1] == drag[3]


@pytest.mark.parametrize('before,after,stationary', [(59, 60, True), (61, 62, False)])
def test_picker_animation_cannot_hide_a_camera_edge(before, after, stationary):
    from hayday.camera import CameraNavigator
    frames = [Screenshot((FIXTURES/f'care_camera_{i}.png').read_bytes(), 1920, 1080, str(i))
              for i in (before, after)]
    assert not CameraNavigator._same_view(*(CameraNavigator._view(f) for f in frames))
    assert CameraNavigator._same_farm_position(*frames) is stationary


def test_stationary_world_can_settle_despite_menu_or_cloud_animation():
    from hayday.camera import CameraNavigator
    frames = [Screenshot((FIXTURES/f'care_camera_{i}.png').read_bytes(), 1920, 1080, str(i))
              for i in (59, 60)]
    navigator = CameraNavigator(Mock(serial='test'), capture=Mock(side_effect=frames), settle_seconds=0)
    assert navigator._settled_observe() is frames[1]


def test_accidental_cake_oven_picker_is_detected_before_camera_input():
    from hayday.resource_vision import ResourceVision
    scene = ResourceVision().observe((FIXTURES/'care_producer_picker.png').read_bytes())
    assert scene.popups or scene.empty_slots


def test_sugar_menu_dismissal_avoids_the_white_sugar_icon():
    from hayday.camera import CameraNavigator
    from hayday.resource_vision import ResourceVision
    frame = Screenshot((FIXTURES/'sugar_menu_dismissal.png').read_bytes(), 1920, 1080, 'sugar')
    scene = ResourceVision().observe(frame.png)
    old = CameraNavigator._grass_start(frame, 0, 0)
    assert 440 < old[0] < 510 and 320 < old[1] < 450
    ground = CameraNavigator._menu_ground(frame, scene)
    assert ground is not None and 700 < ground[0] < 810 and 800 < ground[1] < 860


def test_wide_feed_mill_queue_leaves_verified_ground_beside_the_palette():
    from hayday.camera import CameraNavigator
    from hayday.resource_vision import ResourceVision
    frame = Screenshot((FIXTURES/'feed_mill_menu_dismissal.png').read_bytes(), 1920, 1080, 'mill')
    scene = ResourceVision().observe(frame.png)
    ground = CameraNavigator._menu_ground(frame, scene)
    assert ground is not None and 1440 < ground[0] < 1530 and 550 < ground[1] < 600


def test_feeding_waits_for_order_panel_close_animation(tmp_path):
    def captured(name, stamp):
        return Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, stamp)

    board = captured('care_board_before_close.png', 'board')
    owner = Mock(state_path=tmp_path/'resources.json', cancel_event=threading.Event())
    owner._capture.side_effect = [captured('care_board_before_close.png', 'confirmed'),
        captured('care_board_closed.png', 'clear-one'), captured('care_board_closed.png', 'clear-two')]
    owner._tap.return_value = captured('care_board_fading.png', 'fading')
    assert AnimalCareWorker(owner)._close_board(board).captured_at == 'clear-two'
    assert owner._tap.call_count == 1


def test_menu_illumination_preserves_the_just_verified_trough(vision):
    before = Screenshot((FIXTURES/'sheep_pen_before_feed.png').read_bytes(), 1920, 1080, 'before')
    after = Screenshot((FIXTURES/'sheep_feed_illuminated.png').read_bytes(), 1920, 1080, 'after')
    pens = vision.pens(before.png, 'sheep', lambda: False)
    assert len(pens) == 1
    assert vision.confirm_pen(before, after, 'sheep', pens[0], lambda: False)


@pytest.mark.parametrize('hide_trough', [False, True])
def test_partial_feeding_timer_requires_a_still_visible_trough_and_fence(vision, hide_trough):
    import cv2
    import numpy as np
    before = Screenshot((FIXTURES/'sheep_partial_feed_clear.png').read_bytes(), 1920, 1080, 'before')
    png = (FIXTURES/'sheep_partial_feed_timer.png').read_bytes()
    pen = vision.pens(before.png, 'sheep', lambda: False)[0]
    if hide_trough:
        image = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
        x, y, w, h = pen[0].box
        image[y:y+h, x:x+w] = (20, 140, 40)
        png = cv2.imencode('.png', image)[1].tobytes()
    after = Screenshot(png, 1920, 1080, 'after')
    confirmed = vision.confirm_pen(before, after, 'sheep', pen, lambda: False)
    assert bool(confirmed) is (not hide_trough)


def test_saved_pen_can_be_relocated_after_camera_zoom_without_relaxing_plot_gestures():
    import cv2
    import numpy as np

    from hayday.farming import FarmingWorker
    before = Screenshot((FIXTURES/'sheep_pen_before_feed.png').read_bytes(), 1920, 1080, 'before')
    image = cv2.imdecode(np.frombuffer(before.png, np.uint8), 1)
    matrix = np.float32([[.8, 0, 192], [0, .8, 108]])
    changed = cv2.warpAffine(image, matrix, (1920, 1080))
    after = Screenshot(cv2.imencode('.png', changed)[1].tobytes(), 1920, 1080, 'zoomed')
    center = (1066, 600)
    assert FarmingWorker._translated_plot(before, after, center) is None
    relocated = FarmingWorker._translated_plot(before, after, center, allow_scale=True)
    assert relocated is not None
    assert np.linalg.norm(np.subtract(relocated, matrix @ np.array([*center, 1]))) < 3


def test_native_intermediate_view_relocates_pen_after_board_recenters(tmp_path, vision):
    from hayday.farming import FarmingWorker

    def frame(name):
        return Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, name)

    original = frame('care_saved_sheep.png')
    bridge = frame('care_bridge_sheep.png')
    distant = frame('care_distant_board.png')
    path = tmp_path/'animal_care.json'
    key = request(path, frame=original, harvest_operation=None,
                  polygon=((1252, 607), (1023, 683), (895, 609), (1079, 519)))
    owner = SimpleNamespace(state_path=tmp_path/'resources.json', cancel_event=threading.Event())
    worker = AnimalCareWorker(owner)
    worker.state = json.loads(path.read_text())
    worker.vision = vision
    job = worker.state['jobs'][key]
    original_record = dict(job)
    assert worker._project(job, distant) is None
    worker._remember_view(bridge)
    assert len(worker._target_pens(job, bridge)) == 1
    assert worker._project(job, distant) == (1961, 923)
    assert worker._target_pens(job, distant) == ()  # Navigation cannot authorize feeding offscreen.
    assert all(job[k] == value for k, value in original_record.items())

    restarted = AnimalCareWorker(owner)
    restarted.state = json.loads(path.read_text())
    saved = restarted.state['jobs'][key]
    assert restarted._project(saved, distant) == (1961, 923)
    # A bridge never creates the next bridge; retain the last direct match.
    assert FarmingWorker._translated_plot(original, distant, saved['center'],
                                         require_visible=False, allow_scale=True) is None
    restarted._remember_view(distant)
    assert saved['navigation_view'] == job['navigation_view']
    evidence = restarted._evidence/saved['navigation_view']['file']
    evidence.write_bytes(b'changed')
    with pytest.raises(ResourceChanged, match='evidence changed'):
        restarted._project(saved, distant)


def test_replacing_navigation_view_preserves_original_pen_and_pending_gesture(tmp_path):
    from dataclasses import replace
    path = tmp_path/'animal_care.json'
    key = request(path, harvest_operation=None)
    worker = AnimalCareWorker(SimpleNamespace(state_path=tmp_path/'resources.json'))
    worker.state = json.loads(path.read_text())
    job = worker.state['jobs'][key]
    original_path = worker._evidence/job['file']
    original_bytes = original_path.read_bytes()
    worker.state['pending'] = {'job': key, 'stock_before': 3, 'operation': 'unconfirmed'}
    first = Screenshot(original_bytes, 1920, 1080, 'first')
    worker._remember_view(first)
    previous = worker._evidence/job['navigation_view']['file']
    # Use a second native camera view with a verified original-world transform.
    second = replace(first, png=(FIXTURES/'sheep_pen_before_feed.png').read_bytes(), captured_at='second')
    worker._remember_view(second)
    assert job['navigation_view']['captured_at'] == 'second'
    assert not previous.exists()
    assert original_path.read_bytes() == original_bytes
    assert worker.state['pending']['operation'] == 'unconfirmed'


def test_camera_settling_requires_distinct_observations():
    from hayday.camera import CameraNavigator
    frame = Screenshot((FIXTURES/'care_camera_59.png').read_bytes(), 1920, 1080, 'stale')
    navigator = CameraNavigator(Mock(serial='test'), capture=Mock(return_value=frame), settle_seconds=0)
    with pytest.raises(TimeoutError, match='kept moving'):
        navigator._settled_observe()


def test_resource_finish_saves_clear_view_before_board_navigation(tmp_path):
    from hayday.resources import ResourceWorker
    path = tmp_path/'animal_care.json'
    key = request(path, frame=Screenshot((FIXTURES/'care_saved_sheep.png').read_bytes(),
                  1920, 1080, 'original'), harvest_operation=None,
                  polygon=((1252, 607), (1023, 683), (895, 609), (1079, 519)))
    worker = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json', vision=Mock())
    worker._menu_open = True
    bridge = Screenshot((FIXTURES/'care_bridge_sheep.png').read_bytes(), 1920, 1080, 'clear')
    worker._observe = Mock(side_effect=[SimpleNamespace(popups=(object(),), empty_slots=()),
                                      SimpleNamespace(popups=(), empty_slots=())])
    worker._dismiss = Mock(return_value=bridge)
    result = worker._finish('waiting', 'Growing', Mock(), item='wool')
    saved = json.loads(path.read_text())['jobs'][key]
    assert result.status == 'waiting'
    assert saved['navigation_view']['captured_at'] == 'clear'
    distant = Screenshot((FIXTURES/'care_distant_board.png').read_bytes(), 1920, 1080, 'board')
    assert worker._animal_care._project(saved, distant) == (1961, 923)


def test_feed_mill_return_does_not_reload_the_active_care_job(tmp_path):
    from hayday.resources import ResourceWorker
    request(tmp_path/'animal_care.json', harvest_operation=None)
    owner = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json',
                           vision=Mock(), fruits=Mock())
    owner._animal_care = Mock()
    owner._observe = Mock(return_value=SimpleNamespace(popups=(), empty_slots=()))
    owner._animal_care.step.side_effect = lambda frame: owner._finish('queued', 'Feed queued', frame)
    assert owner.care_animals(Mock()).status == 'queued'
    owner._animal_care.remember_view.assert_not_called()
    # A resource visit after care can refresh the navigation evidence again.
    frame = Mock()
    owner._finish('waiting', 'Growing', frame)
    owner._animal_care.remember_view.assert_called_once_with(frame)


def ready_worker(tmp_path, stock=3, enabled=True):
    from dataclasses import replace

    from hayday.herd_vision import FeedMenu
    initial = Screenshot((FIXTURES/'sheep_pen_before_feed.png').read_bytes(), 1920, 1080, 'initial')
    request(tmp_path/'animal_care.json', frame=initial, harvest_operation=None)
    owner = Mock(state_path=tmp_path/'resources.json', serial='test',
                 cancel_event=threading.Event(), fruits=SimpleNamespace(state={'items': {}}))
    owner._capture.side_effect = [replace(initial, captured_at='before'), replace(initial, captured_at='fresh')]
    owner._tap.return_value = replace(initial, captured_at='opened')
    owner._observe.return_value = SimpleNamespace(popups=(), empty_slots=())
    owner._state = {'items': {}}
    worker = AnimalCareWorker(owner)
    worker._close_board = lambda f: f
    pen = (VisualTarget(1082, 513, 80, 66, .98), VisualTarget(1133, 605, 10, 10, .98),
           ((1242, 601), (1056, 674), (905, 600), (1062, 521)))
    worker._target_pens = Mock(return_value=(pen,))
    worker.vision = Mock()
    worker.vision.menu.return_value = FeedMenu('sheep', VisualTarget(880, 220, 107, 79, .99),
                                             stock, enabled, b'feed-icon', (780, 173, 119, 73))
    worker.vision.confirm_pen.return_value = pen[2]
    worker._dismiss = Mock()
    worker._stock_receipts = Mock()
    worker._feed = Mock(return_value=ResourceResult('waiting', 'fed'))
    worker._supply = Mock(return_value=ResourceResult('queued', 'feed queued'))
    return worker, initial


def test_accidental_producer_menu_is_closed_before_feeding(tmp_path):
    worker, frame = ready_worker(tmp_path)
    picker = SimpleNamespace(popups=(), empty_slots=(VisualTarget(900, 600, 100, 80, .99),))
    worker.owner._observe.side_effect = [picker, SimpleNamespace(popups=(), empty_slots=())]
    worker._dismiss.return_value = frame
    actions = Mock()
    actions.attach_mock(worker._dismiss, 'dismiss')
    actions.attach_mock(worker._target_pens, 'find_pen')
    actions.attach_mock(worker._feed, 'feed')
    worker.step(frame)
    assert [call[0] for call in actions.mock_calls] == ['dismiss', 'find_pen', 'find_pen', 'feed']
    worker.owner.client.swipe.assert_not_called()


def test_persistent_producer_menu_cannot_authorize_a_camera_or_feed_gesture(tmp_path):
    worker, frame = ready_worker(tmp_path)
    worker.owner._observe.return_value = SimpleNamespace(popups=(), empty_slots=(object(),))
    worker._dismiss.return_value = frame
    result = worker.step(frame)
    assert result.status == 'waiting'
    assert worker._dismiss.call_count == 3
    worker._target_pens.assert_not_called()
    worker.owner.client.swipe.assert_not_called()
    worker.owner.client.drag_path.assert_not_called()
    worker._feed.assert_not_called()
    worker._supply.assert_not_called()
    from hayday.work_scheduling import utc_timestamp
    job = next(iter(worker.state['jobs'].values()))
    assert job['stage'] == 'needed' and job['next_due'] > utc_timestamp()+590


def test_disabled_feed_does_not_use_feed_or_start_a_batch(tmp_path):
    worker, frame = ready_worker(tmp_path, stock=0, enabled=False)
    worker.step(frame)
    assert next(iter(worker.state['jobs'].values()))['stage'] == 'complete'
    worker._feed.assert_not_called()
    worker._supply.assert_not_called()


def test_disabled_feed_needs_no_visible_fence_after_opening_the_verified_pen(tmp_path):
    worker, frame = ready_worker(tmp_path, stock=1, enabled=False)
    worker.vision.confirm_pen.return_value = None  # A growing-animal timer covers the fence.
    result = worker.step(frame)
    assert result.status == 'waiting' and 'already fed' in result.message
    assert next(iter(worker.state['jobs'].values()))['stage'] == 'complete'
    assert worker._target_pens.call_count == 2
    worker.vision.confirm_pen.assert_not_called()
    worker._stock_receipts.assert_called_once()
    worker._feed.assert_not_called()
    worker._supply.assert_not_called()


def test_unknown_stock_sends_no_feed_or_production_input(tmp_path):
    worker, frame = ready_worker(tmp_path, stock=None)
    with pytest.raises(ResourceChanged, match='unreadable'):
        worker.step(frame)
    worker._feed.assert_not_called()
    worker._supply.assert_not_called()


def test_empty_feed_visits_its_source_instead_of_dragging(tmp_path):
    worker, frame = ready_worker(tmp_path, stock=0)
    worker.step(frame)
    worker._feed.assert_not_called()
    worker._supply.assert_called_once()


def test_restart_reconciles_a_saved_sweep_before_any_new_input(tmp_path):
    worker, frame = ready_worker(tmp_path, stock=1, enabled=False)
    state = json.loads(worker.path.read_text())
    job_id = next(iter(state['jobs']))
    state['pending'] = {'job': job_id, 'animal': 'sheep', 'stock_before': 3, 'operation': 'saved-feed'}
    worker.path.write_text(json.dumps(state))
    result = worker.step(frame)
    assert result.details['feed_confirmed']
    assert worker.state['pending'] is None
    assert worker.state['jobs'][job_id]['stage'] == 'complete'
    worker._feed.assert_not_called()
    worker._supply.assert_not_called()


def test_pending_production_is_reconciled_before_collected_feed_is_consumed(tmp_path):
    worker, frame = ready_worker(tmp_path)
    state = json.loads(worker.path.read_text())
    next(iter(state['jobs'].values()))['feed_key'] = 'existing-feed'
    worker.path.write_text(json.dumps(state))
    worker.owner._state['items']['existing-feed'] = {'pending_batch': True, 'pending_stage': 'queued'}
    worker.step(frame)
    worker._feed.assert_not_called()
    worker._supply.assert_called_once()


def test_feed_input_error_preserves_intent_for_restart(tmp_path):
    from hayday.fruit import FruitWorker
    worker, frame = ready_worker(tmp_path)
    worker.state = json.loads(worker.path.read_text())
    job_id, job = next(iter(worker.state['jobs'].items()))
    worker.owner.fruits._same = FruitWorker._same
    worker.owner.client.drag_path.side_effect = RuntimeError('device disconnected')
    pen = worker._target_pens.return_value[0]
    menu = worker.vision.menu.return_value
    with pytest.raises(RuntimeError, match='disconnected'):
        AnimalCareWorker._feed(worker, job_id, job, frame, pen, frame, menu)
    saved = json.loads(worker.path.read_text())
    assert saved['pending']['job'] == job_id
    assert saved['pending']['stock_before'] == 3
    worker.owner.client.drag_path.assert_called_once()


def test_feed_control_change_cannot_authorize_a_sweep(tmp_path):
    from dataclasses import replace

    from hayday.fruit import FruitWorker
    worker, frame = ready_worker(tmp_path)
    worker.state = json.loads(worker.path.read_text())
    job_id, job = next(iter(worker.state['jobs'].items()))
    worker.owner.fruits._same = FruitWorker._same
    pen = worker._target_pens.return_value[0]
    menu = worker.vision.menu.return_value
    worker.vision.menu.return_value = replace(menu, enabled=False)
    with pytest.raises(ResourceChanged, match='changed before feeding'):
        AnimalCareWorker._feed(worker, job_id, job, frame, pen, frame, menu)
    worker.owner.client.drag_path.assert_not_called()
    assert worker.state['pending'] is None

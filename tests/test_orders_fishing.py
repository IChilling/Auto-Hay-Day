"""MuMu fishing observations, bounded input, and durable stock reconciliation."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.fishing import FishingWorker
from hayday.fishing_vision import FishingVision, LureMenu, LureWorkbench
from hayday.orders import OrderRunner
from hayday.resource_vision import VisualTarget
from hayday.resources import ResourceWorker
from hayday.work_scheduling import retry_delay

FIXTURES = Path(__file__).parent/'fixtures/orders'


@pytest.fixture(scope='module')
def vision():
    return FishingVision()


@pytest.mark.parametrize('name,x', [('fishing_menu.png',770), ('fishing_lures.png',831),
                                  ('fishing_lure_stock_one_fresh.png',831)])
def test_recorded_lure_menu_has_stock_and_home(vision, name, x):
    png = (FIXTURES/name).read_bytes()
    menu = vision.menu(png)
    assert menu is not None and menu.stock == 1
    assert abs(menu.tool[0]-x) < 3 and abs(menu.tool[1]-455) < 3
    assert vision.home(png) is not None


def test_scaled_menu_uses_observed_controls(vision):
    image = cv2.imread(str(FIXTURES/'fishing_menu.png'))
    image = cv2.resize(image, (1280,720), interpolation=cv2.INTER_AREA)
    png = cv2.imencode('.png', image)[1].tobytes()
    menu = vision.menu(png)
    assert menu is not None and menu.stock in (None,1)
    assert np.linalg.norm(np.array(menu.tool)-np.array([770,455])*2/3) < 5
    assert vision.home(png) is not None


def test_farm_and_wrong_goods_never_authorize_fishing(vision):
    assert vision.menu((FIXTURES/'cherry_small.png').read_bytes()) is None
    assert vision.home((FIXTURES/'cherry_small.png').read_bytes()) is None
    assert vision.supports(vision.item, lambda: False)
    cherry = Path(__file__).resolve().parents[1]/'images/fruit/cherry_item.png'
    assert not vision.supports(cherry.read_bytes(), lambda: False)


def test_workbench_is_not_a_casting_menu(vision):
    assert vision.menu((FIXTURES/'fishing_workbench.png').read_bytes()) is None


def test_fishing_cooldown_is_recognized_without_speedup(vision):
    assert vision.cooldown((FIXTURES/'fishing_cooldown.png').read_bytes())
    assert not vision.cooldown((FIXTURES/'cherry_small.png').read_bytes())


def test_recorded_empty_red_lure_is_zero_not_unknown(vision):
    assert vision.menu((FIXTURES/'fishing_empty_lures.png').read_bytes()).stock == 0


@pytest.mark.parametrize('name,empty,queued,recipe', [
    ('fishing_workbench_empty.png',2,False,False),
    ('fishing_free_lure_recipe.png',2,False,True),
    ('fishing_lure_queued.png',1,True,False),
    ('fishing_lure_queued_wide.png',1,True,False),
])
def test_recorded_workbench_excludes_premium_slot_and_verifies_queue(vision,name,empty,queued,recipe):
    png = (FIXTURES/name).read_bytes()
    bench = vision.workbench(png)
    assert bench and bench.stock == 0 and len(bench.empty_slots) == empty
    assert all(slot.center[0] < 1400 for slot in bench.empty_slots)
    assert vision.queued_red_lure(png,bench) is queued
    assert vision.first_slot_empty(bench) is (not queued)
    assert vision.free_red_recipe(png,bench) is recipe
    assert vision.menu(png) is None


def test_lure_location_requires_the_red_lure_title_and_navigation(vision):
    popup = vision.lure_location((FIXTURES/'fishing_lure_location.png').read_bytes())
    assert popup and popup.navigation[0].center == (1346,541)
    for name in ('fishing_menu.png','fishing_free_lure_recipe.png','fishing_cooldown.png'):
        assert vision.lure_location((FIXTURES/name).read_bytes()) is None


def test_workbench_anchor_is_separate_from_lure_and_premium_controls(vision):
    anchor = vision.workbench_anchor((FIXTURES/'fishing_workbench_empty.png').read_bytes())
    assert anchor and np.linalg.norm(np.array(anchor.center)-(1106,480)) < 4
    assert vision.workbench_anchor((FIXTURES/'fishing_menu.png').read_bytes()) is None


@pytest.mark.parametrize('name,center',[
    ('fishing_workbench_closed_wide.png',(1157,287)),
    ('fishing_workbench_closed_shifted.png',(1053,313)),
])
def test_closed_workbench_is_recognized_without_the_overlapping_menu_arrow(vision,name,center):
    png = (FIXTURES/name).read_bytes()
    anchor = vision.workbench_anchor(png)
    assert anchor and np.linalg.norm(np.subtract(anchor.center,center)) < 8
    assert vision.workbench(png) is None


def test_feedback_capture_skips_tapping_recovery_and_preserves_session_checks(tmp_path):
    frame = Screenshot(b'frame',1920,1080,'fresh')
    client = SimpleNamespace(serial='mumu-test',capture_fast=lambda:frame,
        capture=lambda:pytest.fail('Slow capture was used'))
    runner = OrderRunner(client,tmp_path)
    runner._deadline = time.monotonic()+60
    runner._recover_capture = lambda *a:pytest.fail('Recovery ran during a held contact')
    assert runner._capture_fishing_feedback() is frame and runner.frame is frame
    runner._size = (1280,720)
    with pytest.raises(RuntimeError,match='resolution'):
        runner._capture_fishing_feedback()


@pytest.mark.parametrize('name,point,expected', [
    ('fishing_hooked.png',(1081,812),(1002,474)),
    ('fishing_near_catch.png',(1192,960),(1020,541)),
])
def test_line_tracking_bridges_ripples(vision, name, point, expected):
    end = vision.line_end((FIXTURES/name).read_bytes(), point)
    assert end is not None and np.linalg.norm(np.array(end)-expected) < 10


def test_bent_line_reaches_the_visible_bobber_instead_of_signaling_a_bite(vision):
    png = (FIXTURES/'fishing_bent_line_bobber.png').read_bytes()
    end = vision.line_end(png, (958,750))
    assert end and np.linalg.norm(np.subtract(end, (1033,566))) < 10
    assert vision.bobber(png, end)


def test_ring_center_is_local_and_does_not_follow_other_water_ripples(vision):
    png = (FIXTURES/'fishing_near_catch.png').read_bytes()
    assert np.linalg.norm(np.array(vision.ring_center(png,(1012,589)))-(1019,543)) < 5
    assert vision.ring_center(png,(450,400)) is None


def worker_fixture(tmp_path, monkeypatch, *, stock=1, failure=False):
    cancel = threading.Event()
    monkeypatch.setattr(cancel, 'wait', lambda seconds: cancel.is_set())
    path = tmp_path/'fishing.json'
    calls = []
    client = SimpleNamespace(serial='mumu-test', farm=False)

    def drag(*args, **kwargs):
        entry = json.loads(path.read_text())['items']['fish']
        assert entry['stage'] == 'attempted' and entry['inventory_before']['available'] == 0
        assert kwargs['cancel_event'] is cancel and kwargs['max_seconds'] <= 60
        calls.append('cast')
        if failure:
            raise RuntimeError('connection lost')

    def tap(*args, **kwargs):
        if args == home.center:
            calls.append('home')
            client.farm = True
        else:
            calls.append('lure')

    client.drag_feedback, client.tap = drag, tap
    home = VisualTarget(20,900,150,140,.99)
    vision = SimpleNamespace(supports=lambda *a: True,
        menu=lambda *a: LureMenu((770,455),stock,1.),
        lure_location=lambda *a: None,
        home=lambda *a: None if client.farm else home)
    frame = Screenshot(b'lures',1920,1080,'frame1')
    worker = FishingWorker(client, lambda: frame, cancel, lambda text: None, path, vision=vision)
    return worker, frame, calls


def test_cast_intent_survives_restart_and_requires_two_stock_gains(tmp_path, monkeypatch):
    worker, frame, calls = worker_fixture(tmp_path, monkeypatch)
    result = worker.work_if_recognized(frame,b'fish','fish',
                                      baseline={'status':'missing','available':0,'required':1})
    assert result.details['pending_harvest'] and calls == ['cast','home']
    assert retry_delay(result) is None
    restored = FishingWorker(worker.client,worker.capture,worker.cancel_event,
                              worker.progress,worker.state_path,vision=worker.vision)
    restored.work_if_recognized(frame,b'fish','fish')
    restored.observe_inventory('fish','missing',0,1,'unchanged')
    restored.observe_inventory('fish','fulfilled',1,2,'wrong-requirement')
    assert restored.state['items']['fish']['stage'] == 'attempted'
    restored.observe_inventory('fish','fulfilled',1,1,'gain1')
    restored.observe_inventory('fish','fulfilled',1,1,'gain1')
    assert restored.state['items']['fish']['stage'] == 'attempted'
    restored.observe_inventory('fish','fulfilled',1,1,'gain2')
    assert restored.state['items']['fish']['stage'] == 'confirmed'
    assert calls == ['cast','home']


def test_interrupted_cast_is_not_replayed(tmp_path, monkeypatch):
    worker, frame, calls = worker_fixture(tmp_path, monkeypatch, failure=True)
    with pytest.raises(RuntimeError,match='connection lost'):
        worker.work_if_recognized(frame,b'fish','fish',
                                  baseline={'status':'missing','available':0,'required':1})
    restored = FishingWorker(worker.client,worker.capture,worker.cancel_event,
                              worker.progress,worker.state_path,vision=worker.vision)
    assert restored.work_if_recognized(frame,b'fish','fish').details['pending_harvest']
    assert calls == ['cast']
    for observation in ('first', 'second'):
        restored.observe_inventory('fish', 'missing', 0, 1, observation)
    assert not restored.can_check_returned_lure('fish')


@pytest.mark.parametrize('returned_stock', [0, 1, 2, None])
def test_completed_failed_cast_requires_unchanged_fish_and_returned_lure(tmp_path, monkeypatch, returned_stock):
    worker, frame, calls = worker_fixture(tmp_path, monkeypatch)
    worker.work_if_recognized(frame, b'fish', 'fish',
                             baseline={'status':'missing','available':0,'required':1})
    worker.observe_inventory('fish', 'missing', 0, 1, 'first')
    worker.observe_inventory('fish', 'missing', 0, 1, 'first')
    assert not worker.can_check_returned_lure('fish')
    worker.observe_inventory('fish', 'missing', 0, 1, 'second')
    assert worker.can_check_returned_lure('fish')
    worker.client.farm = False
    worker.vision.menu = lambda *args: LureMenu((770,455), returned_stock, 1.)
    result = worker.work_if_recognized(frame, b'fish', 'fish')
    assert calls.count('cast') == 1
    assert worker.state['items']['fish']['stage'] == ('no_effect' if returned_stock == 1 else 'attempted')
    assert bool(result.details.get('lure_returned')) is (returned_stock == 1)
    restored = FishingWorker(worker.client,worker.capture,worker.cancel_event,
                              worker.progress,worker.state_path,vision=worker.vision)
    assert restored.state['items']['fish']['stage'] == worker.state['items']['fish']['stage']


def test_changed_fish_inventory_invalidates_returned_lure_reconciliation(tmp_path, monkeypatch):
    worker, frame, _ = worker_fixture(tmp_path, monkeypatch)
    worker.work_if_recognized(frame, b'fish', 'fish',
                             baseline={'status':'missing','available':0,'required':1})
    for observation in ('first', 'second'):
        worker.observe_inventory('fish', 'missing', 0, 1, observation)
    assert worker.can_check_returned_lure('fish')
    worker.observe_inventory('fish', 'missing', 0, 2, 'different_order')
    assert not worker.can_check_returned_lure('fish')


def test_missing_bobber_without_a_visible_ring_does_not_start_fish_control(tmp_path, monkeypatch):
    worker, frame, _ = worker_fixture(tmp_path, monkeypatch)
    bobbers = iter((True, False))
    worker.vision.line_end = lambda *args: (850,500)
    worker.vision.bobber = lambda *args: next(bobbers)
    worker.vision.ring_center = lambda *args: None

    def drag(start, update, **kwargs):
        point = update(start, 1.)
        update(point, 5.)

    worker.client.drag_feedback = drag
    worker.work_if_recognized(frame, b'fish', 'fish',
                             baseline={'status':'missing','available':0,'required':1})
    operation = worker.state['items']['fish']['operation']
    evidence = worker.state_path.parent/'fishing_evidence'
    tracking = json.loads((evidence/f'{operation}_tracking.json').read_text())
    assert len(tracking) == 2 and all(sample['center'] is None for sample in tracking)
    assert not (evidence/f'{operation}_hooked.png').exists()


def test_visible_catch_ring_preserves_contact_through_short_line_gaps(tmp_path, monkeypatch):
    worker, frame, _ = worker_fixture(tmp_path, monkeypatch)
    endpoints = iter(((850,500), (850,500), None, None, (850,500)))
    bobbers = iter((True, False))
    worker.vision.line_end = lambda *args: next(endpoints)
    worker.vision.bobber = lambda *args: next(bobbers)
    worker.vision.ring_center = lambda *args: (850,522)

    def drag(start, update, **kwargs):
        point = start
        for elapsed in range(1, 6):
            point = update(point, float(elapsed))
            assert point is not None

    worker.client.drag_feedback = drag
    worker.work_if_recognized(frame, b'fish', 'fish',
                             baseline={'status':'missing','available':0,'required':1})
    assert worker.state['items']['fish']['stage'] == 'attempted'


def test_catch_photo_can_dismiss_before_verified_return_home(tmp_path,monkeypatch):
    worker, frame, calls = worker_fixture(tmp_path,monkeypatch,stock=0)
    def tap(*args, **kwargs):
        calls.append('home')
        worker.client.farm = len(calls) == 2
    worker.client.tap = tap
    result = worker._lure_wait('fish','No lures')
    assert result.details['needs_red_lure'] and calls == ['home','home']


@pytest.mark.parametrize('stock', [0,None])
def test_empty_or_unknown_lure_stock_never_casts(tmp_path, monkeypatch, stock):
    worker, frame, calls = worker_fixture(tmp_path,monkeypatch,stock=stock)
    result = worker.work_if_recognized(frame,b'fish','fish')
    assert 'cast' not in calls and not worker.state_path.exists()
    if stock == 0:
        assert calls == ['lure','home'] and retry_delay(result) == 600
    else:
        assert result.status == 'unsupported'


def test_resource_worker_reconciles_fishing_without_source_navigation(tmp_path, monkeypatch):
    worker, frame, calls = worker_fixture(tmp_path,monkeypatch)
    worker.work_if_recognized(frame,b'fish','fish',
                              baseline={'status':'missing','available':0,'required':1})
    resources = ResourceWorker(worker.client, capture=worker.capture,
                               state_path=tmp_path/'resources.json',fishing=worker)
    monkeypatch.setattr(resources,'_known_icon',lambda icon:'fish')
    monkeypatch.setattr('hayday.resources.OrderReader.item_icon',lambda *a:b'fish')
    item = SimpleNamespace(status='missing',available=0,required=1,quantity_bounds=(0,0,1,1))
    result = resources.pending_reconciliation(frame,item)
    assert result.details['inventory_only'] and result.details['unchanged_stock']
    assert result.details['item'] == 'fish'
    resources._acknowledge_inventory('fish','fulfilled',1,1,'gain1')
    resources._acknowledge_inventory('fish','fulfilled',1,1,'gain2')
    assert resources.pending_reconciliation(frame,item,item_key='fish').status == 'collected'


def test_resource_worker_allows_only_lure_inspection_after_two_unchanged_stock_reads(tmp_path, monkeypatch):
    worker, frame, calls = worker_fixture(tmp_path, monkeypatch)
    worker.work_if_recognized(frame, b'fish', 'fish',
                             baseline={'status':'missing','available':0,'required':1})
    resources = ResourceWorker(worker.client, capture=worker.capture,
                               state_path=tmp_path/'resources.json', fishing=worker)
    monkeypatch.setattr(resources, '_known_icon', lambda icon: 'fish')
    monkeypatch.setattr('hayday.resources.OrderReader.item_icon', lambda *args: b'fish')
    item = SimpleNamespace(status='missing', available=0, required=1, quantity_bounds=(0,0,1,1))
    for observation in ('first', 'second'):
        resources._acknowledge_inventory('fish', 'missing', 0, 1, observation)
    result = resources.pending_reconciliation(frame, item)
    assert result.details['lure_check_ready'] and result.details['pending_harvest']
    assert not result.details.get('inventory_only')
    assert worker.state['items']['fish']['stage'] == 'attempted'
    assert calls == ['cast','home']
    assert calls == ['cast','home']


def lure_worker(tmp_path, monkeypatch, *, failure=False, queue_visible=True):
    cancel = threading.Event()
    monkeypatch.setattr(cancel,'wait',lambda seconds: cancel.is_set())
    path = tmp_path/'fishing.json'
    client = SimpleNamespace(serial='mumu-test',scene='menu',inputs=[])
    red, arrow, ground, anchor_point = (920,480),(1346,541),(500,850),(1106,480)
    home = VisualTarget(20,900,150,140,.99)
    anchor = VisualTarget(1025,420,162,121,.99)
    title = VisualTarget(1243,380,404,63,.99)
    slots = (VisualTarget(1040,716,132,56,.99),VisualTarget(1259,698,121,52,.99))

    def tap(x,y,**kwargs):
        client.inputs.append(('tap',(x,y)))
        if (x,y) == home.center:
            client.scene = 'farm'
        elif (x,y) == red and client.scene == 'menu':
            client.scene = 'link'
        elif (x,y) == arrow and client.scene == 'link':
            client.scene = 'bench'
        elif (x,y) == red and client.scene == 'bench':
            client.scene = 'recipe'
        elif (x,y) == ground:
            client.scene = 'closed'
        elif (x,y) == anchor_point and client.scene == 'closed':
            client.scene = 'stocked'
        else:
            pytest.fail(f'Unexpected input in {client.scene}: {(x,y)}')

    def drag(points,**kwargs):
        assert points == (red,red,slots[0].center,slots[0].center)
        assert json.loads(path.read_text())['red_lure']['stage'] == 'attempted'
        client.inputs.append(('queue',points))
        if failure:
            raise RuntimeError('ADB disconnected')
        client.scene = 'queued' if queue_visible else 'bench'

    def bench(png,*args):
        scene = png.decode()
        if scene not in {'bench','recipe','queued','stocked'}:
            return None
        return LureWorkbench(red,1 if scene == 'stocked' else 0,1.,
                             slots[1:] if scene == 'queued' else slots,title)

    client.tap, client.drag_path = tap, drag
    vision = SimpleNamespace(
        supports=lambda *a:True, menu=lambda *a:LureMenu(red,0,1.),
        home=lambda *a:None if client.scene == 'farm' else home,
        lure_location=lambda png,*a:SimpleNamespace(navigation=(VisualTarget(1342,537,8,8,.99),)) if png == b'link' else None,
        workbench=bench, free_red_recipe=lambda png,*a:png == b'recipe',
        first_slot_empty=FishingVision.first_slot_empty,
        queued_red_lure=lambda png,*a:png == b'queued',
        workbench_anchor=lambda png,*a:anchor if png == b'closed' else None)
    sequence = 0

    def capture():
        nonlocal sequence
        sequence += 1
        return Screenshot(client.scene.encode(),1920,1080,str(sequence))

    monkeypatch.setattr('hayday.fishing_navigation.water_points',lambda *a, **kw:(ground,))
    monkeypatch.setattr('hayday.fishing_navigation.FishingNavigator._fresh_water',
                        lambda navigator, points, **kw:navigator.worker._frame())
    worker = FishingWorker(client,capture,cancel,lambda text:None,path,vision=vision)
    return worker


def test_empty_picker_navigates_and_queues_exactly_one_free_lure(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    result = worker.work_if_recognized(worker.capture(),b'fish','fish')
    assert result.status == 'queued' and retry_delay(result) == 600
    assert worker.state['red_lure']['stage'] == 'queued'
    assert worker.state['items'] == {}  # Lure work is not a fish harvest.
    assert sum(kind == 'queue' for kind,_ in worker.client.inputs) == 1
    assert worker.client.scene == 'farm'


def test_interrupted_lure_queue_is_reconciled_without_repeating(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch,failure=True)
    with pytest.raises(RuntimeError,match='disconnected'):
        worker.work_if_recognized(worker.capture(),b'fish','fish')
    assert json.loads(worker.state_path.read_text())['red_lure']['stage'] == 'attempted'
    restored = FishingWorker(worker.client,worker.capture,worker.cancel_event,
                              worker.progress,worker.state_path,vision=worker.vision)
    worker.client.scene = 'bench'
    frame = worker.capture()
    assert restored._queue_lure(frame,worker.vision.workbench(frame.png),'fish').status == 'unsupported'
    worker.client.scene = 'queued'
    frame = worker.capture()
    assert restored._queue_lure(frame,worker.vision.workbench(frame.png),'fish').status == 'queued'
    assert restored.state['red_lure']['stage'] == 'queued'
    assert sum(kind == 'queue' for kind,_ in worker.client.inputs) == 1


def test_missing_queue_artwork_does_not_confirm_production(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch,queue_visible=False)
    result = worker.work_if_recognized(worker.capture(),b'fish','fish')
    assert result.status == 'unsupported' and retry_delay(result) is None
    assert worker.state['red_lure']['stage'] == 'attempted'


def test_existing_red_lure_batch_is_adopted_without_another_drag(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    worker.client.scene = 'queued'
    frame = worker.capture()
    result = worker._queue_lure(frame,worker.vision.workbench(frame.png),'fish')
    assert result.status == 'queued' and worker.state['red_lure']['stage'] == 'queued'
    assert all(kind != 'queue' for kind,_ in worker.client.inputs)


def test_finished_lure_is_collected_then_confirmed_by_stock(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    worker._record_lure(stage='queued')
    worker.client.scene = 'bench'
    frame = worker.capture()
    result = worker._queue_lure(frame,worker.vision.workbench(frame.png),'fish')
    assert result.status == 'waiting' and retry_delay(result) == 1
    assert result.details['needs_red_lure'] is False
    assert worker.state['red_lure']['stage'] == 'stocked'
    assert worker.state['red_lure']['stock_after'] == 1
    assert all(kind != 'queue' for kind,_ in worker.client.inputs)


def test_unconfirmed_collection_preserves_batch_and_never_requeues(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    worker._record_lure(stage='queued')
    original = worker.vision.workbench
    worker.vision.workbench = lambda png,*a:original(b'bench' if png == b'stocked' else png)
    worker.client.scene = 'bench'
    frame = worker.capture()
    result = worker._queue_lure(frame,worker.vision.workbench(frame.png),'fish')
    assert retry_delay(result) == 600 and worker.state['red_lure']['stage'] == 'queued'
    assert all(kind != 'queue' for kind,_ in worker.client.inputs)


@pytest.mark.parametrize('missing_frames', [2, 100])
def test_collection_waits_for_swaying_artwork_with_a_bounded_retry(tmp_path, monkeypatch, missing_frames):
    worker = lure_worker(tmp_path, monkeypatch)
    worker._record_lure(stage='queued')
    worker.client.scene = 'bench'
    frame = worker.capture()
    bench = worker.vision.workbench(frame.png)
    original = worker.vision.workbench
    observations = 0

    def swaying(png, *args):
        nonlocal observations
        observations += 1
        return None if observations <= missing_frames else original(png, *args)

    worker.vision.workbench = swaying
    result = worker._collect_lure(frame, bench, 'fish')
    if missing_frames == 100:
        assert result.status == 'changed' and observations == 6
        assert worker.client.inputs == [] and worker.state['red_lure']['stage'] == 'queued'
    else:
        assert result.details['needs_red_lure'] is False
        assert worker.state['red_lure']['stage'] == 'stocked'
        assert all(kind != 'queue' for kind,_ in worker.client.inputs)


def test_recipe_failure_and_cancel_never_queue(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    worker.vision.free_red_recipe = lambda *a:False
    result = worker.work_if_recognized(worker.capture(),b'fish','fish')
    assert result.status == 'unsupported' and not worker.state_path.exists()
    worker.cancel_event.set()
    before = list(worker.client.inputs)
    with pytest.raises(RuntimeError,match='cancelled'):
        worker.work_if_recognized(worker.capture(),b'fish','fish')
    assert worker.client.inputs == before


def test_corrupt_lure_state_prevents_input(tmp_path,monkeypatch):
    worker = lure_worker(tmp_path,monkeypatch)
    worker.state['red_lure'] = {'stage':'unknown'}
    worker.state_path.write_text(json.dumps(worker.state))
    with pytest.raises(ValueError,match='invalid'):
        FishingWorker(worker.client,worker.capture,worker.cancel_event,worker.progress,
                       worker.state_path,vision=worker.vision)

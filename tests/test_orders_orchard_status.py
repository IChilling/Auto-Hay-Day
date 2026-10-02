"""An exhausted linked tree defers its requirement without stopping other orders."""

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.fruit import FruitVision, FruitWorker
from hayday.orchard_status import OrchardStatusVision
from hayday.orders import OrderRunner
from hayday.resources import ResourceResult, ResourceWorker
from hayday.work_scheduling import retry_delay

FIXTURES = Path(__file__).parent / 'fixtures/orders'


@pytest.fixture(scope='module')
def vision():
    return OrchardStatusVision()


@pytest.fixture(scope='module')
def apple():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'apple_juice_recipe.png').read_bytes(), np.uint8), 1)
    return cv2.imencode('.png', image[545:655, 1040:1140])[1].tobytes()


@pytest.mark.parametrize('name', ['apple_exhausted.png', 'apple_exhausted_earlier.png'])
def test_native_exhausted_apple_menu(vision, apple, name):
    result = vision.exhausted((FIXTURES / name).read_bytes(), apple)
    assert result is not None and result.species == 'apple'
    assert 450 < result.saw.center[0] < 550


@pytest.mark.parametrize('name', ['apple_juice_recipe.png', 'apple_tree_dismissed.png', 'cherry_small.png'])
def test_other_farm_states_do_not_count_as_exhaustion(vision, apple, name):
    assert vision.exhausted((FIXTURES / name).read_bytes(), apple) is None


def test_exhausted_saw_menu_does_not_claim_an_unsupported_fruit(vision):
    fruit = FruitVision()
    assert vision.exhausted((FIXTURES / 'apple_exhausted.png').read_bytes(),
                            fruit.raspberry_item) is None


def test_native_cherry_link_to_exhausted_tree(vision):
    result = vision.exhausted((FIXTURES / 'cherry_exhausted.png').read_bytes(), FruitVision().cherry_item)
    assert result is not None and result.species == 'cherry'
    assert 1150 < result.saw.center[0] < 1300


@pytest.mark.parametrize('name', ['saw', 'help', 'arrow'])
def test_each_independent_control_is_required(vision, apple, name):
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'apple_exhausted.png').read_bytes(), np.uint8), 1)
    left, top, right, bottom = vision.manifest['features'][name]['box']
    image[top:bottom, left:right] = 0
    assert vision.exhausted(cv2.imencode('.png', image)[1].tobytes(), apple) is None


def make_worker(tmp_path, vision, next_frame):
    # A client with no input methods also proves this path performs no tree action.
    client = SimpleNamespace(serial='test', capture=lambda: next_frame)
    reader = SimpleNamespace(basket=lambda *args, **kwargs: None,
                             exhausted_orchard=vision.exhausted)
    return FruitWorker(client, lambda: next_frame, threading.Event(), lambda message: None,
                       tmp_path / 'fruit.json', vision=reader)


def test_fresh_exhausted_menu_defers_and_survives_restart_without_harvest_intent(tmp_path, vision, apple):
    frame = Screenshot((FIXTURES / 'apple_exhausted.png').read_bytes(), 1920, 1080, 'first')
    fresh = Screenshot((FIXTURES / 'apple_exhausted_earlier.png').read_bytes(), 1920, 1080, 'second')
    worker = make_worker(tmp_path, vision, fresh)
    result = worker.work_if_recognized(frame, apple, 'apple')
    assert retry_delay(result) == 600 and result.details['exhausted_source']
    assert worker.state['items'] == {} and not worker.state_path.exists()
    resources = ResourceWorker(worker.client, state_path=tmp_path / 'resources.json')
    resources._remember_wait('apple', result)
    restored = ResourceWorker(worker.client, state_path=resources.state_path)
    assert 590 < restored._wait_remaining('apple') <= 600


@pytest.mark.parametrize('name,stamp', [('apple_tree_dismissed.png', 'second'), ('apple_exhausted.png', 'first')])
def test_changed_or_reused_frame_cannot_confirm_deferral(tmp_path, vision, apple, name, stamp):
    frame = Screenshot((FIXTURES / 'apple_exhausted.png').read_bytes(), 1920, 1080, 'first')
    fresh = Screenshot((FIXTURES / name).read_bytes(), 1920, 1080, stamp)
    worker = make_worker(tmp_path, vision, fresh)
    result = worker.work_if_recognized(frame, apple, 'apple')
    assert result.status == 'changed' and retry_delay(result) is None
    assert worker.state['items'] == {}


def test_pending_harvest_is_preserved_before_exhausted_source_check(tmp_path, vision, apple):
    frame = Screenshot((FIXTURES / 'apple_exhausted.png').read_bytes(), 1920, 1080, 'first')
    worker = make_worker(tmp_path, vision, frame)
    worker._record('apple', stage='attempted', operation='unconfirmed')
    result = worker.work_if_recognized(frame, apple, 'apple')
    assert result.details['pending_harvest'] and retry_delay(result) is None
    assert worker.state['items']['apple']['operation'] == 'unconfirmed'


def test_exhausted_dependency_parks_parent_ticket_and_advances(tmp_path):
    result = ResourceResult('waiting', 'The linked apple tree is exhausted.',
        {'item': 'apple', 'exhausted_source': True, 'defer_item': True, 'retry_after_seconds': 600})
    worker = SimpleNamespace(work=lambda frame, item: result)
    runner = OrderRunner(SimpleNamespace(serial='test'), tmp_path, worker=worker)
    item = SimpleNamespace(index=0, fingerprint='1'*16, available=0, required=1,
                           status='missing', red_quantity=True)
    ticket = SimpleNamespace(slot_id=5, fingerprint='2'*16)
    next_ticket = SimpleNamespace(slot_id=6, fingerprint='3'*16)
    selected = SimpleNamespace(items=(item,), complete=True, obstructed=False, fingerprint='4'*16)
    assert runner._resource(ticket, SimpleNamespace(selected=selected)) == 'parked'
    assert runner._state['active'] is None
    parked = runner._state['parked_orders'][0]
    assert parked['slot_id'] == 5 and parked['last_resource']['details']['item'] == 'apple'
    assert runner._next_work_ticket([ticket, next_ticket]) is next_ticket
    assert runner._item_waiting(parked, item)
    # A real stock change allows this requirement to be considered again.
    item.available = 1
    assert not runner._item_waiting(parked, item)

"""Silo recovery drains surplus, including partial stacks, across restarts."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingRunner
from hayday.wheating_restart import WheatRestart
from hayday.wheating_shop import WheatShop
from hayday.wheating_silo import WheatSiloFull
from hayday.wheating_vision import SaleSlot, ShopView, WheatingVision


@pytest.mark.parametrize('released', [0, 20, 50, 100])
def test_recovery_continues_past_old_limit_until_surplus_is_empty(tmp_path, released):
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    frame = Screenshot(b'farm', 1920, 1080, 'fresh')
    run._capture_raw = Mock(return_value=frame)
    run.open_shop = Mock()
    run.wait = Mock()
    run.state['silo_recovery'] = dict(version=1, released=released, reserve=40,
                                    harvests={'crop': 'operation'})
    worker = SimpleNamespace(state={'items': {'crop': {'stage': 'harvest_attempted',
        'operation': 'operation'}}}, state_path=tmp_path/'fields.json')
    fields = SimpleNamespace(worker=worker, _known_points=[(700, 500)]*51, next_harvest=10)
    recovery = WheatSiloFull(run)
    recovery.observe = Mock(return_value=None)
    shop = SimpleNamespace(observe=Mock(return_value=(frame, ShopView('overview'))),
                           close_to_farm=Mock())
    amounts = iter([10]*12+[7])
    def service(**kwargs):
        assert 'listing_limit' not in kwargs
        assert run.state['seed_reserve'] == 51
        assert 'crop' in worker.state['items']
        amount = next(amounts)
        run.state['silo_recovery']['released'] += amount
        run.state['silo_recovery']['surplus_empty'] = amount == 7
        return False
    shop.service = Mock(side_effect=service)
    assert recovery.recover(fields, shop)
    assert shop.service.call_count == 13
    assert fields.next_harvest == 0 and not worker.state['items']
    assert 'silo_recovery' not in run.state


def test_final_partial_stack_keeps_reserve_and_finishes_without_waiting_for_buyer(tmp_path):
    frame = Screenshot(b'shop', 1920, 1080, 'fresh')
    controls = {name: VisualTarget(600+i*25, 600, 10, 10, 1.) for i, name in
                enumerate(('close', 'wheat', 'minus_quantity', 'max_price', 'submit'))}
    state = dict(quantity=10, stock=58, price=36)
    actions = []
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    run.state.update(seed_reserve=51, silo_recovery={'released': 120})
    run.device_root.mkdir(parents=True, exist_ok=True)
    run.wait = Mock()
    run._captured = time.monotonic()
    run.frame = frame
    def view():
        return ShopView('composer', **controls, **state)
    def tap(x, y, **kwargs):
        name = next(n for n, t in controls.items() if t.center == (x, y))
        actions.append(name)
        if name == 'minus_quantity':
            state['quantity'] -= 1
            state['price'] = state['quantity']*36//10
        if name == 'submit':
            assert state['quantity'] == 7 and state['price'] == 25
            assert state['stock']-state['quantity'] == 51
    run.client.tap = tap
    shop = WheatShop(run)
    shop._open_slot = Mock(return_value=(frame, view()))
    shop._select_wheat = lambda *a, **kw: (frame, view())
    shop.current = lambda *a: (frame, view())
    shop.observe = lambda: (frame, view())
    shop._await_slot = Mock()
    slot = SaleSlot('empty', VisualTarget(400, 300, 200, 200, 1))
    assert shop._list(frame, ShopView('overview', slots=(slot,)), slot)
    assert actions.count('minus_quantity') == 3 and actions.count('submit') == 1
    assert run.state['silo_recovery'] == {'released': 127, 'surplus_empty': True}
    assert json.loads(run.state_path.read_text())['silo_recovery']['surplus_empty']
    shop.observe = Mock(side_effect=AssertionError('Do not wait for a buyer or open another composer'))
    assert shop.service()


def test_quantity_minus_is_recognized_in_real_composer():
    root = Path(__file__).resolve().parents[1]
    frame = Screenshot((root/'images/wheating/live_wheat_10_36.png').read_bytes(), 1920, 1080, 'fresh')
    view = WheatingVision().shop(frame)
    assert view.minus_quantity and view.minus_quantity.center == (1194, 302)


def test_restart_confirms_final_surplus_sale_from_saved_stock(tmp_path):
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path)
    run.state.update(seed_reserve=51, silo_recovery={'released': 120},
        pending={'operation': 'list', 'slot': [400, 300, 200, 200], 'quantity': 7, 'stock': 58})
    run.open_shop = Mock()
    run.wait = Mock()
    slot = SaleSlot('wheat', VisualTarget(400, 300, 200, 200, 1))
    shop = WheatShop(run)
    shop.observe = Mock(return_value=(Screenshot(b'shop', 1920, 1080, 'fresh'),
                                     ShopView('overview', slots=(slot,))))
    WheatRestart(run).resume_pending(shop)
    assert run.state['silo_recovery'] == {'released': 127, 'surplus_empty': True}
    assert run.state['pending'] is None

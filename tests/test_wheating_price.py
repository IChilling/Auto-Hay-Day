"""Price selection persists and only verified prices may be submitted."""
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.services import Services
from hayday.storage import AppData, Settings
from hayday.wheating import WheatingBlocked, WheatingRunner
from hayday.wheating_shop import WheatShop
from hayday.wheating_vision import SaleSlot, ShopView, WheatingVision


def sale(tmp_path, mode, stock=61, *, minimum=True, missed=False):
    frame = Screenshot(b'shop', 1920, 1080, 'fresh')
    names = ['wheat', 'close', 'minus_quantity', 'plus_quantity', 'max_price',
             'minus_price', 'plus_price', 'submit']
    if minimum:
        names.append('min_price')
    controls = {name: VisualTarget(600+i*25, 600, 10, 10, 1.)
                for i, name in enumerate(names)}
    state = dict(quantity=10, stock=stock, price=36)
    actions = []
    run = WheatingRunner(SimpleNamespace(serial='test'), tmp_path, sell_price=mode)
    run.state.update(seed_reserve=51, silo_recovery={'released': 0})
    run.device_root.mkdir(parents=True)
    run.frame, run._captured = frame, time.monotonic()
    run.wait = Mock()
    def view():
        return ShopView('composer', **controls, **state)

    def tap(x, y, **kw):
        name = next(n for n, t in controls.items() if t.center == (x, y))
        actions.append(name)
        if name in ('minus_quantity', 'plus_quantity'):
            state['quantity'] += 1 if name == 'plus_quantity' else -1
            state['price'] = state['quantity']*36//10
        elif name == 'max_price':
            state['price'] = state['quantity']*36//10
        elif name == 'min_price' and not missed:
            state['price'] = 1
        elif name == 'minus_price':
            state['price'] -= 1
        elif name == 'submit':
            assert state['price'] == (1 if mode == 'min' else state['quantity']*36//10)
            assert state['stock']-state['quantity'] >= 51

    run.client.tap = tap
    shop = WheatShop(run)
    shop._open_slot = Mock(side_effect=lambda *a: (frame, view()))
    shop._select_wheat = lambda *a, **kw: (frame, view())
    shop.observe = lambda: (frame, view())
    shop.current = lambda *a: (frame, view())
    shop._await_slot = Mock()
    slot = SaleSlot('empty', VisualTarget(400, 300, 200, 200, 1.))
    return run, shop, frame, slot, actions, state


@pytest.mark.parametrize('mode', ['max', 'min'])
@pytest.mark.parametrize('stock', [52, 58, 61])
def test_full_and_partial_sales_preserve_reserve_and_selected_price(tmp_path, mode, stock):
    run, shop, frame, slot, actions, state = sale(tmp_path, mode, stock)
    assert shop._list(frame, ShopView('overview', slots=(slot,)), slot)
    quantity = min(10, stock-51)
    assert run.listed == quantity
    assert state['price'] == (1 if mode == 'min' else quantity*36//10)
    assert actions.count('submit') == 1
    assert ('max_price' if mode == 'min' else 'min_price') not in actions
    assert run.state['pending'] is None


def test_minimum_falls_back_to_verified_minus_control(tmp_path):
    run, shop, frame, slot, actions, _ = sale(tmp_path, 'min', minimum=False)
    assert shop._list(frame, ShopView('overview'), slot)
    assert actions.count('minus_price') == 35
    assert run.listed == 10


def test_missed_minimum_tap_never_submits_wrong_price(tmp_path):
    run, shop, frame, slot, actions, _ = sale(tmp_path, 'min', missed=True)
    with pytest.raises(WheatingBlocked, match='readable and stable'):
        shop._list(frame, ShopView('overview'), slot)
    assert 'submit' not in actions and run.listed == 0


def test_missing_price_controls_never_submit(tmp_path):
    run, shop, frame, slot, actions, _ = sale(tmp_path, 'min', minimum=False)
    original = shop._select_wheat
    shop._select_wheat = lambda *a, **kw: (frame, replace(original()[1], minus_price=None))
    with pytest.raises(WheatingBlocked, match='changed before adjusting'):
        shop._list(frame, ShopView('overview'), slot)
    assert 'submit' not in actions and run.listed == 0


def test_final_fresh_price_mismatch_never_submits(tmp_path):
    run, shop, frame, slot, actions, state = sale(tmp_path, 'min')
    state['price'] = 1
    original = shop.current
    shop.current = lambda *a: (frame, replace(original()[1], price=36))
    with pytest.raises(WheatingBlocked, match='selected sell price'):
        shop._list(frame, ShopView('overview'), slot)
    assert 'submit' not in actions and run.listed == 0


def test_real_composer_recognizes_minimum_and_minus_price():
    path = Path(__file__).resolve().parents[1]/'images/wheating/live_wheat_10_36.png'
    view = WheatingVision().shop(Screenshot(path.read_bytes(), 1920, 1080, 'fixture'))
    assert view.min_price.center == (1287, 577)
    assert view.minus_price.center == (1194, 472)


def test_price_setting_roundtrip_and_old_settings_default(tmp_path):
    data = AppData(tmp_path)
    try:
        data.settings_path.write_text(json.dumps({'farm_name': 'Old farm'}))
        assert data.load_settings().wheating_sell_price == 'max'
        data.save_settings(Settings(wheating_sell_price='min'))
        assert data.load_settings().wheating_sell_price == 'min'
    finally:
        data.close()


@pytest.mark.parametrize('value', ['middle', '', None, [], 1])
def test_invalid_price_setting_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='max or min'):
        Settings(wheating_sell_price=value)
    with pytest.raises(ValueError, match='max or min'):
        WheatingRunner(SimpleNamespace(serial='test'), tmp_path, sell_price=value)


def test_price_change_is_saved_and_blocked_during_a_run(tmp_path):
    services = Services(tmp_path)
    try:
        services.select_wheating_sell_price('min')
        assert services.data.load_settings().wheating_sell_price == 'min'
        services._wheating_runner = object()
        with pytest.raises(Exception, match='Stop Wheating'):
            services.select_wheating_sell_price('max')
        assert services.settings.wheating_sell_price == 'min'
        services._wheating_runner = None
    finally:
        services.close()

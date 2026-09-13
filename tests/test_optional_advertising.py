"""An optional ad must not strand wheat sales behind a durable pending intent."""
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import AdbError, Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingBlocked, WheatingCancelled, WheatingRunner
from hayday.wheating_advertising import reconcile, save_trace
from hayday.wheating_restart import WheatRestart
from hayday.wheating_shop import WheatShop
from hayday.wheating_vision import SaleSlot, ShopView, WheatingVision


class AdGame:
    def __init__(self, tmp_path, outcome='miss'):
        self.kind, self.checked, self.cooldown = 'overview', False, None
        self.slot = SaleSlot('wheat', VisualTarget(260, 566, 227, 238, 1.))
        self.controls = {name: VisualTarget(700+i*100, 400, 50, 50, 1.)
                         for i, name in enumerate(('close', 'wheat', 'newspaper', 'submit'))}
        self.outcome, self.submits, self.frames, self.opens = outcome, 0, 0, 0
        self.client = SimpleNamespace(serial='test', capture=self.capture, tap=self.tap,
                                      force_stop_hay_day=Mock())
        self.run = WheatingRunner(self.client, tmp_path)
        self.run.device_root.mkdir(parents=True, exist_ok=True)
        self.run.vision = SimpleNamespace(shop=lambda frame: self.view(), farm=lambda frame: False)
        self.run.wait = lambda seconds: self.run.check()
        self.shop = WheatShop(self.run)

    def capture(self):
        self.frames += 1
        return Screenshot(b'fake', 1920, 1080, str(self.frames))

    def view(self):
        if self.kind == 'overview':
            return ShopView('overview', close=self.controls['close'], slots=(self.slot,))
        return ShopView(self.kind, **self.controls, ad_free=self.cooldown is None,
                        ad_checked=self.checked, cooldown=self.cooldown)

    def tap(self, x, y, **kwargs):
        if (x, y) == self.slot.target.center:
            self.kind = 'edit'
            self.opens += 1
        elif (x, y) == self.controls['close'].center:
            self.kind = 'overview'
        elif (x, y) == self.controls['newspaper'].center:
            self.checked = True
        elif (x, y) == self.controls['submit'].center:
            self.submits += 1
            assert self.run.state['pending']['operation'] == 'advertise'
            if self.outcome == 'marker':
                self.slot = replace(self.slot, advertised=True)
                self.kind = 'overview'
            elif self.outcome in {'sold', 'empty'}:
                self.slot = replace(self.slot, kind=self.outcome)
                self.kind = 'overview'
            elif self.outcome in {'cooldown', 'delivered_error'}:
                self.cooldown = 299
                if self.outcome == 'delivered_error':
                    raise AdbError('Transport returned an error after delivery')
            elif self.outcome == 'unmarked':
                self.kind = 'overview'

    def advertise(self):
        frame, view = self.shop.observe()
        self.shop._advertise(frame, view, self.slot)


@pytest.mark.parametrize('outcome,confirmed', [('miss', False), ('marker', True),
    ('sold', False), ('empty', False), ('cooldown', True), ('delivered_error', True), ('unmarked', False)])
def test_one_submission_and_nonblocking_outcome(tmp_path, outcome, confirmed):
    game = AdGame(tmp_path, outcome)
    game.advertise()
    assert game.submits == 1
    assert game.run.state['pending'] is None
    assert game.run.state['ad_after'] > time.time()+290
    assert game.run.advertisements == int(confirmed)
    assert game.kind == 'overview'
    game.client.force_stop_hay_day.assert_not_called()
    history = json.loads((game.run.device_root/'advertisements.json').read_text())
    assert len(history) == 1
    assert any(event['stage'] == 'before_tap' for event in history[0])


def test_no_ad_retry_during_retained_cooldown(tmp_path):
    game = AdGame(tmp_path)
    game.advertise()
    game.shop._stock_empty_until = float('inf')
    assert game.shop.service()
    assert game.submits == 1 and game.opens == 1


def test_stale_tap_guard_does_not_claim_ad_sent_or_restart(tmp_path):
    game = AdGame(tmp_path)
    actual = game.shop._tap
    def tap(point, frame, **kwargs):
        if point == game.controls['submit'].center:
            raise WheatingBlocked('The observed shop control became stale before input.')
        return actual(point, frame, **kwargs)
    game.shop._tap = tap
    game.advertise()
    assert game.submits == game.run.advertisements == 0
    assert game.run.state['pending'] is None
    history = json.loads((game.run.device_root/'advertisements.json').read_text())[-1]
    assert any(e['stage'] == 'error' and 'stale' in e['error'] for e in history)
    assert not any(e['stage'] == 'tap_returned' for e in history)


def test_saved_ad_recovers_locally_without_force_stop(tmp_path):
    game = AdGame(tmp_path)
    game.run._lock_file = Mock()
    game.run.state['pending'] = {'operation': 'advertise', 'at': time.time(), 'slot': list(game.slot.target.box)}
    game.run.state['field_ready_at'] = 0
    restart = WheatRestart(game.run)
    restart.recover(WheatingBlocked('Earlier shop action is still unconfirmed'), None)
    game.client.force_stop_hay_day.assert_not_called()
    assert restart.stage == 'advertisement_deferred_locally'
    assert game.run.state['pending'] is None
    assert game.submits == game.opens == game.run.advertisements == 0


def test_resume_pending_does_not_open_shop_again_or_submit(tmp_path):
    game = AdGame(tmp_path)
    game.run.state['pending'] = {'operation': 'advertise', 'at': time.time(), 'slot': list(game.slot.target.box)}
    game.run.state['field_ready_at'] = 0
    game.run.open_shop = Mock(side_effect=AssertionError('Already in shop'))
    WheatRestart(game.run).resume_pending(game.shop)
    assert game.run.state['pending'] is None
    assert game.submits == game.opens == 0


def test_saved_real_failure_is_recognized_and_retired_without_input(tmp_path):
    root = Path(__file__).parent/'fixtures'
    frame = Screenshot((root/'unconfirmed_ad_shop.png').read_bytes(), 1920, 1080, 'first')
    client = SimpleNamespace(serial='test', tap=Mock(), capture=Mock(return_value=frame))
    run = WheatingRunner(client, tmp_path)
    run.vision = WheatingVision()
    run.wait = Mock()
    run.state['pending'] = {'operation': 'advertise', 'at': time.time(), 'slot': [263,566,227,238]}
    run.state['field_ready_at'] = 0
    shop = WheatShop(run)
    assert run.vision.shop(frame).kind == 'overview'
    assert reconcile(shop)
    assert run.state['last_advertisement']['outcome'] == 'uncertain'
    assert run.state['pending'] is None
    assert run.advertisements == 0
    client.tap.assert_not_called()


def test_optional_recovery_preserves_sale_and_collection_intents(tmp_path):
    game = AdGame(tmp_path)
    for operation in ('list', 'collect'):
        pending = {'operation': operation, 'slot': list(game.slot.target.box)}
        game.run.state['pending'] = pending
        assert not reconcile(game.shop)
        assert game.run.state['pending'] is pending


def test_saved_ad_can_inspect_cooldown_without_resubmitting(tmp_path):
    game = AdGame(tmp_path)
    game.cooldown = 250
    game.run.state['pending'] = {'operation': 'advertise', 'at': time.time(), 'slot': list(game.slot.target.box)}
    assert reconcile(game.shop)
    assert game.opens == 1 and game.submits == 0
    assert game.run.advertisements == 1
    assert game.kind == 'overview' and game.run.state['pending'] is None


def test_unresolved_sale_still_requires_real_confirmation(tmp_path):
    game = AdGame(tmp_path)
    game.slot = replace(game.slot, kind='empty')
    game.run.open_shop = Mock()
    pending = {'operation': 'list', 'quantity': 10, 'slot': list(game.slot.target.box)}
    game.run.state['pending'] = pending
    with pytest.raises(WheatingBlocked, match='intent was preserved'):
        WheatRestart(game.run).resume_pending(game.shop)
    assert game.run.state['pending'] is pending
    assert game.run.listed == 0


def test_unknown_screen_preserves_pending_ad(tmp_path):
    game = AdGame(tmp_path)
    game.kind = 'unknown'
    game.run.state['pending'] = {'operation': 'advertise', 'at': time.time(), 'slot': list(game.slot.target.box)}
    assert not reconcile(game.shop)
    assert game.run.state['pending'] is not None


def test_cancellation_and_device_guards_remain_effective(tmp_path):
    game = AdGame(tmp_path)
    game.run.cancel_event.set()
    with pytest.raises(WheatingCancelled):
        game.advertise()
    game.run.cancel_event.clear()
    game.client.serial = 'other'
    with pytest.raises(WheatingBlocked):
        game.advertise()
    assert game.submits == 0


def test_ad_diagnostics_remain_bounded(tmp_path):
    game = AdGame(tmp_path)
    for _ in range(12):
        game.run.state['ad_after'] = 0
        game.advertise()
    path = game.run.device_root/'advertisements.json'
    records = json.loads(path.read_text())
    assert len(records) == 8 and all(len(events) <= 16 for events in records)
    assert len(list(game.run.device_root.glob('advertisement_*.png'))) == 2
    save_trace(game.shop)
    assert json.loads(path.read_text()) == records


@pytest.fixture
def ad_clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(time, 'time', lambda: now[0])
    return now


def test_full_silo_checks_live_free_ad_despite_stale_timer(tmp_path, ad_clock):
    game = AdGame(tmp_path, 'marker')
    game.run.state.update(silo_recovery={'released': 0}, ad_after=ad_clock[0]+240)
    assert game.shop.service()
    assert game.opens == game.submits == game.run.advertisements == 1
    assert game.kind == 'overview'


def test_full_silo_uses_observed_cooldown_and_checks_at_expiry(tmp_path, ad_clock):
    game = AdGame(tmp_path, 'marker')
    game.run.state.update(silo_recovery={'released': 0}, ad_after=ad_clock[0]+240)
    game.cooldown = 60
    assert game.shop.service()
    assert game.opens == 1 and game.submits == 0
    assert game.run.state['ad_after'] == ad_clock[0]+61
    ad_clock[0] += 60
    assert game.shop.service()
    assert game.opens == 1
    ad_clock[0] += 1
    game.cooldown = None
    assert game.shop.service()
    assert game.opens == 2 and game.submits == 1


def test_full_silo_retries_unclear_availability_without_reopening_each_loop(tmp_path, ad_clock):
    game = AdGame(tmp_path, 'marker')
    game.run.state.update(silo_recovery={'released': 0}, ad_after=ad_clock[0]+240)
    original = game.view
    game.view = lambda: replace(original(), ad_free=False)
    assert game.shop.service()
    assert game.opens == 1 and game.submits == 0
    # Recreating the shop must retain the backoff while recovery is in progress.
    game.run.state = json.loads(game.run.state_path.read_text())
    game.shop = WheatShop(game.run)
    ad_clock[0] += 29
    assert game.shop.service()
    assert game.opens == 1
    ad_clock[0] += 1
    game.view = original
    assert game.shop.service()
    assert game.opens == 2 and game.submits == 1


@pytest.mark.parametrize('outcome', ['miss', 'unmarked', 'cooldown'])
def test_full_silo_live_check_preserves_recent_submission_guard(tmp_path, ad_clock, outcome):
    game = AdGame(tmp_path, outcome)
    game.advertise()
    assert game.submits == 1
    previous_opens = game.opens
    # Even a free-looking panel cannot authorize replay of a recent attempt.
    game.cooldown = None
    game.run.state.update(silo_recovery={'released': 0}, ad_after=ad_clock[0]+240)
    game.shop = WheatShop(game.run)
    assert game.shop.service()
    assert game.opens == previous_opens+1 and game.submits == 1
    assert game.run.state['ad_after'] >= ad_clock[0]+301
    ad_clock[0] += 300
    assert game.shop.service()
    assert game.opens == previous_opens+1 and game.submits == 1
    ad_clock[0] += 1
    game.outcome = 'marker'
    assert game.shop.service()
    assert game.submits == 2


@pytest.mark.parametrize('operation', ['advertise', 'list', 'collect'])
def test_full_silo_ad_probe_does_not_touch_pending_transactions(tmp_path, ad_clock, operation):
    game = AdGame(tmp_path)
    pending = {'operation': operation, 'at': ad_clock[0], 'slot': list(game.slot.target.box)}
    game.run.state.update(silo_recovery={'released': 0}, ad_after=ad_clock[0]+240,
                          pending=pending)
    assert game.shop.service()
    assert game.opens == game.submits == 0
    assert game.run.state['pending'] == pending


def test_normal_shop_keeps_cached_ad_schedule(tmp_path, ad_clock):
    game = AdGame(tmp_path, 'marker')
    game.run.state['ad_after'] = ad_clock[0]+240
    assert game.shop.service()
    assert game.opens == game.submits == 0


def test_unclear_silo_probe_keeps_submission_guard_after_recovery_ends(tmp_path, ad_clock):
    game = AdGame(tmp_path)
    game.advertise()
    game.run.state['silo_recovery'] = {'released': 0}
    original = game.view
    game.view = lambda: replace(original(), ad_free=False)
    assert game.shop.service()
    assert game.run.state['ad_after'] >= ad_clock[0]+301
    game.run.state.pop('silo_recovery')
    game.view = original
    ad_clock[0] += 30
    assert game.shop.service()
    assert game.submits == 1

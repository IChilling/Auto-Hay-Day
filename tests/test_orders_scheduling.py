"""Slow animal care cannot starve later requirements on the same ticket."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from hayday.orders import OrderRunner, OrdersLimit
from hayday.resources import ResourceResult
from hayday.work_scheduling import utc_timestamp


def test_expired_production_wait_does_not_starve_other_missing_items():
    items = [SimpleNamespace(index=index) for index in (0, 2, 3)]
    active = {}
    visits = []
    # Every delay has elapsed between visits, e.g. during a Feed Mill trip.
    for _ in range(6):
        item = OrderRunner._next_resource_item(active, items)
        visits.append(item.index)
        active['last_item_index'] = item.index
        active['last_resource'] = {'details': {'pending_stage': 'queued'}}
        active = json.loads(json.dumps(active))  # Same choice after restart.
    assert visits == [0, 2, 3, 0, 2, 3]


def test_unconfirmed_harvest_stays_ahead_of_rotation():
    items = [SimpleNamespace(index=index) for index in (0, 2, 3)]
    active = {'last_item_index': 2, 'last_resource': {'details': {'pending_harvest': True}}}
    assert OrderRunner._next_resource_item(active, items).index == 2
    active['last_resource']['details'] = {}
    assert OrderRunner._next_resource_item(active, items).index == 3


def test_fulfilled_previous_requirement_does_not_block_remaining_items():
    items = [SimpleNamespace(index=index) for index in (0, 3)]
    active = {'last_item_index': 2, 'last_resource': {'details': {'pending_harvest': True}}}
    assert OrderRunner._next_resource_item(active, items).index == 3


def test_legacy_pending_harvest_is_pinned_by_saved_item_fingerprint():
    items = [SimpleNamespace(index=index, fingerprint=str(index)) for index in (0, 2, 3)]
    active = {'item_fingerprint': '2', 'last_resource': {'details': {'pending_harvest': True}}}
    assert OrderRunner._next_resource_item(active, items).index == 2


def test_long_navigation_does_not_erase_a_completed_pass_of_deferred_items():
    item = SimpleNamespace(index=2, fingerprint='0'*32, available=0, required=1)
    active = {'deferred_items': [{'index': 2, 'fingerprint': item.fingerprint,
                                 'available': 0, 'required': 1, 'retry_at': 1}]}
    assert not OrderRunner._item_waiting(active, item)
    assert OrderRunner._item_waiting(active, item, include_due=True)
    item.available = 1
    assert not OrderRunner._item_waiting(active, item, include_due=True)


def test_fulfilled_item_retry_cannot_starve_older_unfinished_orders_after_restart(tmp_path):
    runner = OrderRunner(Mock(serial='test'), tmp_path)
    first = SimpleNamespace(slot_id=0, fingerprint='0'*32)
    other = SimpleNamespace(slot_id=1, fingerprint='f'*32)
    now = utc_timestamp()
    def hold(index, retry_at):
        return {'index': index, 'fingerprint': '0'*32, 'available': 0,
                'required': 1, 'retry_at': retry_at, 'reason': 'Growing'}
    def parked(ticket, holds):
        return {'slot_id': ticket.slot_id, 'ticket_fingerprint': ticket.fingerprint,
                'order_fingerprint': '0'*32, 'actions': 0, 'deferred_items': holds}
    runner._state = {'active': None, 'parked_orders': [
        parked(first, [hold(0, now-1000), hold(1, now+600)]),
        parked(other, [hold(0, now-500)])]}
    runner._persist = Mock()
    runner._progress = Mock()
    assert runner._next_work_ticket([first, other]) is first
    items = [SimpleNamespace(index=i, fingerprint='0'*32, available=1-i, required=1,
                             status='fulfilled' if i == 0 else 'missing', red_quantity=i == 1)
             for i in range(2)]
    observation = SimpleNamespace(selected=SimpleNamespace(complete=True, obstructed=False,
        fingerprint='0'*32, items=items))
    assert runner._resource(first, observation) == 'parked'
    runner._state = json.loads(json.dumps(runner._state))
    assert runner._parked_order(first)['deferred_items'] == [hold(1, now+600)]
    assert runner._next_work_ticket([first, other]) is other


def test_changed_inventory_clears_its_old_hold_without_discarding_resource_intent(tmp_path):
    worker = SimpleNamespace(work=Mock(return_value=ResourceResult('waiting', 'Still growing',
        {'defer_item': True, 'retry_after_seconds': 600})))
    runner = OrderRunner(Mock(serial='test'), tmp_path, worker=worker)
    ticket = SimpleNamespace(slot_id=0, fingerprint='0'*32)
    intent = {'pending_harvest': True, 'item': 'cotton', 'operation': 'saved-gesture'}
    worker.pending_reconciliation = Mock(return_value=ResourceResult('waiting', 'Check inventory', intent))
    runner._state = {'active': {'slot_id': 0, 'ticket_fingerprint': ticket.fingerprint,
        'actions': 0, 'last_resource': {'details': intent}, 'deferred_items': [
            {'index': 0, 'fingerprint': '0'*32, 'available': 0, 'required': 3,
             'retry_at': utc_timestamp()+600}]}}
    item = SimpleNamespace(index=0, fingerprint='0'*32, available=2, required=3,
                           status='missing', red_quantity=True)
    runner._persist = Mock()
    runner._progress = Mock()
    runner._wait = Mock()
    runner._check = Mock()
    observation = SimpleNamespace(selected=SimpleNamespace(complete=True, obstructed=False,
        fingerprint='0'*32, items=[item]))
    runner._resource(ticket, observation)
    assert runner._state['active']['deferred_items'] == []
    worker.pending_reconciliation.assert_called_once_with(None, item, item_key='cotton')
    worker.work.assert_not_called()
    assert runner._state['active']['last_resource']['details'] == intent


def run_worker(tmp_path, ready=False):
    worker = Mock()
    runner = OrderRunner(Mock(serial='test'), tmp_path,
                         reader=SimpleNamespace(reference_error=''), worker=worker, max_deliveries=1)
    ticket = SimpleNamespace(slot_id=0, ready=ready, sent=False)
    board = SimpleNamespace(bonus_prompt=None, navigation=None, tickets=(ticket,))
    runner._panel = Mock(return_value=board)
    runner._select = Mock(return_value=(ticket, board))
    runner._guarded_ticket = lambda _: False
    runner._next_work_ticket = lambda _: ticket
    return runner, worker


def test_ready_truck_is_sent_before_searching_for_feeding_pens(tmp_path):
    runner, worker = run_worker(tmp_path, ready=True)
    runner._ready = lambda _: True
    def dispatch(*_):
        runner.deliveries += 1
        return True
    runner._dispatch = dispatch
    result = runner.run()
    assert result.deliveries == 1
    worker.care_animals.assert_not_called()


def test_resource_work_and_care_alternate_without_a_startup_search(tmp_path):
    runner, worker = run_worker(tmp_path)
    actions = []
    def resource(*_):
        actions.append('resource')
        if actions.count('resource') == 2:
            raise OrdersLimit()
        return 'waiting'
    def care(*_):
        actions.append('care')
        return ResourceResult('waiting', 'Feeding is saved for later.')
    runner._resource = resource
    worker.care_animals.side_effect = care
    assert runner.run().status == 'limit'
    assert actions == ['resource', 'care', 'resource']

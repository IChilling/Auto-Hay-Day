"""Stock seen before a queue vacancy survives until production reconciliation."""
from types import SimpleNamespace

import pytest

from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.resources import ResourceChanged, ResourceWorker


def worker_with_batch(tmp_path, stage='queued'):
    worker = ResourceWorker(SimpleNamespace(serial='test', capture=lambda: None),
                            state_path=tmp_path/'resources.json')
    worker._record('butter', 'test_batch', pending_batch=True, pending_stage=stage,
                   inventory_before={'available': 0, 'required': 1, 'status': 'missing'},
                   pending_slot={'dx': 300, 'dy': 200, 'menu_width': 100, 'label_width': 120})
    worker._finish = lambda status, message, frame, **details: details
    return worker


def test_stock_before_vacancy_survives_restart_but_cannot_retire_occupied_queue(tmp_path):
    worker = worker_with_batch(tmp_path)
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'first')
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'second')
    entry = worker._state['items']['butter']
    assert entry['pending_batch'] and entry['inventory_confirmation']['count'] == 2
    # The delivered order may consume the collected goods before this machine
    # is needed again. That later stock decrease cannot erase its receipt.
    worker._acknowledge_inventory('butter', 'missing', 0, 1, 'after_delivery')
    assert worker._state['items']['butter']['inventory_confirmation'] is None
    assert worker._state['items']['butter']['inventory_receipt']['count'] == 2
    worker = ResourceWorker(worker.client, state_path=worker.state_path)
    worker._finish = lambda status, message, frame, **details: details
    menu = VisualTarget(100, 100, 100, 100, .99)
    worker._pending_at_source(None, SimpleNamespace(empty_slots=()), menu, 'butter')
    assert worker._state['items']['butter']['pending_batch']
    slot = VisualTarget(390, 325, 120, 50, .99)
    worker._pending_at_source(None, SimpleNamespace(empty_slots=(slot,)), menu, 'butter')
    assert not worker._state['items']['butter']['pending_batch']


@pytest.mark.parametrize('stage', ['queued', 'awaiting_collection'])
def test_stock_confirmation_needs_distinct_consistent_readings(tmp_path, stage):
    worker = worker_with_batch(tmp_path, stage)
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'first')
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'first')
    assert worker._state['items']['butter']['inventory_confirmation']['count'] == 1
    worker._acknowledge_inventory('butter', 'missing', 0, 1, 'second')
    assert worker._state['items']['butter']['inventory_confirmation'] is None
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'third')
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'fourth')
    assert worker._state['items']['butter']['pending_batch'] is (stage == 'queued')


@pytest.mark.parametrize('stage', ['queued', 'uncertain'])
def test_other_requirement_or_uncertain_production_does_not_retire_a_batch(tmp_path, stage):
    worker = worker_with_batch(tmp_path, stage)
    for observation in ('first', 'second'):
        worker._acknowledge_inventory('butter', 'fulfilled', 1, 2, observation)
    assert worker._state['items']['butter'].get('inventory_confirmation') is None
    assert worker._state['items']['butter']['pending_batch']


@pytest.mark.parametrize('stage', ['queued', 'awaiting_collection'])
def test_parent_recipe_records_receipt_before_ingredient_stock_is_consumed(tmp_path, stage):
    worker = worker_with_batch(tmp_path, stage)
    row = SimpleNamespace(icon_png=b'butter', missing=False, quantity_box=(1,2,3,4))
    recipe = SimpleNamespace(rows=(row,), recipe_complete=True, title_png=b'cake')
    worker._known_icon = lambda icon: 'butter' if icon == b'butter' else None
    worker._quantity = lambda *args: (1, 1)
    worker.capture = lambda: Screenshot(b'fresh-recipe', 1920, 1080, 'second')
    worker.vision = SimpleNamespace(observe=lambda *args, **kwargs: SimpleNamespace(popups=(recipe,)),
                                    titles_match=lambda a, b: a == b)
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'first')
    assert worker._confirm_recipe_receipts(recipe) is None
    restored = ResourceWorker(worker.client, state_path=worker.state_path)
    restored._acknowledge_inventory('butter', 'missing', 0, 1, 'consumed')
    entry = restored._state['items']['butter']
    assert entry['pending_batch'] is (stage == 'queued')
    if stage == 'queued':
        assert entry['inventory_receipt']['count'] == 2


def test_changed_recipe_cannot_confirm_or_consume_pending_ingredient(tmp_path):
    worker = worker_with_batch(tmp_path, 'awaiting_collection')
    row = SimpleNamespace(icon_png=b'butter', missing=False, quantity_box=(1,2,3,4))
    recipe = SimpleNamespace(rows=(row,), recipe_complete=True, title_png=b'cake')
    worker._known_icon = lambda icon: 'butter'
    worker.capture = lambda: Screenshot(b'changed', 1920, 1080, 'second')
    worker.vision = SimpleNamespace(observe=lambda *args, **kwargs: SimpleNamespace(popups=()),
                                    titles_match=lambda a, b: a == b)
    worker._acknowledge_inventory('butter', 'fulfilled', 1, 1, 'first')
    with pytest.raises(ResourceChanged, match='recipe changed'):
        worker._confirm_recipe_receipts(recipe)
    assert worker._state['items']['butter']['pending_batch']
    assert worker._state['items']['butter']['inventory_confirmation']['count'] == 1

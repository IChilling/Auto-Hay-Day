"""Shared animal tools authorize bounded neighboring pens, never animal poses."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.animal_care import request_feeding
from hayday.animal_groups import adjacent_pens
from hayday.chickens import ChickenVision
from hayday.cows import CowVision
from hayday.fruit import FruitWorker
from hayday.herd_vision import HerdVision
from hayday.resource_vision import _decode, _png

FIXTURES = Path(__file__).parent/'fixtures/orders'


@pytest.fixture(scope='module')
def cow_group():
    vision = CowVision()
    png = (FIXTURES/'arranged_cow_menu.png').read_bytes()
    return vision.harvest(png, vision.item, lambda: False)


def test_global_bucket_supports_both_adjacent_cow_pens(cow_group):
    assert cow_group.target and len(cow_group.pen_polygons) == 2
    assert len(cow_group.sweep_paths) == 2
    for polygon, path in zip(cow_group.pen_polygons, cow_group.sweep_paths, strict=True):
        assert path and all(cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0
                            for point in path)


def test_losing_a_neighbor_invalidates_the_group_before_input(cow_group):
    hidden = replace(cow_group, pen_polygons=cow_group.pen_polygons[:1],
                     sweep_paths=cow_group.sweep_paths[:1])
    assert not FruitWorker._same_pen(cow_group, hidden)
    reordered = replace(cow_group, pen_polygons=cow_group.pen_polygons[::-1])
    assert FruitWorker._same_pen(cow_group, reordered)


def test_disconnected_same_species_pen_is_not_added(cow_group):
    pens = [(target, target, poly) for target, poly in
            zip(cow_group.fruit_features, cow_group.pen_polygons, strict=True)]
    # Width here is an enclosure landmark's width, not the tiny soil target.
    pens = [(replace(t, width=64), g, p) for t, g, p in pens]
    far = (pens[0][0], pens[0][1], tuple((x+650, y) for x, y in pens[0][2]))
    assert len(adjacent_pens((*pens, far), pens[0])) == 2


def test_enabled_egg_basket_uses_pens_without_ready_hen_templates(monkeypatch):
    vision = ChickenVision()
    search = vision.matcher._search

    def checked(image, reference, *args, **kwargs):
        assert all(reference is not vision.references[name] for name in ('ready', 'ready_facing'))
        return search(image, reference, *args, **kwargs)

    monkeypatch.setattr(vision.matcher, '_search', checked)
    png = (FIXTURES/'arranged_chicken_menu.png').read_bytes()
    group = vision.basket(png, vision.item, lambda: False)
    assert group.target and len(group.pen_polygons) == 2


def test_grey_egg_basket_blocks_collection_but_preserves_menu_identity():
    vision = HerdVision()
    png = (FIXTURES/'arranged_chicken_menu.png').read_bytes()
    menu = vision.menu(png, 'chicken', lambda: False)
    assert menu.collection_enabled is True
    image = _decode(png)
    x, y, w, h = menu.collection_tool.box
    region = image[y:y+h, x:x+w]
    region[:] = cv2.cvtColor(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    grey = _png(image)
    assert vision.chicken.basket(grey, vision.chicken.item, lambda: False) is None
    assert vision.menu(grey, 'chicken', lambda: False).collection_enabled is False


def test_live_group_collection_turns_the_shared_basket_grey():
    vision = HerdVision()
    before = (FIXTURES/'grouped_chicken_before.png').read_bytes()
    after = (FIXTURES/'grouped_chicken_after.png').read_bytes()
    assert vision.menu(before, 'chicken', lambda: False).collection_enabled is True
    assert vision.menu(after, 'chicken', lambda: False).collection_enabled is False
    assert vision.chicken.basket(after, vision.chicken.item, lambda: False) is None


def test_group_harvest_saves_separate_feed_jobs_under_one_receipt(tmp_path, cow_group):
    path = tmp_path/'animal_care.json'
    frame = Screenshot((FIXTURES/'arranged_cow_menu.png').read_bytes(), 1920, 1080, 'native')
    for index, polygon in enumerate(cow_group.pen_polygons):
        request_feeding(path, 'mumu-test', animal='cow', frame=frame, polygon=polygon,
                        item_key='milk', harvest_operation='one-harvest', pen_index=index)
    state = json.loads(path.read_text())
    assert len(state['jobs']) == 2
    assert {job['harvest_operation'] for job in state['jobs'].values()} == {'one-harvest'}
    assert {job['stage'] for job in state['jobs'].values()} == {'awaiting_harvest'}
    assert {job['animal'] for job in state['jobs'].values()} == {'cow'}


def test_three_pen_drag_is_bounded_and_persisted_before_input(tmp_path, monkeypatch, cow_group):
    third = tuple((x, y-210) for x, y in cow_group.pen_polygons[0])
    guide = replace(cow_group, pen_polygons=(*cow_group.pen_polygons, third),
                    sweep_paths=(*cow_group.sweep_paths, tuple((x, y-210) for x, y in cow_group.sweep_paths[0])))
    frame = Screenshot((FIXTURES/'arranged_cow_menu.png').read_bytes(), 1920, 1080, 'native')
    cancel = threading.Event()
    monkeypatch.setattr(cancel, 'wait', lambda _: False)
    gestures = []

    def drag(points, **options):
        state = json.loads((tmp_path/'fruit.json').read_text())
        jobs = json.loads((tmp_path/'animal_care.json').read_text())['jobs']
        assert state['items']['milk']['stage'] == 'attempted' and len(jobs) == 3
        assert 250 <= options['duration_ms'] <= 12000
        assert all(point in points for path in guide.sweep_paths for point in path)
        gestures.append(points)

    client = SimpleNamespace(serial='mumu-test', drag_path=drag)
    vision = SimpleNamespace(basket=lambda *args, **kwargs: guide)
    worker = FruitWorker(client, lambda: frame, cancel, lambda _: None, tmp_path/'fruit.json', vision=vision)
    worker.storage = SimpleNamespace(read=lambda *args: SimpleNamespace(full=False))
    baseline = {'status': 'missing', 'available': 0, 'required': 2}
    assert worker.work_if_recognized(frame, b'icon', 'milk', baseline).details['pending_harvest']
    restored = FruitWorker(client, lambda: frame, cancel, lambda _: None, tmp_path/'fruit.json', vision=vision)
    assert restored.work_if_recognized(frame, b'icon', 'milk', baseline).details['pending_harvest']
    assert len(gestures) == 1

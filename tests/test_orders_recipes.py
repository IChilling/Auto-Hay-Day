"""Resource headers retain all item-name lines without neighboring controls."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.quantities import Quantity, read_quantity
from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.resources import ResourceResult, ResourceWorker

FIXTURES = Path(__file__).parent/'fixtures/orders'
REFERENCES = Path(__file__).resolve().parents[1]/'images/reference_captures'


def test_two_line_location_and_recipe_titles_identify_the_same_product():
    vision = ResourceVision()
    location = vision.observe((FIXTURES/'blueberry_cheesecake_location.png').read_bytes()).popups[0]
    png = (FIXTURES/'blueberry_cheesecake_recipe.png').read_bytes()
    recipe = vision.observe(png).popups[0]
    assert location.navigation and not location.rows
    assert recipe.recipe_complete and len(recipe.rows) == 2
    assert vision.titles_match(location.title_png,recipe.title_png)
    assert _decode(recipe.title_png).shape[0] > 90
    assert [read_quantity(png,row.quantity_box) for row in recipe.rows] == [Quantity(0,1),Quantity(4,2)]


def test_single_line_title_does_not_absorb_nearby_machine_controls():
    vision = ResourceVision()
    icecream = vision.observe((REFERENCES/'resource_icecream_recipe.png').read_bytes()).popups[0]
    blueberry = vision.observe((FIXTURES/'blueberry_cheesecake_recipe.png').read_bytes()).popups[0]
    assert _decode(icecream.title_png).shape[:2] == (43,266)
    assert not vision.titles_match(icecream.title_png,blueberry.title_png)


@pytest.mark.parametrize('name', ['red_berry_location_fresh.png', 'red_berry_location_joined.png'])
def test_location_title_survives_connection_to_order_paper(name):
    vision = ResourceVision()
    first = vision.observe((FIXTURES/'red_berry_location.png').read_bytes()).popups[0]
    second = vision.observe((FIXTURES/name).read_bytes()).popups[0]
    assert len(first.navigation) == len(second.navigation) == 1
    assert not first.rows and not second.rows
    assert vision.titles_match(first.title_png, second.title_png)
    assert abs(_decode(first.title_png).shape[1]-_decode(second.title_png).shape[1]) <= 2
    other = vision.observe((FIXTURES/'blueberry_cheesecake_location.png').read_bytes()).popups[0]
    assert not vision.titles_match(second.title_png, other.title_png)


def test_joined_white_sugar_prompt_excludes_order_customer_name():
    vision = ResourceVision()
    location = vision.observe((REFERENCES/'resource_white_sugar_prompt_joined.png').read_bytes()).popups[0]
    recipe = vision.observe((REFERENCES/'resource_white_sugar_recipe.png').read_bytes()).popups[0]
    assert location.navigation and not location.rows
    assert recipe.recipe_complete
    assert vision.titles_match(location.title_png, recipe.title_png)


@pytest.mark.parametrize('change', ['expired', 'other_popup', 'no_slot', 'moved_product', 'ambiguous'])
def test_expired_recipe_reopens_only_the_same_visible_product(tmp_path, change):
    vision = ResourceVision()
    expired = Screenshot((FIXTURES/'wool_hat_expired_recipe.png').read_bytes(), 1920, 1080, 'expired')
    renewed = Screenshot((FIXTURES/'wool_hat_after_harvest.png').read_bytes(), 1920, 1080, 'renewed')
    scene = vision.observe(expired.png)
    recipe = vision.observe(renewed.png)
    assert not scene.popups and scene.empty_slots
    assert recipe.popups[0].recipe_complete
    assert [read_quantity(renewed.png, row.quantity_box) for row in recipe.popups[0].rows] == [
        Quantity(1, 1), Quantity(17, 1)]
    menu = VisualTarget(680, 220, 130, 115, .99)
    products = [menu]
    if change == 'other_popup':
        scene = recipe
    elif change == 'no_slot':
        scene = SimpleNamespace(popups=(), empty_slots=())
    elif change == 'moved_product':
        products = [VisualTarget(400, 220, 130, 115, .99)]
    elif change == 'ambiguous':
        products *= 2
    vision.find_item = Mock(return_value=products)
    worker = ResourceWorker(Mock(serial='test'), capture=Mock(), cancel_event=threading.Event(),
                            progress=Mock(), state_path=tmp_path/'resources.json', vision=vision)
    worker._tap = Mock(return_value=renewed)
    result, observed, _ = worker._refresh_expired_recipe(expired, scene, menu, b'icon', .9, time.monotonic())
    if change == 'expired':
        worker._tap.assert_called_once_with(menu.center, expired)
        assert result is renewed and observed.popups[0].recipe_complete
    else:
        worker._tap.assert_not_called()
        assert result is expired


def test_floating_chicken_feed_recipe_queues_one_batch(tmp_path):
    from hayday.herd_vision import HerdVision
    from hayday.resource_vision import _png
    menu = Screenshot((FIXTURES/'chicken_feed_menu_before_recipe.png').read_bytes(), 1920, 1080, 'menu')
    recipe = Screenshot((FIXTURES/'chicken_feed_recipe_floating.png').read_bytes(), 1920, 1080, 'recipe')
    client = Mock(serial='test', capture=Mock(return_value=recipe))
    unrelated = SimpleNamespace(work_if_recognized=lambda *a, **kw: None)
    worker = ResourceWorker(client, state_path=tmp_path/'resources.json', fields=unrelated,
                            fruits=unrelated, fishing=unrelated, orchards=unrelated)
    worker._record('feed', 'inventory_baseline', inventory_before={
        'status': 'missing', 'available': 0, 'required': 1})
    worker._known_icon = lambda icon: None
    worker._bind_title = lambda key, title: key
    worker._wait = Mock()
    worker._queue_changed = Mock(return_value=True)
    worker._finish = lambda status, message, frame, **details: ResourceResult(status, message, details)
    icon = _png(HerdVision()._controls('chicken')[0])
    result = worker._at_source(menu, icon, 'feed', None, (), 0)
    assert result.status == 'queued'
    assert worker._state['items']['feed']['pending_stage'] == 'queued'
    client.swipe.assert_called_once()
    x, y, tx, ty = client.swipe.call_args.args
    assert 750 < x < 830 and 425 < y < 480  # Actual floated product.
    assert (tx, ty) in [slot.center for slot in worker.vision.observe(recipe.png).empty_slots]
    worker._at_source(menu, icon, 'feed', None, (), 0)
    assert client.swipe.call_count == 1


@pytest.mark.parametrize('name', ['blueberry_cake_palette_queue', 'blueberry_cake_palette_settled'])
def test_recipe_palette_excludes_the_higher_scoring_queued_copy(tmp_path, name):
    frame = Screenshot((FIXTURES/f'{name}.png').read_bytes(), 1920, 1080, name)
    icon = (FIXTURES/'blueberry_cake_identifier.png').read_bytes()
    worker = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json')
    matches = worker.vision.find_item(frame.png, icon, min_scale=.8, max_scale=3.4,
                                     region=(0, 0, 1228, 928))
    assert len(matches) == 2
    if name == 'blueberry_cake_palette_queue':
        assert matches[0].center[0] > 900  # Queued copy wins the artwork score.
    products = worker._production_items(frame, icon, .84)
    assert len(products) == 1
    assert 300 < products[0].center[0] < 450 and 600 < products[0].center[1] < 680
    worker.vision.find_item = Mock(return_value=[t for t in matches if t.center[0] > 900])
    assert not worker._production_items(frame, icon, .84)


@pytest.mark.parametrize('product', ['blue_sweater', 'pancake'])
def test_registered_full_artwork_retains_its_palette_stock_badge(tmp_path, product):
    root = Path(__file__).resolve().parents[1]/'images/resources/products'
    frame = Screenshot((REFERENCES/f'resource_{product}_menu.png').read_bytes(), 1920, 1080, product)
    worker = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json')
    icon = worker.vision.production_reference((root/f'{product}_identifier.png').read_bytes(),
        (root/f'{product}_title.png').read_bytes(), frame.height)
    assert icon is not None
    assert len(worker._production_items(frame, icon, .90)) == 1


def test_cropped_board_hat_keeps_the_entire_badge_above_its_missing_pompom(tmp_path):
    frame = Screenshot((FIXTURES/'wool_hat_palette_badge.png').read_bytes(), 1920, 1080, 'hat')
    icon = (FIXTURES/'wool_hat_palette_identifier.png').read_bytes()
    worker = ResourceWorker(Mock(serial='test'), state_path=tmp_path/'resources.json')
    products = worker._production_items(frame, icon, .84)
    assert len(products) == 1
    assert 700 < products[0].center[0] < 800 and products[0].center[1] < 320

"""Milk collection uses a complete cow pen and independently verified stock."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.animals import SheepVision
from hayday.cows import CowVision, prepare_cows
from hayday.fruit import FruitWorker
from hayday.quantities import Quantity, read_quantity
from hayday.resource_vision import ResourceVision
from hayday.resources import ResourceWorker

FIXTURES = Path(__file__).parent/'fixtures/orders'
REFERENCES = Path(__file__).resolve().parents[1]/'images/reference_captures'


@pytest.fixture(scope='module')
def vision():
    return CowVision()


@pytest.fixture(scope='module')
def ready(vision):
    return vision.harvest((FIXTURES/'cow_ready_pen.png').read_bytes(),vision.item,lambda:False)


def test_ready_cows_require_a_complete_pen_and_bucket(ready):
    assert ready and ready.species == 'milk' and ready.action == 'harvest'
    assert ready.target and ready.available == 6 and ready.sweep_path
    assert 1020 < ready.target.center[0] < 1435 and 520 < ready.target.center[1] < 715
    assert all(cv2.pointPolygonTest(np.array(ready.pen_polygon,np.int32),p,False) >= 0
               for p in ready.sweep_path)


def test_enabled_bucket_and_same_pen_remain_usable_when_cows_turn(vision,ready):
    result = vision.harvest((FIXTURES/'cow_ready_moving.png').read_bytes(),vision.item,lambda:False)
    assert result and result.target and result.available == 6
    assert FruitWorker._same_pen(ready,result)


def test_enabled_bucket_collects_in_the_live_accessory_view(vision):
    png = (FIXTURES/'cow_accessory_bucket.png').read_bytes()
    result = vision.harvest(png,vision.item,lambda:False)
    assert result and result.target and result.available == 1 and result.sweep_path
    assert all(cv2.pointPolygonTest(np.array(result.pen_polygon,np.int32),p,False) >= 0
               for p in result.sweep_path)


def test_timer_can_preserve_a_fresh_pen_with_an_enabled_bucket_and_obscured_arrow(vision):
    before = (FIXTURES/'cow_timer_clear.png').read_bytes()
    after = (FIXTURES/'cow_timer_bucket.png').read_bytes()
    polygon = vision.enclosures(before,lambda:False)[0][2]
    guide = vision.harvest(after,vision.item,lambda:False)
    assert guide and guide.action == 'harvest' and guide.target is None
    found = vision.harvest(after,vision.item,lambda:False,pen_hint=(before,polygon))
    assert found and found.target and found.sweep_path and found.available == 0
    assert vision.same_pen(found.pen_polygon,polygon)
    # A previous camera position must not authorize an unseen boundary.
    image = cv2.imdecode(np.frombuffer(before,np.uint8),cv2.IMREAD_COLOR)
    shifted = cv2.warpAffine(image,np.float32([[1,0,80],[0,1,0]]),(1920,1080))
    stale = vision.harvest(after,vision.item,lambda:False,
        pen_hint=(cv2.imencode('.png',shifted)[1].tobytes(),polygon))
    assert stale and stale.target is None


def test_current_fully_grey_bucket_cannot_authorize_collection_even_with_a_pen_hint(vision):
    before = (FIXTURES/'cow_timer_clear.png').read_bytes()
    grey = (FIXTURES/'cow_accessory_grey_bucket.png').read_bytes()
    polygon = vision.enclosures(before,lambda:False)[0][2]
    assert vision.harvest(grey,vision.item,lambda:False,pen_hint=(before,polygon)) is None


@pytest.mark.parametrize('name',['cow_ready_timer.png','cow_edge_menu.png'])
def test_hidden_or_partial_pen_needs_repositioning_before_harvest(vision,name):
    result = vision.harvest((FIXTURES/name).read_bytes(),vision.item,lambda:False)
    assert result and result.target is None and not result.sweep_path


def test_clear_pen_is_reopenable_but_does_not_authorize_bucket_input(vision):
    png = (FIXTURES/'cow_clear.png').read_bytes()
    assert vision.enclosure(png,vision.item,lambda:False)
    assert vision.harvest(png,vision.item,lambda:False) is None
    assert vision.harvest(png,SheepVision().item,lambda:False) is None
    assert vision.harvest(png,vision.item,lambda:True) is None


def test_clean_milk_recipe_icon_is_supported_without_quantity_pixels(vision):
    recipe = ResourceVision().observe((FIXTURES/'red_berry_cake_recipe.png').read_bytes()).popups[0]
    assert len(recipe.rows) == 4 and recipe.recipe_complete
    assert [vision.supports(row.icon_png,lambda:False) for row in recipe.rows] == [False,False,True,False]
    result = vision.harvest((FIXTURES/'cow_ready_from_recipe.png').read_bytes(),
                            recipe.rows[2].icon_png,lambda:False)
    assert result and result.species == 'milk'


def test_ready_cows_without_fence_cannot_authorize_a_sweep(vision,ready):
    image = cv2.imread(str(FIXTURES/'cow_ready_pen.png'))
    ring = np.zeros(image.shape[:2],np.uint8)
    cv2.polylines(ring,[np.array(ready.pen_polygon,np.int32)],True,255,32)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    wood = (ring > 0) & (hsv[:,:,0] < 19) & (hsv[:,:,1] > 105)
    image[wood] = (35,140,60)
    png = cv2.imencode('.png',image)[1].tobytes()
    result = vision.harvest(png,vision.item,lambda:False)
    assert result and result.target is None and not result.sweep_path


@pytest.mark.parametrize('name',['cow_wide_clear.png','cow_wide_cloud.png','cow_accessory_clear.png'])
def test_visible_trough_rim_locates_pen_despite_cows_and_accessories(vision,name):
    png = (FIXTURES/name).read_bytes()
    found = vision.enclosures(png,lambda:False)
    assert len(found) == 1
    anchor,ground,polygon = found[0]
    assert 1100 < anchor.center[0] < 1400
    assert cv2.pointPolygonTest(np.array(polygon,np.int32),ground.center,False) > 0
    assert max(p[0] for p in polygon) < 1500
    # A clear world pen alone cannot authorize any bucket or feed gesture.
    assert vision.harvest(png,vision.item,lambda:False) is None


def test_enabled_bucket_authorizes_the_pen_without_individual_cow_artwork(vision,ready):
    image = cv2.imread(str(FIXTURES/'cow_ready_pen.png'))
    interior = np.zeros(image.shape[:2],np.uint8)
    cv2.fillConvexPoly(interior,np.array(ready.pen_polygon,np.int32),255)
    interior = cv2.erode(interior,np.ones((13,13),np.uint8))
    interior[521:593,1120:1208] = 0  # Preserve the independently verified trough.
    image[interior > 0] = (60,155,195)
    result = vision.harvest(cv2.imencode('.png',image)[1].tobytes(),vision.item,lambda:False)
    assert result and result.target and result.sweep_path
    assert FruitWorker._same_pen(ready,result)


def test_neighboring_cow_pens_keep_separate_sweep_boundaries(vision):
    image = cv2.imread(str(FIXTURES/'cow_wide_clear.png'))
    image = cv2.warpAffine(image,np.float32([[1,0,-220],[0,1,0]]),
                           (image.shape[1],image.shape[0]))
    found = vision.enclosures(cv2.imencode('.png',image)[1].tobytes(),lambda:False)
    assert len(found) == 2
    assert not vision.same_pen(found[0][2],found[1][2])


def test_clouded_ready_cows_still_require_the_wooden_fence(vision):
    png = (FIXTURES/'cow_wide_cloud.png').read_bytes()
    _,_,polygon = vision.enclosures(png,lambda:False)[0]
    image = cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
    ring = np.zeros(image.shape[:2],np.uint8)
    cv2.polylines(ring,[np.array(polygon,np.int32)],True,255,32)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    wood = (ring > 0) & (hsv[:,:,0] < 19) & (hsv[:,:,1] > 105)
    image[wood] = (35,140,60)
    assert not vision.enclosures(cv2.imencode('.png',image)[1].tobytes(),lambda:False)


def test_menu_dismissal_avoids_grass_under_the_accessories_wedge(vision):
    frame = Screenshot((FIXTURES/'cow_accessory_wedge.png').read_bytes(),1920,1080,'menu')
    ground = vision.menu_ground(frame,lambda:False)
    assert ground and 325 < ground[0] < 1000
    assert 205 < ground[1] < 850
    assert ground[:2] == ground[2:]
    clear = Screenshot((FIXTURES/'cow_wide_clear.png').read_bytes(),1920,1080,'clear')
    assert vision.menu_ground(clear,lambda:False) is None


def test_failed_menu_dismissal_stops_before_camera_movement(vision,monkeypatch):
    frame = Screenshot((FIXTURES/'cow_accessory_wedge.png').read_bytes(),1920,1080,'menu')
    cancel = threading.Event()
    monkeypatch.setattr(cancel,'wait',lambda _:False)
    inputs = []
    owner = SimpleNamespace(vision=SimpleNamespace(_cow=vision),cancel_event=cancel,
        _frame=lambda:frame,_check=lambda:None,
        client=SimpleNamespace(tap=lambda *a,**kw:inputs.append(('tap',a)),
                               swipe=lambda *a,**kw:inputs.append(('swipe',a))))
    result = prepare_cows(owner,frame,vision.item,'milk')
    assert result.status == 'changed'
    assert [kind for kind,_ in inputs] == ['tap']


def test_returning_from_resource_work_also_avoids_the_cow_menu(vision):
    frame = Screenshot((FIXTURES/'cow_accessory_wedge.png').read_bytes(),1920,1080,'menu')
    taps = []
    worker = SimpleNamespace(fruits=SimpleNamespace(vision=SimpleNamespace(_cow=vision)),
        cancel_event=threading.Event(),_tap=lambda point,frame:taps.append(point))
    ResourceWorker._dismiss(worker,frame)
    assert len(taps) == 1 and taps[0][0] < 1000


def test_other_animal_troughs_cannot_identify_a_cow_pen(vision):
    image = cv2.imread(str(FIXTURES/'cow_accessory_clear.png'))
    # Remove both cow pens, retaining the neighboring pigs, sheep and goats.
    for polygon in (((1050,495),(1270,374),(1495,492),(1270,628)),
                    ((1370,650),(1580,534),(1790,650),(1580,790))):
        cv2.fillConvexPoly(image,np.array(polygon,np.int32),(35,140,60))
    assert not vision.enclosures(cv2.imencode('.png',image)[1].tobytes(),lambda:False)


@pytest.mark.parametrize('name,stock,enabled',[
    ('cow_after_harvest.png',6,True),
    ('cow_feed_before.png',6,True),
    ('cow_feed_after.png',1,False),
])
def test_live_harvest_and_feeding_disable_the_bucket(vision,name,stock,enabled):
    png = (FIXTURES/name).read_bytes()
    assert vision.harvest(png,vision.item,lambda:False) is None
    feed = vision.harvest(png,vision.item,lambda:False,feeding=True)
    assert feed and feed.available == stock and feed.enabled is enabled
    assert feed.target and feed.sweep_path


@pytest.mark.parametrize('name,expected',[
    ('cow_butter_recipe.png',[Quantity(0,2)]),
    ('cow_milk_confirmed.png',[Quantity(5,2)]),
    ('resource_violet_dress_recipe_bright.png',[Quantity(2,2),Quantity(11,1),Quantity(0,1)]),
    ('resource_icecream_recipe.png',[Quantity(1,1),Quantity(6,1),Quantity(1,1)]),
])
def test_recipe_crop_includes_first_digit_and_excludes_ingredient_art(name,expected):
    path = FIXTURES/name if name.startswith('cow_') else REFERENCES/name
    png = path.read_bytes()
    rows = [row for popup in ResourceVision().observe(png).popups for row in popup.rows]
    assert [read_quantity(png,row.quantity_box) for row in rows] == expected


def cow_worker(tmp_path,monkeypatch,ready):
    cancel = threading.Event()
    monkeypatch.setattr(cancel,'wait',lambda _:cancel.is_set())
    frame = Screenshot(b'fresh-frame',1920,1080,'first')
    path = tmp_path/'fruit.json'
    inputs = []

    def drag(points,**kwargs):
        assert json.loads(path.read_text())['items']['milk']['stage'] == 'attempted'
        inputs.append(points)

    worker = FruitWorker(SimpleNamespace(serial='mumu-test',drag_path=drag),lambda:frame,cancel,
                         lambda _:None,path,vision=SimpleNamespace(basket=lambda *a,**kw:ready))
    worker.storage = SimpleNamespace(read=lambda *a:SimpleNamespace(full=False))
    monkeypatch.setattr(worker,'_feed_animal',lambda *a:False)
    return worker,frame,inputs


def test_milk_intent_survives_restart_until_two_fresh_inventory_gains(tmp_path,monkeypatch,ready):
    worker,frame,inputs = cow_worker(tmp_path,monkeypatch,ready)
    baseline = {'status':'missing','available':0,'required':2}
    result = worker.work_if_recognized(frame,b'icon','milk',baseline)
    assert result.details['pending_harvest'] and len(inputs) == 1
    restored = FruitWorker(worker.client,worker.capture,worker.cancel_event,worker.progress,
                          worker.state_path,vision=worker.vision)
    assert restored.work_if_recognized(frame,b'icon','milk',baseline).details['pending_harvest']
    assert len(inputs) == 1
    restored.observe_inventory('milk','missing',1,2,'gain-one')
    restored.observe_inventory('milk','missing',1,2,'gain-one')
    assert restored.state['items']['milk']['stage'] == 'attempted'
    restored.observe_inventory('milk','missing',1,2,'gain-two')
    assert restored.state['items']['milk']['stage'] == 'confirmed'


def test_unreadable_milk_baseline_cannot_leave_an_unconfirmable_harvest(tmp_path,monkeypatch,ready):
    worker,frame,inputs = cow_worker(tmp_path,monkeypatch,ready)
    result = worker.work_if_recognized(frame,b'icon','milk',{'status':'missing','required':None})
    assert result.status == 'unsupported' and not inputs and not worker.state['items']


def test_feed_action_cannot_replace_harvest_during_revalidation(tmp_path,monkeypatch,ready):
    worker,frame,inputs = cow_worker(tmp_path,monkeypatch,ready)
    observations = iter([ready,*[replace(ready,action='feed')]*4])
    worker.vision.basket = lambda *a,**kw:next(observations)
    result = worker.work_if_recognized(frame,b'icon','milk',{'status':'missing','available':0,'required':2})
    assert result.status == 'changed' and not inputs

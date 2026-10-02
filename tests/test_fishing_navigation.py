"""Alternate fishing areas are inspected with bounded, revalidated water taps."""
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.fishing_navigation import FishingNavigator, water_clearance, water_pinch, water_points
from hayday.resources import ResourceChanged

FIXTURES = Path(__file__).parent/'fixtures/orders'


def test_water_search_stays_inside_the_world_and_avoids_timer_and_hud():
    png = (FIXTURES/'fishing_cooldown.png').read_bytes()
    points = water_points(png)
    assert len(points) > 2
    clearance = water_clearance(png)
    for x,y in points:
        assert 1920*.22 <= x < 1920*.80 and 1080*.22 <= y < 1080*.80
        assert clearance[y,x] >= 18
        assert not (775 < x < 1750 and 510 < y < 675)
    center,span = water_pinch(png)
    assert all(clearance[center[1],x] >= 18 for x in (center[0]-span//2,center[0]+span//2))


@pytest.mark.parametrize('name', ['fishing_workbench_completed.png', 'fishing_workbench_sway.png'])
def test_workbench_dismissal_avoids_translucent_menu_and_fishing_hut(name):
    png = (FIXTURES/name).read_bytes()
    points = water_points(png, left=.12, minimum=50)
    assert points
    # The hut's large anchor can look like water; require a broad clear patch
    # farther from its dock and from the palette's transparent green backdrop.
    assert 230 < points[0][0] < 350 and 740 < points[0][1] < 865
    assert water_clearance(png, left=.12)[points[0][1],points[0][0]] >= 50


def test_visible_spot_sparkle_prioritizes_nearby_clear_water():
    png = (FIXTURES/'fishing_other_spot.png').read_bytes()
    first = water_points(png)[0]
    assert 1230 < first[0] < 1330 and 590 < first[1] < 650
    assert water_clearance(png)[first[1],first[0]] >= 18


def navigation_worker(monkeypatch,*,ready_after=2,obscure_fresh=False):
    cancel = threading.Event()
    monkeypatch.setattr(cancel,'wait',lambda _:cancel.is_set())
    client = SimpleNamespace(taps=0,pinches=0,captures=0)
    clear = cv2.imencode('.png',np.full((1080,1920,3),(170,145,85),np.uint8))[1].tobytes()

    def capture():
        client.captures += 1
        return Screenshot(clear,1920,1080,str(client.captures))

    def check():
        if cancel.is_set():
            raise ResourceChanged('Fishing cancelled')

    def tap(*a,**kw):
        assert client.captures >= 2
        client.taps += 1

    def pinch(**kw):
        assert kw['cancel_event'] is cancel
        client.pinches += 1

    client.tap,client.pinch_zoom_out = tap,pinch
    def home(*a):
        return None if obscure_fresh and client.captures else object()
    vision = SimpleNamespace(home=home,menu=lambda *a:object() if client.taps >= ready_after else None)
    owner = SimpleNamespace(client=client,vision=vision,cancel_event=cancel,_frame=capture,_check=check,progress=lambda _:None)
    monkeypatch.setattr('hayday.fishing_navigation.CameraNavigator._modal_visible',lambda _:False)
    monkeypatch.setattr('hayday.fishing_navigation.water_pinch',lambda _ :((950,400),400))
    monkeypatch.setattr('hayday.fishing_navigation.water_clearance',lambda _, **kw:np.full((1080,1920),50,np.float32))
    monkeypatch.setattr('hayday.fishing_navigation.water_points',lambda _:tuple((600+i*100,400) for i in range(6)))
    return owner,Screenshot(clear,1920,1080,'initial')


def test_search_returns_a_picker_without_casting_or_changing_stock(monkeypatch):
    owner,frame = navigation_worker(monkeypatch)
    assert FishingNavigator(owner).find_spot(frame)
    assert owner.client.taps == 2 and owner.client.pinches == 1


def test_search_is_bounded_when_every_spot_is_resting(monkeypatch):
    owner,frame = navigation_worker(monkeypatch,ready_after=100)
    assert FishingNavigator(owner).find_spot(frame) is None
    assert owner.client.taps == 6 and owner.client.pinches == 1


def test_ripple_invalidated_point_is_skipped_without_input(monkeypatch):
    owner, frame = navigation_worker(monkeypatch)
    navigator = FishingNavigator(owner)
    original = navigator._fresh_water
    observations = 0

    def fresh(points, **kwargs):
        nonlocal observations
        observations += 1
        return None if observations == 2 else original(points, **kwargs)

    monkeypatch.setattr(navigator, '_fresh_water', fresh)
    assert navigator.find_spot(frame)
    assert observations == 4 and owner.client.taps == 2


def test_disappearing_fishing_view_and_cancellation_prevent_navigation(monkeypatch):
    owner,frame = navigation_worker(monkeypatch,obscure_fresh=True)
    assert FishingNavigator(owner).find_spot(frame) is None
    assert owner.client.taps == owner.client.pinches == 0
    owner.cancel_event.set()
    with pytest.raises(ResourceChanged,match='cancelled'):
        FishingNavigator(owner).find_spot(frame)
    assert owner.client.taps == owner.client.pinches == 0

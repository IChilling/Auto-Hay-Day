"""Farm navigation must leave the fishing area before searching for a board."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.farm_home import FarmHomeRecovery
from hayday.fishing_vision import FishingVision
from hayday.tutorials import TutorialBlocked

FIXTURES = Path(__file__).parent/'fixtures/orders'


def frame(name, stamp):
    return Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, stamp)


@pytest.fixture(scope='module')
def vision():
    return FishingVision()


def test_home_recognizes_remote_fishing_edge_but_not_farm_shop(vision):
    home = vision.home(frame('home_fishing_edge.png', 'edge').png)
    assert home is not None and 50 < home.center[0] < 150 and 900 < home.center[1] < 1040
    assert vision.home(frame('home_farm.png', 'farm').png) is None


def test_return_requires_two_home_frames_and_two_clear_frames(vision):
    first = frame('home_fishing_edge.png', 'first')
    farm = frame('home_farm.png', 'returned')
    capture = Mock(side_effect=[replace(first, captured_at='fresh'), farm,
                               replace(farm, captured_at='confirmed')])
    client = Mock(serial='device')
    recovery = FarmHomeRecovery(client, capture=capture, wait=lambda _: None, vision=vision)
    assert recovery.process(first).captured_at == 'confirmed'
    client.tap.assert_called_once()
    client.swipe.assert_not_called()


@pytest.mark.parametrize('fresh_stamp,allow_input', [('first', True), ('fresh', False)])
def test_stale_home_or_disabled_navigation_cannot_tap(vision, fresh_stamp, allow_input):
    first = frame('home_fishing_edge.png', 'first')
    client = Mock(serial='device')
    recovery = FarmHomeRecovery(client, capture=lambda: replace(first, captured_at=fresh_stamp),
                                wait=lambda _: None, vision=vision)
    with pytest.raises(TutorialBlocked):
        recovery.process(first, allow_input=allow_input)
    client.tap.assert_not_called()


def test_uncertain_home_tap_cannot_repeat(vision):
    first = frame('home_fishing_edge.png', 'first')
    frames = iter(replace(first, captured_at=str(index)) for index in range(20))
    client = Mock(serial='device')
    recovery = FarmHomeRecovery(client, capture=lambda: next(frames), wait=lambda _: None, vision=vision)
    with pytest.raises(TutorialBlocked, match='did not return'):
        recovery.process(first)
    with pytest.raises(TutorialBlocked, match='previous Home tap'):
        recovery.process(first)
    client.tap.assert_called_once()


def test_board_search_uses_returned_farm_frame_before_any_camera_gesture(monkeypatch):
    first = frame('home_fishing_edge.png', 'first')
    farm = frame('home_farm.png', 'returned')
    frames = iter([first, replace(first, captured_at='fresh'), farm,
                   replace(farm, captured_at='confirmed')])
    client = Mock(serial='device')
    verifier = Mock(reference_error='', verify=Mock(return_value=SimpleNamespace(verified=False)))
    detector = Mock(detect=Mock(return_value=SimpleNamespace(match=None)))
    navigator = CameraNavigator(client, capture=lambda: next(frames), verifier=verifier,
                                detector=detector, max_zoom_steps=0)
    seen = []
    monkeypatch.setattr(navigator, '_search_drag', lambda f, *_: seen.append(f) or None)
    result = navigator.open_board()
    assert result.status == 'blocked'
    assert len(seen) == 1 and seen[0].captured_at == 'confirmed'
    client.tap.assert_called_once()
    client.swipe.assert_not_called()
    client.pinch_zoom_out.assert_not_called()

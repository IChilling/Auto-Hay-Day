"""Reward collection can defeat coarse search without moving the board."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator, CameraResult
from hayday.resource_vision import _decode, _png
from hayday.vision import BoardDetector

FIXTURES = Path(__file__).parent/'fixtures/orders'


def reward_navigation():
    before = (FIXTURES/'board_reward_before.png').read_bytes()
    frame = Screenshot((FIXTURES/'board_reward_after.png').read_bytes(), 1920, 1080, 'after')
    detector = BoardDetector()
    previous = detector.detect(before).match
    assert previous is not None and detector.detect(frame.png).match is None
    assert detector.revalidate(frame.png, previous).match is not None
    navigator = CameraNavigator(Mock(serial='test'), detector=detector,
        verifier=Mock(verify=Mock(return_value=SimpleNamespace(verified=False))),
        capture=Mock(side_effect=[frame, replace(frame, captured_at='fresh')]))
    navigator._verify_open = Mock(return_value=CameraResult('opened', 'confirmed'))
    return navigator, previous, frame


def test_reward_retry_revalidates_the_known_board_in_two_fresh_frames():
    navigator, previous, _ = reward_navigation()
    navigator.detector.detect = Mock(wraps=navigator.detector.detect)
    result = navigator._retry_visible_board(previous)
    assert result.success
    navigator.detector.detect.assert_not_called()
    assert navigator.capture.call_count == 2
    navigator.client.tap.assert_called_once_with(404, 371, width=1920, height=1080)


def test_board_disappearing_before_retry_cannot_authorize_another_tap():
    navigator, previous, frame = reward_navigation()
    image = _decode(frame.png)
    image[280:460, 330:475] = 0
    navigator.capture = Mock(side_effect=[frame, replace(frame, png=_png(image), captured_at='hidden')])
    assert navigator._retry_visible_board(previous).status == 'changed'
    navigator.client.tap.assert_not_called()

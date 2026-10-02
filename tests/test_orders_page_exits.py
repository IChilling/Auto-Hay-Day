"""Recorded accidental pages must close only on fresh, exact exit evidence."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.orders import OrderRunner
from hayday.page_exits import PageExitRecovery, PageExitVision
from hayday.tutorials import TutorialBlocked

FIXTURES = Path(__file__).parent/'fixtures/orders'
KINDS = ('achievements', 'county_fair_tutorial', 'county_fair', 'farm_expansion', 'catalog')


def frame(kind, stamp='first'):
    return Screenshot((FIXTURES/f'page_{kind}.png').read_bytes(), 1920, 1080, stamp)


@pytest.fixture(scope='module')
def vision():
    return PageExitVision()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('scale', [1, 2/3])
def test_exact_pages_and_scaled_layouts(vision, kind, scale):
    image = cv2.imdecode(np.frombuffer(frame(kind).png, np.uint8), 1)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    seen = vision.observe(cv2.imencode('.png', image)[1].tobytes())
    assert seen and seen.kind == kind
    assert seen.target.center[0] > image.shape[1]*.8
    assert seen.target.center[1] < image.shape[0]*.2


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('feature', [0, 1, 2])
def test_missing_identity_never_claims_page_and_missing_exit_never_matches(vision, kind, feature):
    image = cv2.imdecode(np.frombuffer(frame(kind).png, np.uint8), 1)
    (x, y, w, h), _ = vision.references[kind][feature]
    image[y:y+h, x:x+w] = (70, 140, 30)
    result = vision.observe(cv2.imencode('.png', image)[1].tobytes())
    assert result is None if feature == 2 else result is None or result.kind == 'incidental_popup'


def test_county_fair_exit_stays_available_with_an_item_popover(vision):
    result = vision.observe((FIXTURES/'page_county_fair_popup.png').read_bytes())
    assert result and result.kind == 'county_fair'


def test_expansion_reuses_the_existing_generic_exit(vision):
    assert np.array_equal(vision.references['farm_expansion'][2][1],
                          vision.references['achievements'][2][1])


def test_expansion_opening_animation_settles_before_one_exit_tap(vision):
    opening = frame('expansion_opening')
    settled = frame('expansion_settled', 'settled')
    clear = Screenshot((FIXTURES/'home_farm.png').read_bytes(), 1920, 1080, 'clear')
    assert vision.observe(opening.png).kind == 'incidental_popup'
    assert vision.observe(settled.png).kind == 'farm_expansion'
    client = Mock(serial='test')
    captures = [settled, replace(settled, captured_at='confirmed'), clear,
                replace(clear, captured_at='clear2')]
    recovery = PageExitRecovery(client, capture=Mock(side_effect=captures), wait=Mock(), vision=vision)
    assert recovery.process(opening).captured_at == 'clear2'
    client.tap.assert_called_once_with(1674, 106, width=1920, height=1080)
    assert recovery.events[-1]['stage'] == 'dismissed'
    assert np.array_equal(vision.references['catalog'][2][1],
                          vision.references['achievements'][2][1])


def test_stacked_tutorial_and_page_use_separate_confirmed_exits(vision):
    clear = Screenshot(cv2.imencode('.png', np.zeros((1080, 1920, 3), np.uint8))[1].tobytes(),
                       1920, 1080, 'clear')
    captures = [frame('county_fair_tutorial', 'confirmed'),
                frame('county_fair', 'revealed'), frame('county_fair', 'settled'),
                frame('county_fair', 'confirmed_main'), clear, replace(clear, captured_at='clear2')]
    client = Mock(serial='test')
    recovery = PageExitRecovery(client, capture=Mock(side_effect=captures), wait=Mock(), vision=vision)
    assert recovery.process(frame('county_fair_tutorial')).captured_at == 'clear2'
    assert client.tap.call_count == 2
    assert [e['kind'] for e in recovery.events if e['stage'] == 'dismissed'] == [
        'county_fair_tutorial', 'county_fair']


@pytest.mark.parametrize('change', ['stale', 'different'])
def test_changed_or_reused_confirmation_sends_no_input(vision, change):
    initial = frame('achievements')
    capture = initial if change == 'stale' else frame('county_fair', 'changed')
    client = Mock(serial='test')
    recovery = PageExitRecovery(client, capture=lambda: capture, wait=Mock(), vision=vision)
    with pytest.raises(TutorialBlocked, match='changed before'):
        recovery.process(initial)
    client.tap.assert_not_called()


def test_uncleared_exit_is_never_repeated(vision):
    initial = frame('achievements')
    captures = [replace(initial, captured_at=str(i)) for i in range(9)]
    client = Mock(serial='test')
    recovery = PageExitRecovery(client, capture=Mock(side_effect=captures), wait=Mock(), vision=vision)
    with pytest.raises(TutorialBlocked, match='remained'):
        recovery.process(initial)
    with pytest.raises(TutorialBlocked, match='uncertain'):
        recovery.process(initial)
    assert client.tap.call_count == 1


def test_pending_delivery_never_runs_incidental_page_recovery(tmp_path):
    initial = frame('county_fair')
    passthrough = Mock(events=[], process=lambda f: f)
    runner = OrderRunner(Mock(serial='test'), tmp_path, maintenance=passthrough,
                         tutorials=passthrough, reconnect=passthrough)
    runner.page_exits = Mock()
    runner._state = {'pending': {'stage': 'sending'}}
    assert runner._recover_capture(initial) is initial
    runner.page_exits.process.assert_not_called()


def test_generic_exit_reuses_button_for_dimmed_illustrated_popup(vision):
    result = vision.observe((FIXTURES/'page_generic_exit.png').read_bytes())
    assert vision.observe((FIXTURES/'dimmed_illustrated_popup.png').read_bytes()) is None
    assert result and result.kind == 'incidental_popup'
    assert result.target.center[0] > 1700 and result.target.center[1] < 150


def test_generic_exit_does_not_close_order_panel_or_farm(vision):
    references = Path(__file__).resolve().parents[1]/'images/reference_captures'
    for name in ('orders_ready.png', 'orders_school_missing.png', 'herd_all-pens.png'):
        source = FIXTURES/name if name.startswith('herd_') else references/name
        assert vision.observe(source.read_bytes()) is None


def test_shifted_achievements_uses_generic_exit_with_partially_dimmed_hud(vision):
    result = vision.observe((FIXTURES/'achievements_shifted.png').read_bytes())
    assert result and result.kind == 'incidental_popup'
    assert 1600 < result.target.center[0] < 1700

"""Recorded sheep motion must not make a fenced soil pen disappear."""
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.animals import SheepVision
from hayday.farming_vision import FarmingVision
from hayday.fruit import FruitWorker
from hayday.resource_vision import VisualTarget, _decode

FIXTURES = Path(__file__).parent/'fixtures/orders'


@pytest.fixture(scope='module')
def sheep():
    return SheepVision()


def test_moving_sheep_preserve_the_verified_feeding_pen(sheep):
    observations = []
    for name in ('sheep_feed_before.png','sheep_feed_shifted.png'):
        png = (FIXTURES/name).read_bytes()
        observed = sheep.harvest(png,sheep.item,lambda:False,feeding=True)
        assert observed and observed.target and observed.available == 1 and observed.sweep_path
        # The production sheep are in the lower wooden-trough enclosure;
        # decorative lambs in the upper blue-trough enclosure stay excluded.
        assert 1100 < observed.target.center[0] < 1250 and 500 < observed.target.center[1] < 600
        assert all(cv2.pointPolygonTest(np.array(observed.pen_polygon,np.int32),p,False) >= 0
                   for p in observed.sweep_path)
        observations.append(observed)
    assert FruitWorker._same_pen(*observations)


def test_clear_pen_is_reopenable_without_using_a_decorative_lamb(sheep):
    png = (FIXTURES/'sheep_pen_clear.png').read_bytes()
    target, polygon = sheep.enclosure(png,sheep.item,lambda:False)
    assert 900 < target.center[0] < 1350 and 500 < target.center[1] < 700
    assert len(polygon) == 4
    assert sheep.enclosure(png,b'wrong item',lambda:True) is None


def test_unfenced_ground_cannot_become_a_sheep_pen(sheep):
    image = _decode((FIXTURES/'sheep_pen_clear.png').read_bytes())
    assert sheep.pen(image,VisualTarget(320,300,82,70,.99)) == ()

    # Retain all soil and animals, but remove the independently required white
    # fence. The occupancy fix must not turn the soil color alone into a pen.
    patch = image[480:740,870:1380]
    white = (patch.min(axis=2) > 175) & (np.ptp(patch,axis=2) < 65)
    patch[white] = (35,140,60)
    assert sheep.pen(image,VisualTarget(1156,512,82,70,.99)) == ()


def test_live_feed_result_has_zero_feed_and_a_wool_growth_timer(sheep):
    png = (FIXTURES/'sheep_fed_timer.png').read_bytes()
    observed = sheep.harvest(png,sheep.item,lambda:False,feeding=True)
    assert observed and observed.available == 0
    assert sheep.harvest(png,sheep.item,lambda:False) is None
    assert FarmingVision().growing(png) is not None


def test_enabled_shears_collect_from_costumed_production_sheep(sheep):
    png = (FIXTURES/'sheep_accessory_ready.png').read_bytes()
    observed = sheep.harvest(png, sheep.item, lambda: False)
    assert observed and observed.action == 'harvest' and observed.target and observed.sweep_path
    # Production sheep wear reindeer accessories here. The upper enclosure
    # contains decorative lambs and must not receive the harvesting gesture.
    assert 1100 < observed.target.center[0] < 1250 and 500 < observed.target.center[1] < 600
    assert all(cv2.pointPolygonTest(np.array(observed.pen_polygon, np.int32), point, False) >= 0
               for point in observed.sweep_path)


def test_grey_shears_cannot_collect_even_with_a_verified_pen(sheep):
    image = _decode((FIXTURES/'sheep_accessory_ready.png').read_bytes())
    original = sheep.harvest(cv2.imencode('.png', image)[1].tobytes(), sheep.item, lambda: False)
    x, y = original.tool.center
    patch = image[y-80:y+85, x-140:x+55]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    blue = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 120) & (hsv[:, :, 1] > 70)
    grey = cv2.cvtColor(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    patch[blue] = grey[blue]
    assert sheep.harvest(cv2.imencode('.png', image)[1].tobytes(), sheep.item, lambda: False) is None


def test_ready_detection_does_not_depend_on_individual_sheep_reference(sheep, monkeypatch):
    original = sheep.matcher._search

    def search(image, reference, *args, **kwargs):
        assert reference is not sheep.references['ready']
        return original(image, reference, *args, **kwargs)

    monkeypatch.setattr(sheep.matcher, '_search', search)
    png = (FIXTURES/'sheep_accessory_ready.png').read_bytes()
    assert sheep.harvest(png, sheep.item, lambda: False).target is not None


def test_grey_feed_remains_disabled_with_positive_stock(sheep):
    observed = sheep.harvest((FIXTURES/'sheep_grey_feed_one.png').read_bytes(),
                             sheep.item, lambda: False, feeding=True)
    assert observed is not None and observed.available == 1 and not observed.enabled


@pytest.mark.parametrize('pending', [False, True])
def test_grey_controls_defer_wool_without_needing_a_timer_or_another_sweep(tmp_path, sheep, monkeypatch, pending):
    frames = [Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, str(i))
              for i, name in enumerate(['sheep_grey_feed_one.png', 'sheep_grey_feed_two.png',
                                        'sheep_grey_feed_two.png'])]
    client = Mock(serial='test')
    worker = FruitWorker(client, Mock(side_effect=frames[1:]), threading.Event(), Mock(),
                         tmp_path/'fruit.json')
    if pending:
        worker.state['items']['wool'] = {'stage': 'confirmed', 'feed_stage': 'attempted',
                                       'feed_operation': 'saved-gesture', 'feed_available_before': 1}
    timer = Mock(side_effect=AssertionError('A timer is not needed to defer two grey tools.'))
    monkeypatch.setattr(FarmingVision, 'growing', timer)
    result = worker.work_if_recognized(frames[0], sheep.item, 'wool')
    assert result.status == 'waiting' and result.details['defer_item']
    assert 'grey' in result.message
    client.tap.assert_not_called()
    client.drag_path.assert_not_called()
    if pending:
        entry = worker.state['items']['wool']
        assert entry['feed_stage'] == 'confirmed' and entry['feed_outcome'] == 'already_fed'
        assert entry['feed_available_after'] == 1 and entry['feed_operation'] == 'saved-gesture'


def test_a_repeated_grey_frame_cannot_confirm_idle_sheep(tmp_path):
    frame = Screenshot((FIXTURES/'sheep_grey_feed_one.png').read_bytes(), 1920, 1080, 'same')
    worker = FruitWorker(Mock(serial='test'), Mock(return_value=frame), threading.Event(), Mock(),
                         tmp_path/'fruit.json')
    assert not worker._sheep_controls_idle(frame)


def test_full_wool_coats_do_not_hide_the_complete_fenced_pen(sheep):
    png = (FIXTURES/'sheep_woolly_clear_pen.png').read_bytes()
    pens = sheep.enclosures(png, lambda: False)
    assert len(pens) == 1
    trough, _, polygon = pens[0]
    assert min(x for x, y in polygon) < 1030 and max(y for x, y in polygon) > 725
    assert sheep.pen(_decode(png), trough) == ()  # Covered soil alone missed this pen.
    assert max(y for x, y in polygon) < 800  # Decorative lambs above stay excluded.
    image = _decode(png)
    white = (image.min(axis=2) > 175) & (np.ptp(image, axis=2) < 65)
    image[white] = (35, 140, 60)
    assert not sheep.fenced_pen(image, trough)


def test_timer_cannot_truncate_a_freshly_verified_complete_sheep_pen(sheep):
    clear = (FIXTURES/'sheep_woolly_clear_pen.png').read_bytes()
    polygon = sheep.enclosures(clear, lambda: False)[0][2]
    menu = (FIXTURES/'sheep_ready_timer_settled.png').read_bytes()
    assert FarmingVision().growing(menu) is not None
    observed = sheep.harvest(menu, sheep.item, lambda: False, pen_hint=(clear, polygon))
    assert observed.target and observed.pen_polygon == polygon
    assert max(y for x, y in observed.sweep_path) > 700


def test_enabled_shears_with_individual_timer_harvest_instead_of_feeding_or_deferring(tmp_path, sheep):
    before = Screenshot((FIXTURES/'sheep_ready_timer_arriving.png').read_bytes(), 1920, 1080, 'start')
    menu = Screenshot((FIXTURES/'sheep_ready_timer_settled.png').read_bytes(), 1920, 1080, 'menu')
    clear = Screenshot((FIXTURES/'sheep_woolly_clear_pen.png').read_bytes(), 1920, 1080, 'clear')
    frames = [menu, clear, replace(clear, captured_at='clear2'), replace(menu, captured_at='reopened'),
              replace(menu, captured_at='checked'), replace(menu, captured_at='after'),
              replace(menu, captured_at='after2')]
    cancel = threading.Event()
    cancel.wait = Mock(return_value=False)
    client = Mock(serial='test')
    worker = FruitWorker(client, Mock(side_effect=frames), cancel, Mock(), tmp_path/'fruit.json')
    worker._feed_sheep = Mock(side_effect=AssertionError('Enabled shears cannot enter feeding or timer deferral.'))
    result = worker.work_if_recognized(before, sheep.item, 'wool', {'available': 0, 'required': 1})
    assert result.details['pending_harvest']
    assert client.tap.call_count == 2  # Dismiss timer, then select verified pen soil.
    client.drag_path.assert_called_once()
    entry = worker.state['items']['wool']
    assert entry['stage'] == 'attempted' and entry['inventory_before']['available'] == 0
    assert max(y for x, y in entry['pen_polygon']) > 725


@pytest.mark.parametrize('failure', ['missing_pen', 'repeated_frame'])
def test_unverified_ready_pen_does_not_get_reported_as_growing_or_fed(tmp_path, sheep, failure):
    before = Screenshot((FIXTURES/'sheep_ready_timer_arriving.png').read_bytes(), 1920, 1080, 'start')
    menu = Screenshot((FIXTURES/'sheep_ready_timer_settled.png').read_bytes(), 1920, 1080, 'menu')
    clear = Screenshot((FIXTURES/'sheep_woolly_clear_pen.png').read_bytes(), 1920, 1080, 'clear')
    cancel = threading.Event()
    cancel.wait = Mock(return_value=False)
    client = Mock(serial='test')
    second_clear = replace(clear, captured_at='clear2') if failure != 'repeated_frame' else clear
    worker = FruitWorker(client, Mock(side_effect=[menu, clear, second_clear]),
                         cancel, Mock(), tmp_path/'fruit.json')
    worker.vision._sheep = sheep
    with pytest.MonkeyPatch.context() as patch:
        if failure == 'missing_pen':
            patch.setattr(sheep, 'enclosures', lambda *args: ())
        result = worker.work_if_recognized(before, sheep.item, 'wool', {'available': 0, 'required': 1})
    assert result.status == 'waiting' and 'Wool is ready' in result.message
    assert 'wool' not in worker.state['items']
    client.drag_path.assert_not_called()

"""Finished output needs its producer when small artwork resembles buildings."""
from pathlib import Path

import pytest

from hayday.production_ready import ReadyProductVision
from hayday.resource_vision import _decode, _png
from hayday.resources import ResourceWorker

ROOT = Path(__file__).resolve().parents[1]/'images'


def sugar_scene():
    vision = ReadyProductVision()
    png = (ROOT/'reference_captures/resource_brown_sugar_ready.png').read_bytes()
    recipe = vision.matcher.observe(
        (ROOT/'reference_captures/resource_pancake_recipe_sugar_missing.png').read_bytes())
    icon = next(row.icon_png for popup in recipe.popups for row in popup.rows if row.missing)
    return vision, png, icon


def test_sugar_ingredient_finds_its_finished_output_and_sugar_mill():
    vision, png, icon = sugar_scene()
    assert vision.supports(icon)
    found = vision.find(png, icon)
    assert len(found) == 1 and found[0].name == 'Brown Sugar'
    assert 830 < found[0].target.center[0] < 850
    assert 480 < found[0].target.center[1] < 500
    assert 845 < found[0].output.center[0] < 865


@pytest.mark.parametrize('feature', ['anchor', 'support', 'output'])
def test_collection_needs_both_machine_features_and_the_actual_output(feature):
    vision, png, icon = sugar_scene()
    spec = next(p for p in vision.products if p['name'] == 'Brown Sugar')
    image = _decode(png)
    x, y, w, h = spec['features'][feature]['box']
    image[y:y+h, x:x+w] = 0  # Deliberately missing evidence; original fixture stays intact.
    assert not vision.find(_png(image), icon)


def test_generic_sugar_match_cannot_choose_the_dairy_over_close_scoring_output():
    vision, png, icon = sugar_scene()
    candidates = vision.matcher.find_item(png, icon, min_scale=.22, max_scale=.72)
    candidates = [c for c in candidates if 1920*.14 < c.center[0] < 1920*.86
                  and 1080*.2 < c.center[1] < 1080*.84]
    assert candidates[0].center[0] > 950  # Dairy wall has the highest raw score.
    assert any(840 < c.center[0] < 865 for c in candidates)
    assert ResourceWorker._collection_candidate(candidates) is None
    assert ResourceWorker._collection_candidate(candidates[:1]) == candidates[0]


@pytest.mark.parametrize('source', ['resource_wool_hat_ready', 'resource_brown_sugar_ready'])
def test_finished_blue_hat_requires_the_loom_and_visible_output(source):
    vision = ReadyProductVision()
    png = (ROOT/f'reference_captures/{source}.png').read_bytes()
    icon = (ROOT/'production_ready/blue_woolly_hat_identifier.png').read_bytes()
    title = (ROOT/'production_ready/blue_woolly_hat_title.png').read_bytes()
    assert vision.supports(icon, title)
    found = vision.find(png, icon, title)
    if source == 'resource_wool_hat_ready':
        assert len(found) == 1 and found[0].name == 'Blue Woolly Hat'
        assert 960 < found[0].target.center[0] < 980
        assert 595 < found[0].target.center[1] < 615
        image = _decode(png)
        image[655:695, 940:985] = 0
        assert not vision.find(_png(image), icon, title)
    else:
        assert not found  # No Loom/output is visible in this view.

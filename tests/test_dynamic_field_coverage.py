"""Fresh coverage is independent of the last verified planting cache."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from hayday.adb import Screenshot
from hayday.wheating_fields import WheatFields
from hayday.wheating_vision import WheatingVision


def test_smaller_rearranged_section_replaces_cached_coordinates():
    frame = Screenshot(b'farm', 1920, 1080, 'old')
    fields = WheatFields.__new__(WheatFields)
    fields._known_before = frame
    fields._known_points = [(600, 450), (655, 478), (710, 505)]
    fields.run = SimpleNamespace(publish=Mock(), state={'field_layout': {'before': {'file': 'old.png'},
        'points': [list(p) for p in fields._known_points]}})
    fields.worker = SimpleNamespace(plant_before=frame,
        planted_points=[(708, 507), (765, 533)],
        _evidence=Mock(return_value={'file': 'new.png'}))
    fields._remember_layout({'replanted': True})
    assert fields._known_points == [(708, 507), (765, 533)]
    fields._remember_layout({'replanted': True})
    assert len(fields._known_points) == 2
    assert fields.run.state['field_layout']['before'] == {'file': 'new.png'}
    assert fields.run.state['field_layout']['version'] == 2


def test_camera_shift_never_projects_old_positions_into_new_layout():
    old = Screenshot(b'old', 1920, 1080, 'old')
    new = Screenshot(b'new', 1920, 1080, 'new')
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = old, [(600, 450), (655, 478)]
    fields.run = SimpleNamespace(publish=Mock(), state={'field_layout': {'points': [[600,450],[655,478]]}})
    fields.worker = SimpleNamespace(plant_before=new, planted_points=[(755, 578), (810, 605)],
        _saved_frame=lambda proof: new, _evidence=Mock(return_value={'file': 'latest.png'}))
    fields._remember_layout({'replanted': True})
    assert fields._known_points == [(755,578), (810,605)]


def test_growing_known_section_does_not_hide_ripe_or_bare_section():
    for kind in ('ripe', 'empty'):
        frame = Screenshot(b'farm', 1920, 1080, 'new')
        fields = WheatFields.__new__(WheatFields)
        fields._known_before, fields._known_points = frame, [(600,450)]
        fields._tracked_wheat = Mock(return_value=[])
        fields.run = SimpleNamespace(capture=lambda: frame,
            vision=SimpleNamespace(farm=lambda f: True, plots=lambda f, k: [object()] if k == kind else []))
        assert fields.ready_to_harvest()


def test_live_partial_field_requires_expanding_incomplete_route():
    png = (Path(__file__).parent/'fixtures/partial_field_live.png').read_bytes()
    frame = Screenshot(png, 1920, 1080, 'live')
    vision = WheatingVision()
    assert vision.plots(frame, 'ripe') and vision.plots(frame, 'empty')
    assert vision.wheat_outside(frame, [(704,552)])
    sweep = vision.harvest_sweep(frame)
    assert len(sweep) > 10
    sampled = [tuple(p) for a, b in zip(sweep[::2], sweep[1::2])
               for p in np.linspace(a, b, 40)]
    assert not vision.wheat_outside(frame, sampled)


def test_real_picker_camera_shift_keeps_only_freshly_verified_points():
    root = Path(__file__).parent/'fixtures'
    points = json.loads((root/'field_merge.json').read_text())
    old = Screenshot((root/'field_merge_before.png').read_bytes(), 1920, 1080, 'old')
    new = Screenshot((root/'field_merge_after.png').read_bytes(), 1920, 1080, 'new')
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = old, [tuple(p) for p in points['old']]
    fields.run = SimpleNamespace(publish=Mock(), state={'field_layout': {'points': points['old']}})
    fields.worker = SimpleNamespace(plant_before=new, planted_points=points['new'],
        _saved_frame=lambda proof: new, _evidence=Mock(return_value={'file': 'latest.png'}))
    fields._remember_layout({'replanted': True})
    assert len(fields._known_points) == 51
    assert fields._known_before is new
    assert fields._known_points == points['new']
    fields._remember_layout({'replanted': True})
    assert len(fields._known_points) == 51


def test_legacy_merged_layout_is_not_loaded():
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = None, []
    fields.run = SimpleNamespace(state={'field_layout': {'points': [[600,450]]}})
    fields.worker = SimpleNamespace(_saved_frame=Mock())
    fields._load_layout()
    fields.worker._saved_frame.assert_not_called()
    assert fields._known_points == []

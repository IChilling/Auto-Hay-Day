"""The two live stalls and independently varying decorative-font OCR glyphs."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hayday.ad_text import AdTextLine, AdTextObservation
from hayday.adb import Screenshot
from hayday.wheating import WheatingRunner
from hayday.wheating_event_board import WheatEventBoard
from hayday.wheating_neighborhood import WheatNeighborhood
from hayday.wheating_recovery import WheatLaunchRecovery, WheatRecovery
from hayday.wheating_tutorial_panel import tutorial_message
from hayday.wheating_vision import WheatingVision

ROOT = Path(__file__).parent/'fixtures/tutorial_ocr'


def recorded(kind):
    port = '16448' if kind == 'neighborhood' else '16480'
    data = json.loads((ROOT/'text.json').read_text())[port]
    frame = Screenshot((ROOT/f'{kind}.png').read_bytes(), *data['size'], 'initial')
    text = AdTextObservation(lines=tuple(AdTextLine(**line) for line in data['text']['lines']))
    run = SimpleNamespace(vision=WheatingVision(text_reader=Mock(read=Mock(return_value=text))),
        client=SimpleNamespace(serial='test', foreground_package=Mock(return_value='com.supercell.hayday')),
        diagnostics=None, cancel_event=threading.Event(), check=Mock(), wait=Mock(),
        publish=Mock(), tap=Mock())
    popup = WheatNeighborhood(run) if kind == 'neighborhood' else WheatEventBoard(run)
    return frame, run, popup


@pytest.mark.parametrize('kind', ['neighborhood', 'event_info'])
def test_live_stalled_pages_are_recognized_with_recorded_ocr(kind):
    frame, run, popup = recorded(kind)
    observed = popup.observe(frame)
    assert observed is not None
    if kind == 'event_info':
        assert observed[0] == 'info'
    run.tap.assert_not_called()


@pytest.mark.parametrize('kind', ['neighborhood', 'event_info'])
def test_live_pages_qualify_launch_and_reconnect_readiness(kind, tmp_path, monkeypatch):
    frame, run, _ = recorded(kind)
    runner = WheatingRunner(run.client, tmp_path, vision=run.vision)
    runner._recovery = recovery = WheatRecovery(runner)
    launch = WheatLaunchRecovery(runner, capture=Mock())
    monkeypatch.setattr('hayday.launcher.LaunchRecovery._returned_game', lambda *args: False)
    assert recovery.ready(frame)
    assert launch._returned_game(frame)
    run.client.foreground_package.return_value = 'com.google.android.gms'
    assert not launch._returned_game(frame)


@pytest.mark.parametrize('kind', ['neighborhood', 'event_info'])
def test_recognized_live_pages_still_require_fresh_confirmation(kind):
    from hayday.wheating import WheatingPrerequisite
    frame, run, popup = recorded(kind)
    run._capture_raw = Mock(return_value=frame)
    with pytest.raises(WheatingPrerequisite, match='changed before'):
        popup.process(frame)
    run.tap.assert_not_called()


@pytest.mark.parametrize('kind', ['neighborhood', 'event_info'])
@pytest.mark.parametrize('change', ['extra', 'partial', 'unknown', 'error', 'outside'])
def test_ocr_variants_do_not_relax_complete_text_or_panel_checks(kind, change):
    frame, run, popup = recorded(kind)
    text = run.vision.text.read(frame.png)
    if change == 'error':
        text = replace(text, error='OCR failed')
    elif change == 'partial':
        text = replace(text, lines=text.lines[:-1])
    elif change == 'outside':
        text = replace(text, lines=tuple(replace(line, bounds=(10, 10, 100, 20)) for line in text.lines))
    else:
        suffix = ' Buy diamonds' if change == 'extra' else ''
        lines = (replace(text.lines[0], text=text.lines[0].text+suffix
            if change == 'extra' else 'Tap to confirm the purchase'),)+text.lines[1:]
        text = replace(text, lines=lines)
    run.vision.text.read.return_value = text
    assert popup.process(frame) is frame
    run.tap.assert_not_called()


@pytest.mark.parametrize('variant,expected', [
    ('My, my what do we haue here?', 'my my what do we have here'),
    ('My, mg what do we have here?', 'my my what do we have here'),
    ('Tap on bhe info button to show more details and rewards!',
     'tap on the info button to show more details and rewards'),
    ('Tap on the inpo bubbon bo show more details and rewards!',
     'tap on the info button to show more details and rewards'),
    ('Congratulations goujusb unlocked bhe event board!',
     'congratulations you just unlocked the event board'),
    ('Congratulations you jusb unlocked the event board!',
     'congratulations you just unlocked the event board'),
])
def test_recorded_glyph_variations_can_vary_independently(variant, expected):
    assert tutorial_message(variant) == expected


def test_normalization_keeps_unknown_words_and_extra_instructions():
    assert tutorial_message('Tap on the info button to buy diamonds') == 'tap on the info button to buy diamonds'
    assert tutorial_message('Tap on the inxo button') == 'tap on the inxo button'

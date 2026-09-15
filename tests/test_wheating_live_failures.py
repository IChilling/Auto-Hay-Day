"""Regression coverage for the MuMu end-to-end failures."""
import io
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from hayday.ad_text import AdTextLine, AdTextObservation
from hayday.adb import Screenshot
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingPrerequisite, WheatingRunner
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_fields import WheatFields
from hayday.wheating_notifications import WheatNotifications, notification_target
from hayday.wheating_restart import WheatRestart
from hayday.wheating_vision import WheatingVision

FIXTURES = Path(__file__).parent/'fixtures'
LINES = (AdTextLine('Allow Hay Day to send you notifications?', (689,455,542,30)),
         AdTextLine('ALLOW', (926,561,67,15)), AdTextLine("DON'T ALLOW", (895,650,131,16)))


def shot(name, stamp='first'):
    return Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, stamp)


@pytest.mark.parametrize('width,height', [(1920,1080), (1280,720)])
def test_actual_shop_unlock_banner_is_recognized(width, height):
    frame = shot('mumu_shop_locked.png')
    image = Image.open(io.BytesIO(frame.png)).resize((width,height))
    output = io.BytesIO(); image.save(output, 'PNG')
    vision = WheatingVision()
    assert vision.shop_locked_level(Screenshot(output.getvalue(),width,height,'lock')) == 7
    assert vision.shop_locked_level(shot('mumu_soil_bare.png')) is None


def test_locked_shop_stops_after_one_verified_building_tap(tmp_path):
    locked = shot('mumu_shop_locked.png', 'locked')
    client = SimpleNamespace(serial='test', capture=Mock(return_value=locked), tap=Mock())
    runner = WheatingRunner(client, tmp_path)
    runner.vision = WheatingVision(); runner.wait = Mock()
    runner.frame = shot('mumu_soil_bare.png'); runner._captured = time.monotonic()
    with pytest.raises(WheatingPrerequisite, match='unlocks at level 7'):
        runner._open_verified_shop(runner.frame, VisualTarget(200,500,200,200,1.))
    client.tap.assert_called_once()
    restart = WheatRestart(runner)
    with pytest.raises(WheatingPrerequisite):
        restart.recover(WheatingPrerequisite('unlocks at level 7'), None)
    assert restart.history == [] and not restart.active


@pytest.mark.parametrize('question', ['Allow Other Game to send you notifications?',
                                    'Allow Hay Day to access your photos?', 'ALLOW'])
def test_other_permissions_and_apps_are_not_automatically_answered(question):
    lines = (replace(LINES[0],text=question), *LINES[1:])
    assert notification_target(shot('mumu_notifications.png'),lines) is None


def notification_runner(tmp_path, frames, package=None):
    notification = shot('mumu_notifications.png')
    client = SimpleNamespace(serial='test', capture=Mock(side_effect=frames), tap=Mock())
    runner = WheatingRunner(client,tmp_path)
    client.foreground_package = (lambda: package or ('com.android.permissioncontroller'
        if runner.frame.png == notification.png else 'com.supercell.hayday'))
    runner.wait = Mock(); runner.publish = Mock()
    reader = Mock(read=Mock(return_value=AdTextObservation(lines=LINES)))
    runner.vision = WheatingVision(text_reader=reader)
    handler = WheatNotifications(runner)
    runner._recovery = SimpleNamespace(notifications=handler)
    return runner, client


def test_two_notification_observations_decline_once_then_verify_return_to_game(tmp_path):
    notification = shot('mumu_notifications.png')
    farm = shot('mumu_soil_bare.png','third')
    runner, client = notification_runner(tmp_path,[notification,replace(notification,captured_at='second'),
                                                   farm,replace(farm,captured_at='fourth')])
    result = runner._capture_raw()
    assert result.png == farm.png
    client.tap.assert_called_once_with(960,658,width=1920,height=1080)
    assert client.capture.call_count == 4


def test_identical_capture_timestamp_cannot_authorize_notification_input(tmp_path):
    notification = shot('mumu_notifications.png')
    runner,client = notification_runner(tmp_path,[notification,notification])
    with pytest.raises(WheatingPrerequisite,match='changed before dismissal'):
        runner._capture_raw()
    client.tap.assert_not_called()


def test_matching_notification_artwork_in_another_app_is_not_clicked(tmp_path):
    runner,client = notification_runner(tmp_path,[shot('mumu_notifications.png')],package='other.app')
    runner._capture_raw()
    client.tap.assert_not_called()
    runner.vision.text.read.assert_not_called()


def test_failed_notification_dismissal_is_never_repeated(tmp_path):
    frames = [shot('mumu_notifications.png',str(i)) for i in range(8)]
    runner,client = notification_runner(tmp_path,frames)
    with pytest.raises(WheatingPrerequisite,match='no repeated tap'):
        runner._capture_raw()
    assert client.tap.call_count == 1


def test_missing_picker_checkpoint_recovers_from_live_soil_instead_of_old_sample(tmp_path):
    worker = WheatCropWorker(SimpleNamespace(serial='test'),Mock(),threading.Event(),Mock(),
                             tmp_path/'fields.json',vision=WheatFarmingVision())
    picker = shot('mumu_soil_picker.png'); bare = shot('mumu_soil_bare.png')
    worker.state['items']['field'] = {'stage':'harvested_needs_replant','operation':'test'}
    worker._close = Mock(return_value=bare)
    worker._replant_after_harvest = Mock()
    with patch('hayday.wheating_crop_resume.rediscover_bare_field',return_value='fresh planting') as rediscover:
        assert worker.work_if_recognized(picker,b'wheat','field') == 'fresh planting'
    worker._close.assert_called_once_with(picker)
    assert rediscover.call_args.args[1] is bare
    worker._replant_after_harvest.assert_not_called()


def test_attempted_plant_keeps_its_reconciliation_path(tmp_path):
    worker = WheatCropWorker(SimpleNamespace(serial='test'),Mock(),threading.Event(),Mock(),
                             tmp_path/'fields.json',vision=WheatFarmingVision())
    worker.state['items']['field'] = {'stage':'plant_attempted','points':[[600,500]]}
    worker._resume_planted = Mock(return_value='reconciled')
    with patch('hayday.wheating_crop_resume.rediscover_bare_field') as rediscover:
        assert worker.work_if_recognized(shot('mumu_soil_picker.png'),b'wheat','field') == 'reconciled'
    rediscover.assert_not_called()


def test_single_page_seed_controls_are_clearable_on_the_way_to_the_shop():
    vision = WheatingVision()
    assert vision.crop_controls(shot('mumu_soil_picker.png'))[0] == 'seed'
    assert vision.crop_controls(shot('mumu_soil_bare.png')) is None


def test_harvest_does_not_follow_gold_board_trim_into_chicken_feed():
    path = WheatingVision().harvest_sweep(shot('mumu_wheat_mature.png'))
    assert len(path) >= 8
    # Diagonal lane ends deliberately overrun foliage by ten pixels.
    assert all(775 <= x <= 1245 and 505 <= y <= 740 for x,y in path)
    assert min(x for x,y in path) < 900  # The left wheat clump stays included.


def test_actual_mumu_shop_title_and_nine_unlocked_slots_are_recognized():
    vision = WheatingVision()
    frame = shot('mumu_shop_overview.png')
    assert vision.shop_is_open(frame)
    view = vision.shop(frame)
    assert view.kind == 'overview'
    assert sum(slot.kind == 'empty' for slot in view.slots) == 8
    assert sum(slot.kind == 'sold' for slot in view.slots) == 1
    assert len(view.slots) == 9  # The friend-unlock control is not a sale slot.


def test_shop_counter_at_mumu_world_scale_is_found_from_a_cold_start():
    vision = WheatingVision()
    found = vision.shop_building(shot('mumu_shop_farm.png'))
    assert found is not None and 100 < found.x < 125 and 485 < found.y < 505


@pytest.mark.parametrize('old_tip', range(4))
def test_one_cached_arrow_cannot_hide_the_rest_of_a_new_crop_fan(old_tip):
    vision = WheatFarmingVision()
    picker = shot('mumu_level7_picker.png')
    tips = vision._matches(picker.png, 'guide_tip', .94)
    vision._control_locations['guide_tip'] = (tips[old_tip],)
    vision._cache.clear()
    assert vision.seed_menu(picker.png)
    assert vision.empty_plot(picker.png) is not None


def test_seed_reserve_counts_the_bare_grid_without_planting(tmp_path):
    client = SimpleNamespace(serial='test', drag_path=Mock())
    run = WheatingRunner(client,tmp_path)
    run.vision = WheatingVision(); run.publish = Mock()
    fields = WheatFields(run)
    bare = shot('mumu_soil_bare.png'); picker = shot('mumu_soil_picker.png')
    plot = fields.worker.vision.empty_plot(picker.png)
    fields._clear = Mock(return_value=bare)
    fields._center_field = lambda f:f
    run.capture = Mock(return_value=replace(bare,captured_at='fresh'))
    fields._select = Mock(return_value=(picker,plot.center,plot,True))
    fields.worker._close = Mock()
    assert fields.bare_seed_reserve() == 9
    client.drag_path.assert_not_called()


def test_isolated_texture_match_does_not_hide_the_complete_mumu_wheat_field():
    frame = shot('mumu_wheat_complete.png')
    vision = WheatingVision()
    textures = vision._ripe_textures(frame)
    assert len(textures) >= 3
    assert all(900 < t.center[0] < 1270 and 485 < t.center[1] < 670 for t in textures)
    assert len(vision.plots(frame, 'ripe')) == 1
    path = vision.harvest_sweep(frame)
    assert len(path) >= 12
    assert min(x for x, y in path) < 940 and max(x for x, y in path) > 1200
    assert all(835 < x < 1295 and 465 < y < 695 for x, y in path)


def test_one_strong_native_arrow_does_not_hide_the_other_picker_arrows():
    picker = shot('mumu_picker_native_arrow.png')
    vision = WheatFarmingVision()
    tips = vision._matches(picker.png, 'guide_tip', .94)
    assert len(tips) == 4 and tips[0].score > .96
    assert vision.seed_menu(picker.png)
    assert vision.empty_plot(picker.png) is not None
    assert vision.full_outline

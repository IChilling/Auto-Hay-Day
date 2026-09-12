"""Nearby wheat sections should regain one growth cycle after partial repairs."""
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_field_group import FieldGroupView, WheatFieldGroup, lattice_pitch, observe_group
from hayday.wheating_fields import WheatFields


def view(ripe=0, growing=0, bare=0):
    return FieldGroupView(tuple((600+i*53, 450, kind) for i, kind in enumerate(
        ['ripe']*ripe+['growing']*growing+['bare']*bare)), (53., 26.5))


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hayday.wheating_field_group.time.monotonic', lambda: clock[0])
    run = SimpleNamespace(state={'wheat_empty': False}, persist=Mock(), publish=Mock(), device_root=tmp_path)
    fields = SimpleNamespace(run=run, next_harvest=0., has_fields=False,
        worker=SimpleNamespace(growth_duration=61., growth_ready_at=161.))
    group = WheatFieldGroup(fields)
    def observe(frame):
        return group.view
    group.observe = observe
    return group, clock, Screenshot(b'farm', 1920, 1080, 'first')


def test_mixed_sections_wait_together_then_release_one_harvest(tracker):
    group, clock, frame = tracker
    group.view = view(38, 13)
    assert group.plan(frame) == 'wait'
    assert group.fields.next_harvest == 163.
    clock[0] = 125.
    group.view = view(45, 6)
    assert group.plan(frame) == 'wait'
    assert group.until == 163.  # Repeated observations cannot restart the growth timer.
    group.view = view(51)
    assert group.plan(frame) is None
    assert not group.active and not group.waiting
    assert group.fields.run.state['field_group']['stage'] == 'synchronized'


def test_partial_repair_uses_latest_planting_deadline_and_keeps_group_identity(tracker):
    group, clock, frame = tracker
    group.view = view(ripe=10, growing=5, bare=7)
    assert group.plan(frame) == 'repair'
    group_identity = group.fields.run.state['field_group']
    assert group.seed_reserve == 7
    clock[0] = 120.
    group.fields.worker.growth_ready_at = 181.
    group.planted()
    assert group.fields.next_harvest == group.until == 183.
    assert group.fields.run.state['field_group'] is group_identity
    group.view = view(growing=22)
    assert group.plan(frame) == 'wait'


def test_bare_plots_with_no_stock_wait_for_combined_harvest_seed_supply(tracker):
    group, _, frame = tracker
    group.fields.run.state['wheat_empty'] = True
    group.view = view(ripe=10, growing=5, bare=7)
    assert group.plan(frame) == 'wait'
    assert group.seed_reserve == group.fields.run.state['seed_reserve'] == 7


def test_obscured_section_does_not_disappear_from_readiness_requirement(tracker):
    group, _, frame = tracker
    group.view = view(ripe=38, growing=13)
    group.plan(frame)
    group.view = view(ripe=38)
    assert group.plan(frame) == 'wait'
    group.view = view(ripe=51)
    assert group.plan(frame) is None


def test_temporary_failed_group_observation_does_not_trigger_partial_harvest(tracker):
    group, clock, frame = tracker
    group.view = view(ripe=38, growing=13)
    group.plan(frame)
    group.view = None
    assert group.plan(frame) == 'wait'
    clock[0] = 195.
    assert group.plan(frame) is None


def test_unexpected_growth_does_not_create_an_unbounded_wait(tracker):
    group, clock, frame = tracker
    group.view = view(ripe=38, growing=13)
    group.plan(frame)
    clock[0] = 195.
    assert group.plan(frame) is None
    assert group.fields.run.state['field_group']['stage'] == 'observation_expired'
    assert not group.active


def test_fully_mature_or_uniform_growing_fields_have_no_extra_wait(tracker):
    group, _, frame = tracker
    for current in (view(ripe=12), view(growing=12), view(bare=12)):
        group.view = current
        assert group.plan(frame) is None
        assert not group.active
    group.fields.run.persist.assert_not_called()


def test_new_run_does_not_reuse_an_old_group_size_or_deadline(tracker):
    group, _, frame = tracker
    group.view = view(ripe=38, growing=13)
    group.plan(frame)
    fresh = WheatFieldGroup(group.fields)
    assert not fresh.active and fresh.view is None


def test_single_tile_repair_keeps_verified_group_reference(tracker):
    group, _, frame = tracker
    points = [(600+(i-j)*53,450+(i+j)*26.5) for i in range(4) for j in range(3)]
    group.reference,group.points,group.proof = frame,points,{'file':'complete.png'}
    group.fields.worker.plant_before = frame
    group.fields.worker.planted_points = [points[0]]
    group.fields.run.state['field_layout'] = {'before':{'file':'repair.png'},'points':[points[0]]}
    group.planted()
    assert group.reference is frame and group.points == points
    assert group.fields.run.state['field_group']['reference'] == {'file':'complete.png'}
    assert len(group.fields.run.state['field_group']['points']) == 12


def test_verified_full_reference_survives_reload_after_small_repair(tracker):
    group, _, frame = tracker
    points = [[600+(i-j)*54, 450+(i+j)*27] for i in range(4) for j in range(3)]
    proof = {'width':1920, 'height':1080, 'file':'complete.png'}
    group.fields._known_before = frame
    group.fields._known_points = points[:1]
    group.fields.run.state['field_group'] = {'version':1, 'points':points, 'reference':proof}
    group.fields.worker._size = (0,0)
    group.fields.worker._saved_frame = Mock(return_value=frame)
    fresh = WheatFieldGroup(group.fields)
    assert fresh.points == points and fresh.reference is frame
    assert group.fields.worker._size == (0,0)
    assert not fresh.active  # Readiness must be observed again, not restored blindly.
    group.fields.worker._saved_frame = Mock(return_value=None)
    assert len(WheatFieldGroup(group.fields).points) == 1


def test_completed_group_uses_fresh_whole_field_route(tracker):
    group, _, frame = tracker
    group.view=view(ripe=38,growing=13)
    group.plan(frame)
    group.view=view(ripe=51)
    group.plan(frame)
    assert group.full_route
    points=[(600+(i-j)*53,450+(i+j)*26.5) for i in range(4) for j in range(3)]
    group.fields.worker.planted_points=points
    group.fields.worker.plant_before=frame
    group.planted()
    assert not group.full_route


@pytest.mark.parametrize('saved',[None,[],{'version':1,'points':[[1,2]],'reference':None}])
def test_malformed_group_metadata_does_not_block_a_fresh_run(tracker,saved):
    group,_,_=tracker
    group.fields.run.state['field_group']=saved
    assert not WheatFieldGroup(group.fields).active


def test_lattice_spacing_is_dynamic_and_rejects_incompatible_rows():
    for dx, dy in ((38,19), (56,28), (71,35.5)):
        points = [(500+(i-j)*dx, 400+(i+j)*dy) for i in range(4) for j in range(3)]
        assert lattice_pitch(points) == pytest.approx((dx,dy))
        points[-1] = (points[-1][0]+dx*.4, points[-1][1])
        assert lattice_pitch(points) is None


@pytest.mark.parametrize('name', ['large_ripe.png', 'small_ripe.png'])
def test_recorded_alternating_sections_form_one_mixed_group(name):
    root = Path(__file__).parent/'fixtures/split_cycle'
    source = Screenshot((root/'partial_seed.png').read_bytes(), 1920,1080,'source')
    frame = Screenshot((root/name).read_bytes(),1920,1080,name)
    vision = WheatFarmingVision()
    plot = vision.empty_plot(source.png)
    points = vision.empty_tiles(source,plot,98)
    pitch = lattice_pitch(points)
    moved = WheatCropWorker._translated_plot(source,frame,points[0],require_visible=False)
    assert moved is not None
    projected = np.asarray(points)+np.subtract(moved,points[0])
    result = observe_group(frame,projected,pitch,vision._soil_texture)
    assert result is not None
    assert result.count('ripe') >= 5 and result.count('growing') >= 5
    assert len(result.cells) <= 51  # Known regression farm, never an application constant.
    assert len({(x,y) for x,y,_ in result.cells}) == len(result.cells)


def test_roads_grass_and_distant_fields_are_not_joined():
    image = np.full((1080,1920,3),(30,180,30),np.uint8)
    dx,dy=53,26.5
    points=[]
    for ox,oy in ((600,450),(1200,650)):
        for i in range(3):
            for j in range(3):
                x,y=round(ox+(i-j)*dx),round(oy+(i+j)*dy)
                polygon=np.int32([(x-dx,y),(x,y-dy),(x+dx,y),(x,y+dy)])
                cv2.fillConvexPoly(image,polygon,(0,220,255))
                if ox == 600: points.append((x,y))
    cv2.rectangle(image,(800,600),(1600,670),(55,120,180),-1)  # Brown road is not soil.
    frame=Screenshot(cv2.imencode('.png',image)[1].tobytes(),1920,1080,'synthetic')
    group=observe_group(frame,points,(dx,dy),WheatFarmingVision()._soil_texture)
    assert group and len(group.cells) == len(points)
    assert all(x < 850 for x,y,kind in group.cells)


def test_field_readiness_does_not_override_synchronization(tracker):
    group, _, frame = tracker
    fields = WheatFields.__new__(WheatFields)
    fields._known_before, fields._known_points = frame, [(600,450)]
    fields._tracked_wheat = Mock(return_value=[(600,450)])
    fields._group = group
    group.view = view(ripe=38,growing=13)
    fields.run = SimpleNamespace(capture=lambda:frame,vision=SimpleNamespace(farm=lambda f:True))
    assert not fields.ready_to_harvest()


def test_waiting_group_does_not_enter_field_retry_or_camera_scan(tmp_path):
    run=SimpleNamespace(client=SimpleNamespace(serial='test'), device_root=tmp_path,
        capture=Mock(),cancel_event=threading.Event(),publish=Mock(),state={},check=Mock())
    fields=WheatFields(run)
    frame=Screenshot(b'farm',1920,1080,'frame')
    fields._clear=lambda:frame
    fields._group.waiting=True
    fields.work_view=Mock(return_value=(0,frame))
    assert fields.pass_all() == 0
    fields.work_view.assert_called_once()


def test_mixed_group_defers_the_actual_harvest_worker(tracker):
    group, _, frame = tracker
    fields = WheatFields.__new__(WheatFields)
    fields.run = group.fields.run
    fields.run.check = Mock()
    fields.run.vision = SimpleNamespace(shop_building=lambda f:None)
    fields._group = group
    group.fields = fields
    fields.worker = SimpleNamespace(growth_duration=61.,work_if_recognized=Mock())
    fields._center_field = lambda f:f
    fields._relocate = Mock()
    fields._select = Mock(side_effect=AssertionError('Must not select partially mature wheat'))
    group.view = view(ripe=38,growing=13)
    assert fields.work_view(frame) == (0,frame)
    fields._select.assert_not_called()
    fields.worker.work_if_recognized.assert_not_called()

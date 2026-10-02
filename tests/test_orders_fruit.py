"""Recorded MuMu harvest failures and durable order-resource confirmation."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hayday.adb import Screenshot
from hayday.farming import FarmingWorker
from hayday.farming_vision import FarmingVision
from hayday.fruit import FruitTarget, FruitVision, FruitWorker
from hayday.resource_vision import VisualTarget

FIXTURES = Path(__file__).parent / "fixtures" / "orders"
REFERENCES = Path(__file__).resolve().parents[1] / "images" / "reference_captures"


@pytest.fixture(scope="module")
def vision():
    return FruitVision()


@pytest.mark.parametrize("name", ["cherry_cloud.png", "cherry_small.png"])
def test_recorded_mumu_cherries_have_supported_target(vision, name):
    result = vision.basket((FIXTURES / name).read_bytes(), desired_icon=vision.cherry_item)
    assert result is not None and result.species == "cherry"
    assert result.target is not None and result.fruit_features
    assert 950 < result.target.center[0] < 1150
    assert 500 < result.target.center[1] < 700


@pytest.mark.parametrize("name", [
    "fruit_cherry_cloud_04.png", "fruit_cherry_cloud_05.png",
    "fruit_cherry_cloud_08.png", "fruit_cherry_small_canopy_before.png",
    "fruit_cherry_partly_harvested_guided.png", "fruit_raspberry_ripe_guided.png",
])
def test_existing_supported_fruit_views(vision, name):
    species = "raspberry" if "raspberry" in name else "cherry"
    icon = vision.raspberry_item if species == "raspberry" else vision.cherry_item
    result = vision.basket((REFERENCES / name).read_bytes(), desired_icon=icon)
    assert result is not None and result.species == species and result.target is not None


def test_cherry_tree_does_not_authorize_wrong_requested_item(vision):
    result = vision.basket((FIXTURES / "cherry_small.png").read_bytes(),
                           desired_icon=vision.raspberry_item)
    assert result is None or result.target is None


@pytest.mark.parametrize("color", ["gray", "blue", "orange"])
def test_neutral_lighting_fit_cannot_recolor_other_objects_as_cherries(vision, color):
    reference = vision.canopy_references[-1][1]
    bgr = reference[:, :, :3].copy()
    if color == "gray":
        bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif color == "blue":
        bgr = bgr[:, :, ::-1].copy()
    else:
        bgr[:, :, 1] = np.maximum(bgr[:, :, 1], bgr[:, :, 2] * .8)
    assert not vision._clusters(bgr, reference, (1.,), lambda: False)


def make_worker(tmp_path, monkeypatch, *, fail_input=False):
    cancel = threading.Event()
    monkeypatch.setattr(cancel, "wait", lambda seconds: cancel.is_set())
    tool = VisualTarget(100, 200, 12, 12, .99)
    target = VisualTarget(300, 190, 30, 20, .98)
    arrow = VisualTarget(200, 210, 30, 30, .99)
    observation = FruitTarget(tool, target, .99, 1., arrow, "cherry", (target,))
    path = tmp_path / "fruit.json"
    inputs = []

    def drag(points, **kwargs):
        saved = json.loads(path.read_text())
        assert saved["items"]["cherry"]["stage"] == "attempted"
        inputs.append((points, kwargs))
        if fail_input:
            raise RuntimeError("Device disconnected during gesture")

    client = SimpleNamespace(serial="mumu-test", drag_path=drag)
    frame = Screenshot(b"recorded-frame", 1920, 1080, "fresh-capture")
    worker = FruitWorker(client, lambda: frame, cancel, lambda message: None, path,
                         vision=SimpleNamespace(basket=lambda *args, **kwargs: observation))
    return worker, frame, inputs


def test_harvest_uses_held_touch_and_waits_for_two_positive_stock_observations(tmp_path, monkeypatch):
    worker, frame, inputs = make_worker(tmp_path, monkeypatch)
    result = worker.work_if_recognized(frame, b"icon", "cherry",
                                      baseline={"status": "missing", "available": 0, "required": 1})
    assert result.details["pending_harvest"]
    assert len(inputs) == 1
    points, options = inputs[0]
    assert points[0] == points[1] and points[-1] != points[0]
    assert options["min_waypoint_ms"] >= 100
    assert options["cancel_event"] is worker.cancel_event
    worker.observe_inventory("cherry", "missing", 0, 1, "unchanged")
    assert worker.state["items"]["cherry"]["stage"] == "attempted"
    worker.work_if_recognized(frame, b"icon", "cherry")
    assert len(inputs) == 1
    worker.observe_inventory("cherry", "fulfilled", 1, 1, "first-gain")
    worker.observe_inventory("cherry", "fulfilled", 1, 1, "first-gain")
    assert worker.state["items"]["cherry"]["stage"] == "attempted"
    worker.observe_inventory("cherry", "fulfilled", 1, 1, "second-gain")
    assert worker.state["items"]["cherry"]["stage"] == "confirmed"


def test_failed_harvest_keeps_intent_and_restart_does_not_retry(tmp_path, monkeypatch):
    worker, frame, inputs = make_worker(tmp_path, monkeypatch, fail_input=True)
    with pytest.raises(RuntimeError, match="disconnected"):
        worker.work_if_recognized(frame, b"icon", "cherry")
    restored = FruitWorker(worker.client, worker.capture, worker.cancel_event,
                           worker.progress, worker.state_path, vision=worker.vision)
    result = restored.work_if_recognized(frame, b"icon", "cherry")
    assert result.details["pending_harvest"]
    assert len(inputs) == 1


def test_fishing_lure_menu_never_starts_crop_work(tmp_path):
    frame = Screenshot((FIXTURES / "fishing_lures.png").read_bytes(), 1920, 1080, "fishing")
    detector = FarmingVision()
    assert detector.seed_menu(frame.png)  # Shared paging and arrow artwork.
    assert detector.empty_plot(frame.png) is None
    worker = FarmingWorker(SimpleNamespace(serial="mumu-test"), lambda: frame,
                           threading.Event(), lambda message: None, tmp_path / "farming.json",
                           vision=detector)
    assert worker.work_if_recognized(frame, b"fish-fillet", "fish") is None
    assert worker.state["items"] == {}
    assert not worker.state_path.exists()


def test_selected_soil_still_enters_planting(tmp_path, monkeypatch):
    plot = VisualTarget(400, 200, 60, 30, .99)
    detector = SimpleNamespace(growing=lambda png: None, harvest=lambda png: None,
                               seed_menu=lambda png: True, empty_plot=lambda png: plot)
    frame = Screenshot(b"seed-picker", 1920, 1080, "seed-picker")
    worker = FarmingWorker(SimpleNamespace(serial="mumu-test"), lambda: frame,
                           threading.Event(), lambda message: None, tmp_path / "farming.json",
                           vision=detector)
    calls = []
    monkeypatch.setattr(worker, "_plant_selected", lambda *args: calls.append(args) or "planting")
    assert worker.work_if_recognized(frame, b"seed", "crop") == "planting"
    assert calls[0][3] == plot

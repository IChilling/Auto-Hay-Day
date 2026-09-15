"""Concurrent device workers must never share input, state, or cancellation."""
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hayday.adb import AdbError
from hayday.emulators import EmulatorConnection
from hayday.storage import AppData, Settings
from hayday.wheating import WheatingResult
from hayday.wheating_fleet import WheatingFleet


def instances(count=2):
    return [EmulatorConnection('MuMu', str(i), f'Farm {i}', Path('adb.exe'), f'127.0.0.1:{16384+32*i}')
            for i in range(count)]


class Client:
    def __init__(self, path, serial, **kwargs):
        self.serial, self.closed = serial, False

    def connect(self, endpoint):
        assert self.serial == endpoint

    def list_devices(self):
        return [SimpleNamespace(serial=self.serial, online=True)]

    def close(self):
        self.closed = True


@pytest.mark.parametrize('count', [2, 3, 4])
def test_workers_start_together_and_keep_their_results_separate(tmp_path, count):
    gate = threading.Barrier(count, timeout=3)
    clients, stops, updates = [], [], []

    class Runner:
        def __init__(self, client, root, cancel_event, progress, **kwargs):
            self.client, self.stop, self.progress = client, cancel_event, progress
            clients.append(client)
            stops.append(cancel_event)

        def run(self):
            gate.wait()
            count = 51 if self.client.serial.endswith('16384') else 9
            self.progress(dict(message='Planted', planted=count, listed=count, collected=1))
            return WheatingResult('limit', 'Done', planted=count, listed=count, collected=1)

    fleet = WheatingFleet(instances(count), tmp_path, client_factory=Client, runner_factory=Runner,
                          progress=updates.append)
    result = fleet.run()
    assert result.success and result.planted == result.listed == 51+9*(count-1)
    assert len({id(s) for s in stops}) == len({id(c) for c in clients}) == count
    assert all(client.closed for client in clients)
    for update in updates:
        for key, state in update['instances'].items():
            assert key.endswith(state['serial'])
            assert state['planted'] in {0, 51 if state['serial'].endswith('16384') else 9}


def test_stopping_one_instance_does_not_stop_the_other(tmp_path):
    gate = threading.Barrier(3, timeout=3)
    stops = {}

    class Runner:
        def __init__(self, client, root, cancel_event, **kwargs):
            self.stop = cancel_event
            stops[client.serial] = cancel_event

        def cancel(self):
            self.stop.set()

        def run(self):
            gate.wait()
            assert self.stop.wait(3)
            return WheatingResult('cancelled', 'Stopped')

    choices = instances()
    fleet = WheatingFleet(choices, tmp_path, client_factory=Client, runner_factory=Runner)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fleet.run)
        gate.wait()
        fleet.cancel(choices[0].key)
        assert stops[choices[0].endpoint].is_set()
        assert not stops[choices[1].endpoint].is_set()
        fleet.cancel()
        assert future.result(timeout=3).success


def test_connection_failure_does_not_cancel_a_healthy_instance(tmp_path):
    class OneOffline(Client):
        def connect(self, endpoint):
            if endpoint.endswith('16384'):
                raise AdbError('offline')

    class Runner:
        def __init__(self, client, root, **kwargs):
            assert client.serial.endswith('16416')

        def run(self):
            return WheatingResult('limit', 'Healthy farm completed', planted=9)

    fleet = WheatingFleet(instances(), tmp_path, client_factory=OneOffline, runner_factory=Runner)
    result = fleet.run()
    assert not result.success and result.planted == 9
    assert sorted(r.status for r in result.results.values()) == ['blocked', 'limit']


def test_duplicate_endpoint_is_rejected_before_connecting(tmp_path):
    first = instances()[0]
    alias = EmulatorConnection('MuMu', 'alias', 'Alias', Path('other/adb.exe'), 'localhost:16384')
    with pytest.raises(AdbError, match='more than once'):
        WheatingFleet([first, alias], tmp_path)


def test_saved_multi_selection_preserves_existing_single_connection(tmp_path):
    data = AppData(tmp_path)
    try:
        selected = tuple(i.key for i in instances())
        data.save_settings(Settings(selected_serial='emulator-5554', wheating_instances=selected))
        loaded = data.load_settings()
        assert loaded.wheating_instances == selected and loaded.selected_serial == 'emulator-5554'
    finally:
        data.close()


def test_pre_cancelled_fleet_never_connects(tmp_path):
    stop = threading.Event()
    stop.set()
    def forbidden(*args, **kwargs):
        pytest.fail('Cancelled worker attempted a connection')
    result = WheatingFleet(instances(), tmp_path, cancel_event=stop, client_factory=forbidden).run()
    assert result.success and result.planted == 0


def test_cancelling_an_inflight_connection_releases_only_its_client(tmp_path):
    connected, released = threading.Event(), threading.Event()

    class Connecting(Client):
        def connect(self, endpoint):
            connected.set()
            assert released.wait(3)
            raise AdbError('Connection cancelled')

        def close(self):
            super().close()
            released.set()

    fleet = WheatingFleet(instances()[:1], tmp_path, client_factory=Connecting)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fleet.run)
        assert connected.wait(3)
        fleet.cancel(instances()[0].key)
        assert future.result(timeout=3).success


def test_service_runs_selected_instances_and_preview_never_changes_input_client(tmp_path, monkeypatch):
    from hayday.adb import Screenshot
    from hayday.services import Services
    shots = {}

    class Runner:
        def __init__(self, client, root, progress, **kwargs):
            self.client, self.progress = client, progress

        def run(self):
            frame = Screenshot(b'preview', 1920, 1080, self.client.serial)
            shots[self.client.serial] = frame
            self.progress(dict(frame=frame, message='Planted', planted=9))
            return WheatingResult('limit', 'Done', planted=9, frame=frame)

    def fleet(*args, **kwargs):
        return WheatingFleet(*args, **kwargs, client_factory=Client, runner_factory=Runner)

    monkeypatch.setattr('hayday.wheating_fleet.WheatingFleet', fleet)
    services = Services(tmp_path)
    try:
        services.instances = instances()
        services.select_wheating_instances([i.key for i in services.instances])
        result = services.start_wheating(max_seconds=2)
        assert result.success and result.planted == 18
        assert services.client is None and not services.connected_serial
        assert len(services.wheating_progress['instances']) == 2
        services.select_wheating_preview(services.instances[1].key)
        assert services.frame is shots[services.instances[1].endpoint]
        assert services.frame_serial == services.instances[1].endpoint
        assert services.client is None and not services.connected_serial
        assert services._wheating_runner is None
    finally:
        services.close()


def test_service_rejects_a_missing_instance_and_selection_changes_during_work(tmp_path):
    from hayday.services import Services
    services = Services(tmp_path)
    try:
        services.instances = instances()
        with pytest.raises(AdbError, match='no longer available'):
            services.select_wheating_instances(['missing'])
        services._wheating_runner = object()
        with pytest.raises(AdbError, match='Stop Wheating'):
            services.select_wheating_instances([services.instances[0].key])
    finally:
        services._wheating_runner = None
        services.close()

"""Emulator boundaries using MuMu Android 15 diagnostic fixtures and fake ADB."""
import json
import re
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hayday.adb import AdbClient, AdbError, Screenshot, parse_touch_device, touch_rotation
from hayday.emulators import (
    EmulatorConnection,
    adb_provider,
    discover_mumu_instances,
    mumu_adb_paths,
    parse_mumu_instances,
)
from hayday.launcher import LaunchRecovery
from hayday.reconnect import ReconnectBlocked

FIXTURES = Path(__file__).parent/'fixtures'
MUMU_TOUCH = (FIXTURES/'mumu_touch.txt').read_text('utf-8')
MUMU_INPUT = (FIXTURES/'mumu_input.txt').read_text('utf-8')
BLUE_TOUCH = '''add device 1: /dev/input/event1
  name: "BlueStacks Virtual Touch"
  events:
    KEY (0001): BTN_TOUCH
    ABS (0003): ABS_MT_SLOT : value 0, min 0, max 9
                ABS_MT_POSITION_X : value 0, min 0, max 32767
                ABS_MT_POSITION_Y : value 0, min 0, max 32767
                ABS_MT_TRACKING_ID : value 0, min 0, max 65535
'''


def fake_adb(tmp_path):
    binary = tmp_path/'adb.exe'
    binary.write_bytes(b'fake')
    return AdbClient(binary, '127.0.0.1:16384')


def test_touch_capabilities_and_bluestacks_mapping():
    mumu = parse_touch_device(MUMU_TOUCH)
    assert (mumu.path, mumu.buttons) == ('/dev/input/event4', (325,))
    assert touch_rotation(MUMU_INPUT, mumu, 1920, 1080) == 1
    assert mumu.coordinates(0, 0, 1920, 1080, 1) == (1080, 0)
    assert mumu.coordinates(1919, 1079, 1920, 1080, 1) == (0, 1920)
    blue = parse_touch_device(BLUE_TOUCH)
    assert blue.buttons == (330,)
    assert blue.coordinates(1919, 1079, 1920, 1080) == (32767, 32767)


@pytest.mark.parametrize('rotation,expected', [(0, (0, 0)), (1, (1080, 0)),
                                             (2, (1080, 1920)), (3, (0, 1920))])
def test_all_display_rotations(rotation, expected):
    touch = parse_touch_device(MUMU_TOUCH)
    assert touch.coordinates(0, 0, 1920, 1080, rotation) == expected


@pytest.mark.parametrize('raw', [MUMU_TOUCH+BLUE_TOUCH,
    MUMU_TOUCH.replace('INPUT_PROP_DIRECT', 'INPUT_PROP_POINTER'),
    MUMU_TOUCH.replace('max 31', 'max 0'), MUMU_TOUCH.replace('max 1080', 'max 0')])
def test_rejects_ambiguous_or_unsupported_touch(raw):
    with pytest.raises(AdbError, match='multitouch'):
        parse_touch_device(raw)


def test_rejects_display_mismatch_and_unrelated_viewport():
    touch = parse_touch_device(MUMU_TOUCH)
    with pytest.raises(AdbError, match='resolution'):
        touch_rotation(MUMU_INPUT, touch, 1280, 720)
    with pytest.raises(AdbError, match='mapping'):
        touch_rotation(MUMU_INPUT.replace('Xiaomi Touchscreen', 'Another device'), touch, 1920, 1080)


@pytest.mark.parametrize('gesture', ['drag', 'pinch'])
@pytest.mark.parametrize('failure', ['none', 'cancel', 'command'])
def test_gestures_encode_rotated_contacts_and_always_release(tmp_path, gesture, failure):
    client = fake_adb(tmp_path)
    cancel = threading.Event()
    calls = []

    def run(args, **kw):
        calls.append((args, kw))
        if args == ['shell', 'getevent', '-pl']:
            return MUMU_TOUCH.encode(), b''
        if args == ['shell', 'dumpsys', 'input']:
            return MUMU_INPUT.encode(), b''
        if args[:2] == ['exec-out', 'dd']:
            return b'\x7fELF\x02\x01', b''
        if '_stdin' in kw:
            if failure == 'cancel':
                cancel.set()
            if failure == 'command':
                raise AdbError('interrupted shell')
        return b'', b''

    client._run = run

    def perform():
        if gesture == 'drag':
            client.drag_path([(0, 0), (1919, 1079)], width=1920, height=1080,
                             duration_ms=250, cancel_event=cancel)
        else:
            client.pinch_zoom_out(width=1920, height=1080, cancel_event=cancel)
    if failure == 'none':
        perform()
    else:
        with pytest.raises(AdbError):
            perform()
    script = next(kw['_stdin'].decode() for _, kw in calls if '_stdin' in kw)
    assert 'exec 9>/dev/input/event4' in script and 'trap ' in script
    encoded = re.search(r"printf '%b' '((?:\\0[0-7]{3})+)' >&9", script)[1]
    raw = bytes(int(value, 8) for value in re.findall(r'\\0([0-7]{3})', encoded))
    events = [values[2:] for values in struct.iter_unpack('<qqHHi', raw)]
    assert (1, 325, 1) in events and (1, 330, 1) not in events
    if gesture == 'drag':
        assert (3, 53, 1080) in events and (3, 54, 0) in events
    release, options = calls[-1]
    assert options['_release'] is True and '/dev/input/event4' in release[-1]


def test_nonwritable_touch_sends_no_events(tmp_path):
    client = fake_adb(tmp_path)
    calls = []

    def run(args, **kw):
        calls.append(args)
        if args == ['shell', 'getevent', '-pl']:
            return MUMU_TOUCH.encode(), b''
        if args == ['shell', 'dumpsys', 'input']:
            return MUMU_INPUT.encode(), b''
        raise AdbError('Permission denied')
    client._run = run
    with pytest.raises(AdbError):
        client.drag_path([(100, 100), (200, 200)], width=1920, height=1080)
    assert all('printf' not in ' '.join(args) for args in calls)


def test_mumu_discovery_custom_install_and_runtime_ports(tmp_path, monkeypatch):
    import hayday.emulators as emulators
    root = tmp_path/'Custom emulator'
    shell = root/'nx_main'
    shell.mkdir(parents=True)
    (shell/'adb.exe').touch()
    (shell/'MuMuManager.exe').touch()
    record = dict(index='2', name='Farm', is_android_started=True,
                  adb_host_ip='127.0.0.1', adb_port=19042)
    monkeypatch.setattr(emulators, 'mumu_install_roots', lambda: [root])
    monkeypatch.setattr(emulators.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({'2': record}).encode()))
    found = discover_mumu_instances()
    assert len(found) == 1 and found[0].endpoint == '127.0.0.1:19042'
    assert found[0].adb_path == (shell/'adb.exe').resolve()
    assert mumu_adb_paths([root]) == [(shell/'adb.exe').resolve()]
    assert adb_provider(found[0].adb_path) == 'MuMu'


@pytest.mark.parametrize('changes', [dict(adb_host_ip='192.168.1.10'), dict(adb_port=0),
    dict(adb_port=65536), dict(adb_port=True), dict(is_android_started=False), dict(index='all')])
def test_mumu_discovery_rejects_unusable_endpoints(changes):
    record = dict(index='0', is_android_started=True, adb_host_ip='127.0.0.1', adb_port=16384)
    record.update(changes)
    assert parse_mumu_instances(json.dumps({'0': record}), Path('adb.exe')) == []


def test_mumu_adb_server_does_not_replace_bluestacks_server(tmp_path, monkeypatch):
    import hayday.adb as adb
    client = fake_adb(tmp_path)
    assert client.server_port == 5037
    (tmp_path/'MuMuManager.exe').touch()
    client = AdbClient(client.executable, client.serial)
    calls = []
    # Popen instances must be hashable for cancellation tracking.
    class Process:
        returncode = 0
        def communicate(self, **kw):
            return b'', b''
    def spawn(args, **kw):
        calls.append(args)
        return Process()
    monkeypatch.setattr(adb.subprocess, 'Popen', spawn)
    client.list_devices()
    client.foreground_package()
    assert all(args[1:3] == ['-P', '5038'] for args in calls)
    assert all('-s' in args for args in calls[1:])


def test_android15_focus_fallback_and_home_resolution(tmp_path):
    client = fake_adb(tmp_path)
    def run(args, **kw):
        if args == ['shell', 'dumpsys', 'window', 'windows']:
            return b'Window list without focus', b''
        if args == ['shell', 'dumpsys', 'window']:
            return b'mCurrentFocus=Window{abc u0 app.lawnchair/.Launcher}', b''
        return b'priority=0\napp.lawnchair/.Launcher\r\n', b''
    client._run = run
    assert client.foreground_package() == client.launcher_package() == 'app.lawnchair'


class HomeClient:
    serial = '127.0.0.1:16384'
    foreground = 'app.lawnchair'
    launches = 0
    def foreground_package(self):
        return self.foreground
    def launcher_package(self):
        return 'app.lawnchair'
    def launch_hay_day(self):
        self.launches += 1
        self.foreground = 'com.supercell.hayday'


def recovery(client):
    ticks = [0]
    def capture():
        ticks[0] += 1
        return Screenshot(b'', 1920, 1080, str(ticks[0]))
    runner = LaunchRecovery(client, capture=capture, wait=lambda s: None)
    runner._returned_game = lambda frame: client.foreground == 'com.supercell.hayday'
    return runner, capture


def test_package_recovery_launches_and_verifies_without_icon_taps():
    client = HomeClient()
    runner, capture = recovery(client)
    frame = capture()
    assert runner.process(frame) is not frame
    assert client.launches == 1
    assert [e['stage'] for e in runner.events] == ['launch_attempted', 'launched']


def test_package_recovery_rejects_unrelated_foreground_and_changed_home():
    client = HomeClient()
    runner, capture = recovery(client)
    client.foreground = 'some.ad.app'
    assert runner._package_home() is None
    client.foreground = 'app.lawnchair'
    def changed():
        client.foreground = 'some.ad.app'
        return capture()
    runner.capture = changed
    with pytest.raises(ReconnectBlocked, match='foreground app changed'):
        runner.process(capture())
    assert client.launches == 0


def test_uncertain_launch_cannot_be_repeated():
    client = HomeClient()
    runner, capture = recovery(client)
    def fail():
        client.launches += 1
        raise AdbError('lost connection')
    client.launch_hay_day = fail
    with pytest.raises(AdbError):
        runner.process(capture())
    with pytest.raises(ReconnectBlocked, match='uncertain'):
        runner.process(capture())
    assert client.launches == 1


def test_package_recovery_preserves_hourly_launch_limit():
    client = HomeClient()
    runner, capture = recovery(client)
    runner._launches = [runner.clock()]*3
    with pytest.raises(ReconnectBlocked, match='launch limit'):
        runner.process(capture())
    assert client.launches == 0


def test_discovery_preserves_saved_manual_endpoint(tmp_path, monkeypatch):
    import hayday.services as services
    from hayday.storage import Settings
    app = services.Services(tmp_path/'data')
    binary = tmp_path/'adb.exe'
    binary.touch()
    app.save_settings(Settings(adb_path=str(binary), endpoint='127.0.0.1:17777'))
    monkeypatch.setattr(services, 'discover_adb_executables', lambda: [])
    monkeypatch.setattr(services, 'discover_bluestacks_instances', lambda: [])
    monkeypatch.setattr(services, 'discover_mumu_instances', lambda path: [
        EmulatorConnection('MuMu', '0', 'Farm', binary, '127.0.0.1:16384')])
    monkeypatch.setattr(app, '_refresh', lambda *a: None)
    try:
        app.discover()
        assert app.settings.endpoint == '127.0.0.1:17777'
    finally:
        app.close()


def test_connect_instance_pairs_binary_endpoint_and_serial(tmp_path, monkeypatch):
    import hayday.services as services
    app = services.Services(tmp_path/'data')
    binary = tmp_path/'mumu/adb.exe'
    selected = EmulatorConnection('MuMu', '0', 'MuMu Farm', binary, '127.0.0.1:19042')
    other = EmulatorConnection('BlueStacks', 'Pie64', 'Blue Farm', tmp_path/'blue.exe', '127.0.0.1:5555')
    app.instances = [other, selected]
    calls = []

    class Client:
        def __init__(self, executable, serial=''):
            self.serial = serial
            calls.append(('client', executable))
        def connect(self, endpoint):
            calls.append(('connect', endpoint))
        def list_devices(self):
            return [services.Device(selected.endpoint, 'device', 'Samsung'),
                    services.Device(other.endpoint, 'device', 'Samsung')]
        def capture(self):
            calls.append(('capture', self.serial))
            return Screenshot(b'', 1920, 1080, 'fresh')
        def close(self):
            pass

    monkeypatch.setattr(services, 'AdbClient', Client)
    try:
        app.connect_instance(selected.key)
        assert app.settings.adb_path == str(binary)
        assert app.settings.endpoint == app.connected_serial == app.frame_serial == selected.endpoint
        assert calls == [('client', str(binary)), ('connect', selected.endpoint), ('capture', selected.endpoint)]
    finally:
        app.close()


def test_no_cross_emulator_popup_when_using_bluestacks_adb(tmp_path):
    from hayday.wheating_popup import WheatHostPopup
    binary = tmp_path/'HD-Adb.exe'
    binary.touch()
    binary.with_name('HD-Player.exe').touch()
    client = AdbClient(binary, 'emulator-5554')
    client._run = lambda *a, **kw: (MUMU_TOUCH.encode(), b'')
    run = SimpleNamespace(client=client, serial=client.serial)
    assert WheatHostPopup(run).desktop is None


def test_bluestacks_gestures_keep_display_aligned_mapping(tmp_path):
    client = fake_adb(tmp_path)
    calls = []
    def run(args, **kw):
        calls.append(args)
        return (BLUE_TOUCH.encode(), b'') if args == ['shell', 'getevent', '-pl'] else (b'', b'')
    client._run = run
    touch, rotation = client._touch_configuration(1920, 1080)
    assert rotation == 0 and touch.name == 'BlueStacks Virtual Touch'
    assert ['shell', 'dumpsys', 'input'] not in calls


def test_legacy_mumu12_discovery(tmp_path, monkeypatch):
    import hayday.emulators as emulators
    root = tmp_path/'MuMuPlayerGlobal-12.0'
    shell = root/'shell'
    shell.mkdir(parents=True)
    (shell/'MuMuManager.exe').touch()
    (shell/'adb.exe').touch()
    monkeypatch.setattr(emulators, 'mumu_install_roots', lambda: [root])
    record = dict(index='3', is_android_started=True, adb_host_ip='127.0.0.1', adb_port=16480)
    monkeypatch.setattr(emulators.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps([record]).encode()))
    found = discover_mumu_instances()
    assert [(i.adb_path, i.endpoint) for i in found] == [(shell/'adb.exe', '127.0.0.1:16480')]

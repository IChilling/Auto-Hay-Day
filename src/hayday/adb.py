"""Local emulator discovery, screenshots, and bounded device input.

ADB server commands are unscoped; every device command uses ``-s``. Touch input
uses explicit screenshot bounds; pinch gestures use discovered multitouch slots.
"""

from __future__ import annotations

import io
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from hayday.emulators import adb_provider, mumu_adb_paths


class AdbError(RuntimeError):
    """An actionable connection, device, command, or screenshot failure."""


@dataclass(frozen=True, slots=True)
class AdbExecutable:
    path: Path
    source: str


@dataclass(frozen=True, slots=True)
class BlueStacksInstance:
    name: str
    display_name: str
    adb_port: int

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.adb_port}"


@dataclass(frozen=True, slots=True)
class Device:
    serial: str
    state: str
    model: str

    @property
    def online(self) -> bool:
        return self.state == "device"


@dataclass(frozen=True, slots=True)
class Screenshot:
    png: bytes
    width: int
    height: int
    captured_at: str


@dataclass(frozen=True, slots=True)
class TouchDevice:
    path: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    name: str = 'BlueStacks Virtual Touch'
    buttons: tuple[int, ...] = (330,)

    def button_events(self, down: bool) -> list[tuple[int, int, int]]:
        return [(1, code, int(down)) for code in self.buttons]

    def coordinates(self, x, y, width, height, rotation=0):
        u, v = x/(width-1), y/(height-1)
        u, v = ((u, v), (1-v, u), (1-u, 1-v), (v, 1-u))[rotation]
        return (round(self.x_min+u*(self.x_max-self.x_min)),
                round(self.y_min+v*(self.y_max-self.y_min)))


def parse_touch_device(output: str) -> TouchDevice:
    """Select one direct type-B touchscreen; reject ambiguous/unsupported input."""
    found = []
    for path, block in re.findall(
        r"add device \d+: (/dev/input/event[0-9]+)\s*\n(.*?)(?=add device \d+:|\Z)",
        output, re.DOTALL,
    ):
        name = re.search(r'name:\s*"([^"\r\n]+)"', block)
        if not name or ('INPUT_PROP_DIRECT' not in block and name[1] != 'BlueStacks Virtual Touch'):
            continue
        axes = {
            name: (int(low), int(high)) for name, low, high in re.findall(
                r"(ABS_MT_[A-Z_]+)\s*:\s*value -?\d+, min (-?\d+), max (-?\d+)", block
            )
        }
        slot = axes.get("ABS_MT_SLOT", (-1, -1))
        x = axes.get("ABS_MT_POSITION_X", (0, 0))
        y = axes.get("ABS_MT_POSITION_Y", (0, 0))
        tracking = axes.get("ABS_MT_TRACKING_ID", (0, 0))
        if (slot[0] != 0 or slot[1] < 1 or not (0 <= x[0] < x[1] <= 2**31-1)
                or not (0 <= y[0] < y[1] <= 2**31-1) or tracking[1] < 2):
            continue
        buttons = tuple(code for label, code in (('BTN_TOUCH', 330), ('BTN_TOOL_FINGER', 325))
                        if re.search(r'\b'+label+r'\b', block))
        found.append(TouchDevice(path, *x, *y, name[1], buttons))
    if len(found) != 1:
        raise AdbError("Could not identify one compatible direct multitouch device for drags and zoom.")
    return found[0]


def touch_rotation(output: str, touch: TouchDevice, width: int, height: int) -> int:
    """Read the selected touch mapper's display, not an unrelated mouse viewport."""
    blocks = re.findall(r'^  Device \d+: ([^\r\n]+)\r?\n(.*?)(?=^  Device |\Z)',
                        output, re.M | re.S)
    matches = [block for name, block in blocks if name.strip() == touch.name]
    if len(matches) != 1:
        raise AdbError('Could not identify the touchscreen display mapping.')
    block = matches[0]
    view = re.search(r'Viewport INTERNAL: displayId=0,.*?orientation=([0-3]), '
                     r'logicalFrame=\[0, 0, (\d+), (\d+)\]', block)
    if not view or (int(view[2]), int(view[3])) != (width, height):
        raise AdbError('Touchscreen display does not match the current screenshot resolution.')
    if 'OrientationAware: false' in block:
        return 0
    if 'OrientationAware: true' not in block:
        raise AdbError('Could not establish touchscreen orientation handling.')
    return int(view[1])


def _registry_directories() -> tuple[list[Path], list[Path]]:
    """Support custom BlueStacks install/data drives without changing registry."""
    try:
        import winreg
    except ImportError:
        return [], []
    installs: list[Path] = []
    data: list[Path] = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for name in ("BlueStacks_nxt", "BlueStacks", "BlueStacks_msi5"):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    with winreg.OpenKey(hive, rf"SOFTWARE\{name}", 0, winreg.KEY_READ | view) as key:
                        for value_name, destination in (
                            ("InstallDir", installs),
                            ("DataDir", data),
                            ("UserDefinedDir", data),
                        ):
                            with suppress(OSError):
                                value, _ = winreg.QueryValueEx(key, value_name)
                                if isinstance(value, str) and value.strip():
                                    destination.append(Path(os.path.expandvars(value.strip().strip('"'))))
                except OSError:
                    continue
    return installs, data


def _known_install_roots() -> list[Path]:
    roots = []
    for variable, fallback in (
        ("PROGRAMFILES", r"C:\Program Files"),
        ("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ("LOCALAPPDATA", ""),
    ):
        value = os.environ.get(variable, fallback if os.name == "nt" else "")
        if value:
            roots.append(Path(value))
    return roots


def discover_adb_executables() -> list[AdbExecutable]:
    """Find installed emulator and Android SDK ADB binaries."""
    candidates: list[tuple[Path, str]] = []
    candidates.extend((path, 'MuMu Player') for path in mumu_adb_paths())
    if sys.platform == "darwin":
        for applications in (Path("/Applications"), Path.home() / "Applications"):
            for name in ("BlueStacks.app", "BlueStacks Air.app"):
                for binary in ("MacOS/hd-adb", "MacOS/adb", "Resources/adb"):
                    candidates.append((applications / name / "Contents" / binary, "BlueStacks Air"))
        candidates.append((Path.home() / "Library/Android/sdk/platform-tools/adb", "Android SDK"))
    registry_installs, _ = _registry_directories()
    for root in registry_installs:
        for binary in ("HD-Adb.exe", "adb.exe"):
            candidates.append((root / binary, "BlueStacks registry"))
    for root in _known_install_roots():
        for install in ("BlueStacks_nxt", "BlueStacks", "BlueStacks_msi5"):
            for binary in ("HD-Adb.exe", "adb.exe"):
                candidates.append((root / install / binary, "BlueStacks install"))
        candidates.append((root / "Android" / "Sdk" / "platform-tools" / "adb.exe", "Android SDK"))
    for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = os.environ.get(variable)
        if value:
            for binary in ("adb.exe", "adb"):
                candidates.append((Path(value) / "platform-tools" / binary, variable))
    for binary in ("HD-Adb.exe", "adb.exe", "adb"):
        on_path = shutil.which(binary)
        if on_path:
            candidates.append((Path(on_path), "System PATH"))

    found: list[AdbExecutable] = []
    seen: set[str] = set()
    for candidate, source in candidates:
        try:
            path = candidate.resolve()
            key = os.path.normcase(str(path))
            if key not in seen and path.is_file():
                seen.add(key)
                found.append(AdbExecutable(path, source))
        except OSError:
            continue
    return found


def _configuration_paths() -> list[Path]:
    _, registry_data = _registry_directories()
    roots = list(registry_data)
    program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData" if os.name == "nt" else "")
    if program_data:
        roots.extend(Path(program_data) / name for name in ("BlueStacks_nxt", "BlueStacks", "BlueStacks_msi5"))
    # Some releases register the parent data directory rather than the product directory.
    paths = [root / "bluestacks.conf" for root in roots]
    paths.extend(root / "BlueStacks_nxt" / "bluestacks.conf" for root in registry_data)
    if sys.platform == "darwin":
        paths.extend(root / "bluestacks.conf" for root in (
            Path("/Users/Shared/Library/Application Support/BlueStacks"),
            Path.home() / "Library/BlueStacks",
        ))
    return list(dict.fromkeys(paths))


def discover_bluestacks_instances() -> list[BlueStacksInstance]:
    """Read configured instance names and local ADB ports, without enabling ADB."""
    pattern = re.compile(
        r'^bst\.instance\.([A-Za-z0-9_-]+)\.(adb_port|status\.adb_port|display_name)\s*=\s*"(.*)"$'
    )
    found: list[BlueStacksInstance] = []
    seen: set[tuple[str, int]] = set()
    for config in _configuration_paths():
        try:
            lines = config.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        values: dict[str, dict[str, str]] = {}
        for line in lines:
            match = pattern.fullmatch(line.strip())
            if match:
                name, field, value = match.groups()
                values.setdefault(name, {})[field] = value
        for name, fields in sorted(values.items()):
            # The runtime port can differ from the configured default.
            ports = [fields.get("status.adb_port", ""), fields.get("adb_port", "")]
            port = next((int(value) for value in ports if value.isascii() and value.isdigit() and 1 <= int(value) <= 65535), None)
            if port is None or (name, port) in seen:
                continue
            seen.add((name, port))
            found.append(BlueStacksInstance(name, fields.get("display_name") or name, port))
    return sorted(found, key=lambda item: (item.display_name.casefold(), item.adb_port))


class AdbClient:
    """Thread-safe process lifetime management for a single device selection.

    ``close`` cancels this client's processes and prevents future commands. It
    never kills the shared ADB server or disconnects other applications.
    """

    def __init__(self, executable: str | Path, serial: str = "", timeout_seconds: float = 10) -> None:
        raw_path = str(executable).strip()
        resolved = shutil.which(raw_path) if raw_path and not Path(raw_path).is_file() else raw_path
        if not resolved or not Path(resolved).is_file():
            raise AdbError(f"ADB executable was not found: {raw_path or '(empty path)'}. Browse to your emulator or Android SDK ADB executable.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise AdbError("ADB timeout must be a positive, finite number of seconds.")
        self.executable = Path(resolved).resolve()
        # BlueStacks ships ADB 1.0.36; current MuMu ships 1.0.41. Separate
        # servers prevent either binary from restarting the other's sessions.
        self.server_port = 5038 if adb_provider(self.executable) == 'MuMu' else 5037
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._input_lock = threading.RLock()
        self._active: set[subprocess.Popen[bytes]] = set()
        self._closed = False
        self._serial = ""
        self._fast_capture_unavailable: set[str] = set()
        self.serial = serial

    @property
    def serial(self) -> str:
        with self._lock:
            return self._serial

    @serial.setter
    def serial(self, value: str) -> None:
        if value and (value.startswith("-") or any(character.isspace() or ord(character) < 32 for character in value)):
            raise AdbError("Device serial is malformed. Select a device from the device list.")
        with self._input_lock, self._lock:
            if self._closed:
                raise AdbError("ADB session is closed. Connect again to start a new session.")
            self._serial = value

    def _run(self, command: list[str], *, device: bool, _release: bool = False,
             _stdin: bytes | None = None, _timeout: float | None = None) -> tuple[bytes, bytes]:
        process: subprocess.Popen[bytes] | None = None
        try:
            # Spawn and registration are atomic relative to close(). Otherwise a
            # command could start after close has already cancelled its snapshot.
            with self._lock:
                if self._closed and not _release:
                    raise AdbError("ADB session is closed. Connect again to start a new session.")
                args = [str(self.executable)]
                if self.server_port != 5037:
                    args.extend(['-P', str(self.server_port)])
                if device:
                    if not self._serial:
                        raise AdbError("Select an online emulator device before using device commands.")
                    args.extend(["-s", self._serial])
                args.extend(command)
                process = subprocess.Popen(
                    args,
                    stdin=subprocess.PIPE if _stdin is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # A release only lifts existing contacts. Repeated close() must
                # not interrupt it and leave a finger pressed on the device.
                if not _release:
                    self._active.add(process)
            try:
                timeout = min(2, self.timeout_seconds) if _release else (
                    self.timeout_seconds if _timeout is None else _timeout)
                if _stdin is None:
                    stdout, stderr = process.communicate(timeout=timeout)
                else:
                    stdout, stderr = process.communicate(input=_stdin, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                with suppress(OSError):
                    process.kill()
                with suppress(OSError, subprocess.TimeoutExpired):
                    process.communicate(timeout=2)
                raise AdbError(
                    f"ADB command timed out after {timeout:g}s. Check that the emulator is running and ADB is enabled."
                ) from exc
            with self._lock:
                if self._closed and not _release:
                    raise AdbError("ADB operation cancelled because the session was closed.")
            if process.returncode:
                detail = (stderr or stdout).decode("utf-8", errors="replace")
                self._raise_error(detail)
            return stdout, stderr
        except OSError as exc:
            raise AdbError(f"Could not run ADB: {exc}") from exc
        finally:
            if process is not None:
                with self._lock:
                    self._active.discard(process)

    @staticmethod
    def _raise_error(detail: str) -> None:
        detail = detail.strip()[:1200]
        lowered = detail.lower()
        if "unauthorized" in lowered:
            raise AdbError("Device is unauthorized. Accept the Android debugging prompt in the emulator, then refresh devices.")
        if "offline" in lowered:
            raise AdbError("Device is offline. Open the emulator and reconnect its local ADB endpoint, then refresh devices.")
        if "no devices" in lowered or "not found" in lowered:
            raise AdbError(f"Selected device is unavailable. Refresh devices and select an online instance. {detail}")
        raise AdbError(detail or "ADB command failed without an error message.")

    def list_devices(self) -> list[Device]:
        stdout, _ = self._run(["devices", "-l"], device=False)
        devices: list[Device] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(("List of devices", "*")):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            serial, state = fields[:2]
            if state not in {"device", "offline", "unauthorized", "recovery", "sideload", "bootloader", "host", "connecting", "no"}:
                continue
            model = next((field[6:].replace("_", " ") for field in fields[2:] if field.startswith("model:")), "")
            devices.append(Device(serial, "no permissions" if state == "no" else state, model))
        return devices

    def connect(self, endpoint: str) -> str:
        match = re.fullmatch(
            r"(127\.0\.0\.1|localhost|\[::1\]):([0-9]{1,5})", endpoint, re.IGNORECASE
        )
        if not match or not 1 <= int(match.group(2)) <= 65535:
            raise AdbError(
                "Use a local emulator endpoint: 127.0.0.1:PORT, localhost:PORT, "
                "or [::1]:PORT (port 1–65535)."
            )
        endpoint = endpoint.lower()
        stdout, stderr = self._run(["connect", endpoint], device=False)
        message = "\n".join(part.decode("utf-8", errors="replace").strip() for part in (stdout, stderr) if part).strip()
        lowered = message.lower()
        if "connected to" not in lowered or any(word in lowered for word in ("cannot connect", "failed", "unable", "refused")):
            self._raise_error(message or f"Could not connect to {endpoint}. Enable Android Debug Bridge in the emulator settings.")
        return message

    def capture(self) -> Screenshot:
        stdout, _ = self._run(["exec-out", "screencap", "-p"], device=True)
        if not stdout.startswith(b"\x89PNG\r\n\x1a\n"):
            raise AdbError("ADB did not return a PNG screenshot. Check the selected emulator device.")
        try:
            with Image.open(io.BytesIO(stdout)) as picture:
                if picture.format != "PNG":
                    raise ValueError("not a PNG")
                width, height = picture.size
                if not (1 <= width <= 16384 and 1 <= height <= 16384 and width * height <= 64_000_000):
                    raise ValueError("unexpected screenshot dimensions")
                picture.verify()
            # verify checks PNG structure; load also verifies the pixel stream.
            with Image.open(io.BytesIO(stdout)) as picture:
                picture.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise AdbError("ADB returned a corrupt or unsupported PNG screenshot. Try capturing again.") from exc
        return Screenshot(stdout, width, height, datetime.now(UTC).isoformat(timespec="microseconds"))

    def capture_fast(self) -> Screenshot:
        """Read lossless RGBA pixels, avoiding the emulator's slow PNG encoder.

        Android screencap writes width, height, pixel format and (on newer
        versions) color space before tightly packed rows. Unknown formats use
        the existing PNG capture, once per client/device capability check.
        """
        serial = self.serial
        if serial in self._fast_capture_unavailable:
            return self.capture()
        raw, _ = self._run(['exec-out', 'screencap'], device=True)
        if serial != self.serial:
            raise AdbError('Device selection changed during screenshot capture.')
        valid = False
        if len(raw) >= 12:
            width, height, pixel_format = struct.unpack('<III', raw[:12])
            if (1 <= width <= 16384 and 1 <= height <= 16384
                    and width*height <= 64_000_000 and pixel_format in (1, 2)):
                offset = len(raw)-width*height*4
                valid = offset == 12 or (offset == 16 and struct.unpack('<I', raw[12:16])[0] in (0, 1))
        if not valid:
            self._fast_capture_unavailable.add(serial)
            return self.capture()
        picture = Image.frombytes('RGBA', (width, height), raw[offset:]).convert('RGB')
        output = io.BytesIO()
        picture.save(output, format='PNG', compress_level=1)
        return Screenshot(output.getvalue(), width, height,
                          datetime.now(UTC).isoformat(timespec='microseconds'))

    def tap(self, x: int, y: int, *, width: int, height: int) -> None:
        """Tap one pixel within the caller's current screenshot bounds.

        Coordinates are never rounded or clamped. Callers must keep the same
        device selection from screenshot capture through this input operation.
        """
        if any(type(value) is not int for value in (width, height)) or width <= 0 or height <= 0:
            raise AdbError("Tap screenshot dimensions must be positive integers.")
        if any(type(value) is not int for value in (x, y)) or not (0 <= x < width and 0 <= y < height):
            raise AdbError("Tap coordinates must be integers within the screenshot bounds.")
        with self._input_lock:
            self._run(["shell", "input", "tap", str(x), str(y)], device=True)

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, *, width: int, height: int,
        duration_ms: int = 250,
    ) -> None:
        """Swipe within one screenshot, with a bounded duration and no hold."""
        self._validate_touch_bounds((x1, y1, x2, y2), width, height)
        if type(duration_ms) is not int or not 50 <= duration_ms <= 2000:
            raise AdbError("Swipe duration must be an integer between 50 and 2000 milliseconds.")
        if (x1, y1) == (x2, y2):
            raise AdbError("Swipe start and end must be different.")
        with self._input_lock:
            self._run(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)], device=True)

    def drag_path(self, points, *, width: int, height: int, duration_ms: int = 4000,
                  cancel_event: threading.Event | None = None,
                  max_step_px: int | None = None, min_waypoint_ms: int = 0,
                  time_factors: tuple[float, ...] | None = None,
                  keep_device_open: bool = True) -> None:
        """Hold one contact through a bounded path, always releasing on exit."""
        if (not isinstance(points, (tuple, list)) or not 2 <= len(points) <= 2049
                or any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in points)):
            raise AdbError('Drag path requires between 2 and 2049 coordinate pairs.')
        self._validate_touch_bounds(tuple(v for p in points for v in p), width, height)
        if type(duration_ms) is not int or not 250 <= duration_ms <= 12000:
            raise AdbError('Drag path duration must be between 250 and 12000 milliseconds.')
        if max_step_px is not None and (type(max_step_px) is not int or not 1 <= max_step_px <= 1024):
            raise AdbError('Drag sample spacing must be an integer between 1 and 1024 pixels.')
        if type(min_waypoint_ms) is not int or not 0 <= min_waypoint_ms <= 250:
            raise AdbError('Drag waypoint hold must be an integer between 0 and 250 milliseconds.')
        if type(keep_device_open) is not bool:
            raise AdbError('Keeping the drag device open must be a boolean.')
        if time_factors is not None and (
                not isinstance(time_factors, (tuple, list)) or len(time_factors) != len(points)-1
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 1 <= value <= 1.10 for value in time_factors)):
            raise AdbError('Drag timing factors must provide one value from 1 to 1.10 per segment.')
        time_factors = tuple(time_factors) if time_factors is not None else (1.,)*(len(points)-1)
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:], strict=False)]
        total = sum(lengths)
        if total < 1:
            raise AdbError('Drag path must move.')
        cancelled = cancel_event if cancel_event is not None else threading.Event()

        def checkpoint():
            if cancelled.is_set():
                raise AdbError('Drag path cancelled.')

        with self._input_lock:
            checkpoint()
            touch, rotation = self._touch_configuration(width, height)
            elf, _ = self._run(['exec-out', 'dd', 'if=/system/bin/sh', 'bs=6', 'count=1'], device=True)
            if len(elf) < 6 or elf[:4] != b'\x7fELF' or elf[4] not in (1, 2) or elf[5] != 1:
                raise AdbError('Cannot establish the Android input-event ABI for a drag.')
            event_format = '<qqHHi' if elf[4] == 2 else '<iiHHi'

            def command(events, *, shared=False):
                data = b''.join(struct.pack(event_format, 0, 0, kind, code, value)
                                for kind, code, value in events)
                encoded = ''.join(f'\\0{byte:03o}' for byte in data)
                destination = '>&9' if shared else f'> {touch.path}'
                return f"printf '%b' '{encoded}' {destination}"

            release_events = [(3, 47, 0), (3, 57, -1), *touch.button_events(False), (0, 0, 0)]
            release = command(release_events)
            script = ['set -e', f'trap {shlex.quote(release)} EXIT HUP INT TERM']
            if keep_device_open:
                # BlueStacks charges substantial latency for each evdev open.
                # Keep one descriptor through the gesture, including its trap
                # release. The separate finally release remains a fallback.
                script.append(f'exec 9>{touch.path}')
                shared_release = command(release_events, shared=True)
                script.append(f'trap {shlex.quote(shared_release)} EXIT HUP INT TERM')
            samples = [points[0]]
            delays = []
            sample_factors = []
            waypoints = set()
            for start, end, length, factor in zip(points[:-1], points[1:], lengths, time_factors, strict=True):
                steps = max(1, math.ceil(duration_ms*length/total/35))
                if max_step_px is not None:
                    # Fast crop drags need spatial coverage as well as timing:
                    # a 35 ms interval can otherwise jump over an entire plot.
                    steps = max(steps, math.ceil(length/max_step_px))
                if len(samples)+steps > 4096:
                    raise AdbError('Drag path requires too many input samples.')
                for index in range(1, steps+1):
                    samples.append(tuple(round(a+(b-a)*index/steps) for a, b in zip(start, end, strict=True)))
                    delays.append(duration_ms*length/total/steps/1000)
                    sample_factors.append(factor)
                waypoints.add(len(samples)-1)
            for index, (px, py) in enumerate(samples):
                events = [(3, 47, 0)]
                if index == 0:
                    events.extend([*touch.button_events(True), (3, 57, 1)])
                x, y = touch.coordinates(px, py, width, height, rotation)
                events.extend(((3, 53, x), (3, 54, y), (0, 0, 0)))
                script.append(command(events, shared=keep_device_open))
                delay = delays[index] if index < len(delays) else 0.
                if index in waypoints:
                    # Let the game consume each target before the next move,
                    # including the final target before lifting the contact.
                    delay = max(delay, min_waypoint_ms/1000)
                # Timing varies without moving samples or reducing target holds.
                delay *= sample_factors[min(index, len(sample_factors)-1)]
                if index < len(delays) or delay:
                    script.append(f'sleep {delay:.6f}')
            checkpoint()
            try:
                self._run(['shell', 'sh'], device=True,
                          _stdin=('\n'.join(script)+'\n').encode('ascii'),
                          _timeout=max(self.timeout_seconds, 8+(duration_ms/1000
                                       + (len(points)-1)*min_waypoint_ms/1000)*max(time_factors)
                                       + (len(samples)*.04 if max_step_px is not None else 0)))
                checkpoint()
            finally:
                self._run(['shell', release], device=True, _release=True)

    def is_bluestacks(self) -> bool:
        raw, _ = self._run(['shell', 'getevent', '-pl'], device=True)
        return bool(re.search(rb'name:\s*"BlueStacks Virtual Touch"', raw))

    def _touch_configuration(self, width, height):
        raw, _ = self._run(['shell', 'getevent', '-pl'], device=True)
        touch = parse_touch_device(raw.decode('utf-8', errors='replace'))
        # Preserve BlueStacks' display-aligned virtual device mapping.
        rotation = 0
        if touch.name != 'BlueStacks Virtual Touch':
            raw, _ = self._run(['shell', 'dumpsys', 'input'], device=True)
            rotation = touch_rotation(raw.decode('utf-8', errors='replace'), touch, width, height)
        self._run(['shell', f'test -w {touch.path} || '
                   '{ echo "Touch input is not writable by ADB; check emulator input permissions." >&2; exit 1; }'],
                  device=True)
        return touch, rotation

    def foreground_package(self) -> str | None:
        """Read the focused Android window; unknown focus never implies a crash."""
        raw, _ = self._run(['shell', 'dumpsys', 'window', 'windows'], device=True)
        pattern = r'mCurrentFocus=Window\{[^\r\n}]*\bu\d+\s+([A-Za-z0-9_.]+)/'
        matches = re.findall(pattern, raw.decode('utf-8', errors='replace'))
        if not matches:
            # Android 15 omits display focus from the `windows` subcommand.
            raw, _ = self._run(['shell', 'dumpsys', 'window'], device=True)
            matches = re.findall(pattern, raw.decode('utf-8', errors='replace'))
        return matches[0] if len(matches) == 1 else None

    def launcher_package(self) -> str | None:
        """Resolve Android's default HOME activity without opening it."""
        raw, _ = self._run(['shell', 'cmd', 'package', 'resolve-activity', '--brief',
                           '-a', 'android.intent.action.MAIN', '-c', 'android.intent.category.HOME'],
                          device=True)
        matches = re.findall(r'^([A-Za-z0-9_.]+)/[A-Za-z0-9_.$]+\s*$',
                             raw.decode('utf-8', errors='replace'), re.M)
        return matches[0] if len(matches) == 1 and matches[0] != 'android' else None

    def back(self) -> None:
        """Send exactly one Android Back action; callers verify the current UI."""
        with self._input_lock:
            self._run(["shell", "input", "keyevent", "4"], device=True)

    def home(self) -> None:
        """Open the selected emulator's launcher without changing Windows focus."""
        with self._input_lock:
            self._run(['shell', 'input', 'keyevent', '3'], device=True)

    def force_stop_hay_day(self) -> None:
        """Stop only Hay Day through Android's activity manager; retain app data."""
        with self._input_lock:
            self._run(['shell', 'am', 'force-stop', 'com.supercell.hayday'], device=True)

    def launch_hay_day(self) -> None:
        """Launch Hay Day directly inside the selected emulator without changing Windows focus."""
        with self._input_lock:
            self._run(['shell', 'monkey', '-p', 'com.supercell.hayday',
                       '-c', 'android.intent.category.LAUNCHER', '1'], device=True)

    def hay_day_running(self) -> bool:
        """Verify the package's processes; command failures never mean stopped."""
        raw, _ = self._run(['shell', 'ps', '-A', '-o', 'NAME'], device=True)
        lines = raw.decode('utf-8', errors='strict').splitlines()
        if not lines or lines[0].strip() != 'NAME':
            raise AdbError('Could not verify whether the Hay Day process stopped.')
        return any(name.strip() == 'com.supercell.hayday'
                   or name.strip().startswith('com.supercell.hayday:') for name in lines[1:])

    @staticmethod
    def _validate_touch_bounds(points: tuple[int, ...], width: int, height: int) -> None:
        if (type(width) is not int or type(height) is not int
                or not 2 <= width <= 16384 or not 2 <= height <= 16384):
            raise AdbError("Gesture screenshot dimensions must be integers between 2 and 16384.")
        if any(type(value) is not int or not 0 <= value < (width if i % 2 == 0 else height)
               for i, value in enumerate(points)):
            raise AdbError("Gesture coordinates must be integers within the screenshot bounds.")

    def pinch_zoom_out(
        self, *, width: int, height: int, duration_ms: int = 350,
        cancel_event: threading.Event | None = None,
        center: tuple[int, int] | None = None, span: int | None = None,
    ) -> None:
        """Zoom out with a synchronized two-finger inward pinch through ADB."""
        self._pinch_zoom(width=width, height=height, duration_ms=duration_ms,
                         cancel_event=cancel_event, inward=True, center=center, span=span)

    def pinch_zoom_in(
        self, *, width: int, height: int, duration_ms: int = 350,
        cancel_event: threading.Event | None = None,
        center: tuple[int, int] | None = None, span: int | None = None,
    ) -> None:
        """Zoom in with a synchronized two-finger outward pinch through ADB."""
        self._pinch_zoom(width=width, height=height, duration_ms=duration_ms,
                         cancel_event=cancel_event, inward=False, center=center, span=span)

    def _pinch_zoom(
        self, *, width: int, height: int, duration_ms: int,
        cancel_event: threading.Event | None, inward: bool,
        center: tuple[int, int] | None, span: int | None,
    ) -> None:
        """Generate paired movement frames, releasing both contacts on every exit.

        BlueStacks' Down arrow maps to this host-side gesture. Android's Down
        keyevent is unrelated. No rooting or device permission changes are made.
        The only commands allowed after close are release events for these slots.
        """
        self._validate_touch_bounds((), width, height)
        if type(duration_ms) is not int or not 100 <= duration_ms <= 1500:
            raise AdbError("Pinch duration must be an integer between 100 and 1500 milliseconds.")
        center = (width//2, height//2) if center is None else center
        span = round(width*.44) if span is None else span
        if (type(center) is not tuple or len(center) != 2 or any(type(value) is not int for value in center)
                or type(span) is not int or not 16 <= span < width):
            raise AdbError("Pinch center and span must be valid integer screenshot coordinates.")
        cx, cy = center
        self._validate_touch_bounds((cx-(span+1)//2, cy, cx+(span+1)//2, cy), width, height)
        cancelled = cancel_event if cancel_event is not None else threading.Event()

        def checkpoint():
            if cancelled.is_set():
                raise AdbError("Pinch cancelled.")

        with self._input_lock:
            checkpoint()
            touch, rotation = self._touch_configuration(width, height)
            checkpoint()
            # input_event timeval fields follow the writing process ABI. The
            # shell's ELF class/data identify it without guessing from host CPU.
            elf, _ = self._run(["exec-out", "dd", "if=/system/bin/sh", "bs=6", "count=1"], device=True)
            if len(elf) < 6 or elf[:4] != b"\x7fELF" or elf[4] not in (1, 2) or elf[5] != 1:
                raise AdbError("Cannot establish the Android input-event ABI for a synchronized pinch.")
            event_format = "<qqHHi" if elf[4] == 2 else "<iiHHi"
            checkpoint()
            # All synchronized frames share one shell and one open evdev
            # descriptor, avoiding BlueStacks' per-open input latency.
            def command(events, *, shared=False):
                raw = b"".join(struct.pack(event_format, 0, 0, kind, code, value) for kind, code, value in events)
                encoded = "".join(f"\\0{byte:03o}" for byte in raw)
                destination = '>&9' if shared else f'> {touch.path}'
                return f"printf '%b' '{encoded}' {destination}"

            release = [(3, 47, 0), (3, 57, -1), (3, 47, 1), (3, 57, -1),
                       *touch.button_events(False), (0, 0, 0)]
            release_command = command(release)
            active = False
            try:
                script = ["set -e", f"trap {shlex.quote(release_command)} EXIT HUP INT TERM"]
                script.extend((f'exec 9>{touch.path}',
                               f'trap {shlex.quote(command(release, shared=True))} EXIT HUP INT TERM'))
                for step in range(13):
                    amount = step/12 if inward else 1-step/12
                    radius = span/2 * (1-(1-.05/.22)*amount)
                    events = []
                    if step == 0:
                        events.extend(touch.button_events(True))
                    for slot, pixel in enumerate((cx-radius, cx+radius)):
                        x, y = touch.coordinates(pixel, cy, width, height, rotation)
                        events.append((3, 47, slot))
                        if step == 0:
                            events.append((3, 57, slot+1))
                        events.extend(((3, 53, x), (3, 54, y)))
                    events.append((0, 0, 0))
                    script.append(command(events, shared=True))
                    if step < 12:
                        script.append(f"sleep {duration_ms/12000:.6f}")
                checkpoint()
                active = True  # A failed command may already have sent DOWN.
                # Older BlueStacks ADB servers limit service command names to
                # 4KiB. Stream the generated script over stdin instead.
                self._run(["shell", "sh"], device=True, _stdin=("\n".join(script)+"\n").encode("ascii"))
                checkpoint()
            finally:
                if active:
                    # Release even if command completion became uncertain or the
                    # owning runner closed this client during cancellation.
                    self._run(["shell", release_command], device=True, _release=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            processes = tuple(self._active)
        # The command's worker owns communicate/reaping. Calling communicate
        # concurrently here could race its pipe reads; killing unblocks it.
        for process in processes:
            with suppress(OSError):
                if process.poll() is None:
                    process.kill()

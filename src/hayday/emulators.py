"""Local emulator installation and instance discovery; never starts a player."""
from __future__ import annotations

import json
import os
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmulatorConnection:
    emulator: str
    name: str
    display_name: str
    adb_path: Path
    endpoint: str

    @property
    def key(self) -> str:
        return f"{self.adb_path}|{self.endpoint}"


def adb_provider(path: Path) -> str:
    """Identify the installation, not the guest selected through its ADB server."""
    if path.name.casefold() == 'hd-adb.exe' and path.with_name('HD-Player.exe').is_file():
        return 'BlueStacks'
    for root in list(path.parents)[:4]:
        if ((root/'MuMuManager.exe').is_file()
                or (root/'nx_main/MuMuManager.exe').is_file()):
            return 'MuMu'
    return 'ADB'


def mumu_install_roots() -> list[Path]:
    if os.name != 'nt':
        return []
    import winreg

    roots = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            with suppress(OSError):
                with winreg.OpenKey(hive, r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
                                    0, winreg.KEY_READ | view) as parent:
                    for index in range(winreg.QueryInfoKey(parent)[0]):
                        with suppress(OSError):
                            with winreg.OpenKey(parent, winreg.EnumKey(parent, index)) as key:
                                name = winreg.QueryValueEx(key, 'DisplayName')[0]
                                if isinstance(name, str) and 'mumu' in name.casefold():
                                    value = winreg.QueryValueEx(key, 'InstallLocation')[0]
                                    if isinstance(value, str) and value.strip():
                                        roots.append(Path(os.path.expandvars(value.strip().strip('"'))))
    for variable in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'):
        if base := os.environ.get(variable):
            for name in ('MuMuPlayer', 'MuMuPlayerGlobal-12.0', 'MuMuPlayer-12.0'):
                roots.append(Path(base)/'Netease'/name)
            roots.append(Path(base)/'Nemu')
    return list(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))


def mumu_adb_paths(roots: list[Path] | None = None) -> list[Path]:
    paths = []
    for root in mumu_install_roots() if roots is None else roots:
        candidates = [root/'nx_main/adb.exe', root/'shell/adb.exe',
                      root/'adb.exe', root/'vmonitor/bin/adb_server.exe']
        candidates.extend(sorted(root.glob('nx_device/*/shell/adb.exe')))
        paths.extend(path.resolve() for path in candidates if path.is_file())
    return list(dict.fromkeys(paths))


def parse_mumu_instances(raw: str, adb_path: Path) -> list[EmulatorConnection]:
    data = json.loads(raw.lstrip('\ufeff'))
    if isinstance(data, dict) and 'index' in data:
        records = [data]
    elif isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        return []
    result = []
    for item in records:
        if not isinstance(item, dict) or item.get('is_android_started') is not True:
            continue
        host, port = item.get('adb_host_ip'), str(item.get('adb_port', ''))
        if host not in ('127.0.0.1', 'localhost', '::1', '[::1]'):
            continue
        if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
            continue
        index = str(item.get('index', ''))
        if not index.isascii() or not index.isdigit():
            continue
        host = '[::1]' if host == '::1' else host
        name = str(item.get('name') or f'Instance {index}')
        result.append(EmulatorConnection('MuMu', index, name, adb_path, f'{host}:{int(port)}'))
    return result


def discover_mumu_instances(adb_path: Path | None = None) -> list[EmulatorConnection]:
    roots = mumu_install_roots()
    if adb_path is not None:
        # A manually chosen binary also supports custom or portable installs.
        roots = list(dict.fromkeys([*list(adb_path.resolve().parents)[:4], *roots]))
    found = {}
    managers = set()
    for root in roots:
        for manager in (root/'nx_main/MuMuManager.exe', root/'shell/MuMuManager.exe',
                        root/'MuMuManager.exe'):
            if not manager.is_file() or manager.resolve() in managers:
                continue
            managers.add(manager.resolve())
            binary = manager.with_name('adb.exe')
            if not binary.is_file():
                continue
            try:
                response = subprocess.run([str(manager), 'info', '-v', 'all'],
                    stdin=subprocess.DEVNULL, capture_output=True,
                    timeout=5, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                instances = parse_mumu_instances(response.stdout.decode('utf-8-sig'), binary.resolve())
            except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
                continue
            for instance in instances:
                found[instance.key] = instance
    return sorted(found.values(), key=lambda item: (item.display_name.casefold(), item.endpoint))

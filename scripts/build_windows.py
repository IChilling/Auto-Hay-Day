"""Build the Windows x64 single-file app and its portable distribution folder."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
import tomllib
from pathlib import Path


def package_release(root, client):
    output = root/'dist/Hay Day Automation'
    documentation = output/'Documentation'
    documentation.mkdir(parents=True, exist_ok=True)
    notices = ['THIRD-PARTY COMPONENTS\nOriginal license notices from bundled Python distributions.\n']
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        if distribution.metadata['Name'] == 'hayday-automation':
            continue
        notices.append(f'\n{"="*76}\n{distribution.metadata["Name"]} {distribution.version}\n')
        for file in distribution.files or []:
            if '.dist-info/' not in str(file) or not any(
                    word in str(file).lower() for word in ('license', 'copying', 'notice')):
                continue
            target = distribution.locate_file(file)
            if target.is_file():
                notices.append(f'\n{file}\n{target.read_text(encoding="utf-8", errors="replace")}\n')
    python_license = Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.is_file():
        notices.append('\nPYTHON\n'+python_license.read_text('utf-8'))
    flutter_notice = client/'data/flutter_assets/NOTICES.Z'
    if flutter_notice.is_file():
        import gzip
        notices.append('\nFLET DESKTOP / FLUTTER\n'+gzip.decompress(
            flutter_notice.read_bytes()).decode('utf-8'))
    (documentation/'Third-party notices.txt').write_text('\n'.join(notices), encoding='utf-8')
    shutil.copy2(root/'docs/WINDOWS_QUICK_START.txt', output/'READ ME.txt')
    shutil.copy2(root/'docs/WINDOWS_RELEASE_NOTES.txt', documentation/'Release notes.txt')
    shutil.copy2(root/'scripts/check_windows.cmd', output/'Check installation.cmd')
    archive = shutil.make_archive(str(root/'dist/Hay Day Automation-Windows-x64'), 'zip',
                                  root_dir=root/'dist', base_dir='Hay Day Automation')
    print(f'Portable package: {archive}')


def main():
    if sys.platform != 'win32':
        raise SystemExit('Build the Windows executable on Windows.')
    import comtypes.client
    import flet_desktop
    import PyInstaller.__main__
    from PyInstaller.utils.hooks import collect_submodules

    root = Path(__file__).resolve().parents[1]
    generated = root/'build/windows-package'
    output = root/'dist/Hay Day Automation'
    generated.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    version = tomllib.loads((root/'pyproject.toml').read_text('utf-8'))['project']['version']
    client = Path(flet_desktop.ensure_client_cached())/'flet'
    if not (client/'flet.exe').is_file():
        raise SystemExit('The matching Flet desktop client was not found.')
    if sys.argv[1:] == ['--package-only']:
        if not (output/'Hay Day Automation.exe').is_file():
            raise SystemExit('Build the executable before packaging it.')
        package_release(root, client)
        return 0
    comtypes.client.GetModule('UIAutomationCore.dll')

    assets = root/'src/hayday/assets'
    manifest = {'version': version, 'assets': {
        file.relative_to(assets).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in assets.rglob('*') if file.is_file() and '__pycache__' not in file.parts
    }}
    (generated/'bundle-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    version_tuple = tuple(int(part) for part in version.split('.'))+(0,)
    metadata = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={version_tuple}, prodvers={version_tuple}, mask=0x3f,
                   flags=0, OS=0x40004, fileType=0x1, subtype=0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('FileDescription', 'Hay Day Automation'),
    StringStruct('FileVersion', '{version}.0'),
    StringStruct('ProductName', 'Hay Day Automation'),
    StringStruct('ProductVersion', '{version}'),
    StringStruct('OriginalFilename', 'Hay Day Automation.exe')
  ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])
"""
    (generated/'version.txt').write_text(metadata, encoding='utf-8')
    arguments = [
        str(root/'scripts/windows_entry.py'), '--name', 'Hay Day Automation',
        '--onefile', '--windowed', '--noconfirm', '--noupx', '--clean',
        '--paths', str(root/'src'), '--paths', str(root/'scripts'),
        '--distpath', str(output), '--workpath', str(generated/'work'),
        '--specpath', str(generated), '--version-file', str(generated/'version.txt'),
        '--add-data', f'{assets}:hayday/assets',
        '--add-data', f'{client}:flet-client',
        '--add-data', f'{generated / "bundle-manifest.json"}:.',
        '--copy-metadata', 'flet', '--copy-metadata', 'flet-desktop',
        '--collect-data', 'flet',
        '--exclude-module', 'pytest', '--exclude-module', 'tkinter',
    ]
    hidden = ['flet_desktop', 'comtypes.gen.UIAutomationClient']+collect_submodules('winrt')
    for module in hidden:
        arguments.extend(['--hidden-import', module])
    PyInstaller.__main__.run(arguments)
    executable = output/'Hay Day Automation.exe'
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    (generated/'build-result.json').write_text(json.dumps({
        'version': version, 'executable': str(executable), 'bytes': executable.stat().st_size,
        'sha256': digest, 'assets': len(manifest['assets'])
    }, indent=2), encoding='utf-8')
    print(f'Built {executable} ({executable.stat().st_size/1024/1024:.1f} MiB)')
    print(f'SHA256: {digest}')
    package_release(root, client)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

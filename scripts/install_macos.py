"""Second stage of the single-file installer. Runs using its private Python."""
from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DOWNLOADS = {
    # URLs and SHA-256 values checked against Homebrew's cask metadata and
    # the official Flet GitHub release on 2026-09-12.
    "adb": (
        "https://dl.google.com/android/repository/platform-tools_r37.0.1-darwin.zip",
        "ee39ad5967e95c2a07f04dbcbde96b1a0c916ba376096db5d2f498b7727a5d1d",
    ),
    "bluestacks": (
        "https://ak-build.bluestacks.com/public/app-player/mac/nxt_mac2/5.21.790.7505/"
        "426aabd288b04f58a8200b751e9bcbcf/BlueStacksInstaller_5.21.790.7505.pkg",
        "137accd707aa21028abf57412820d803fe550b07a1d5110ada291b80e10877aa",
    ),
    "flet": (
        "https://github.com/flet-dev/flet/releases/download/v0.86.0/flet-macos.tar.gz",
        "0860c643e1c5fcc7135cb2be24837ec2cf3c5526f7f1b4fe34dc56829730e80f",
    ),
}
BUNDLE_ID = "local.hayday.automation"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(kind: str, cache: Path) -> Path:
    url, digest = DOWNLOADS[kind]
    target = cache / url.rsplit("/", 1)[1]
    if target.is_file() and sha256(target) == digest:
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    print(f"Downloading {kind}...", flush=True)
    run("/usr/bin/curl", "--fail", "--location", "--show-error", "--retry", "3",
        "--connect-timeout", "30", "--max-time", "3600", "--proto", "=https",
        "--proto-redir", "=https", "--output", partial, url)
    if sha256(partial) != digest:
        partial.unlink()
        raise RuntimeError(f"The {kind} download failed its SHA-256 check. Run setup again.")
    partial.replace(target)
    return target


def find_bluestacks() -> Path | None:
    for parent in (Path("/Applications"), Path.home() / "Applications"):
        for name in ("BlueStacks.app", "BlueStacks Air.app"):
            app = parent / name
            if (app / "Contents/Info.plist").is_file():
                return app
    return None


def install_bluestacks(cache: Path) -> Path:
    existing = find_bluestacks()
    if existing:
        print(f"Using installed BlueStacks: {existing}", flush=True)
        return existing
    package = download("bluestacks", cache)
    run("/usr/sbin/pkgutil", "--check-signature", package)
    print("Installing BlueStacks Air. macOS may request your administrator password.", flush=True)
    # Let macOS handle password entry. It is never read or stored by this script.
    if sys.stdin.isatty():
        run("/usr/bin/sudo", "/usr/sbin/installer", "-pkg", package, "-target", "/")
    else:
        command = shlex.join(["/usr/sbin/installer", "-pkg", str(package), "-target", "/"])
        literal = command.replace("\\", "\\\\").replace('"', '\\"')
        run("/usr/bin/osascript", "-e", f'do shell script "{literal}" with administrator privileges')
    installed = find_bluestacks()
    if not installed:
        raise RuntimeError("BlueStacks installer finished but the app was not found in Applications.")
    return installed


def write_launcher(root: Path) -> Path:
    applications = Path.home() / "Applications"
    applications.mkdir(exist_ok=True)
    app = applications / "Hay Day Automation.app"
    if app.exists():
        try:
            metadata = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"An unrelated file exists at {app}. Rename it and rerun setup.") from exc
        if metadata.get("CFBundleIdentifier") != BUNDLE_ID:
            raise RuntimeError(f"An unrelated app exists at {app}. Rename it and rerun setup.")
    staging = Path(tempfile.mkdtemp(prefix=".hayday-app-", dir=applications))
    contents = staging / "Contents"
    executable = contents / "MacOS/HayDayAutomation"
    executable.parent.mkdir(parents=True)
    launcher = '''#!/bin/bash
set -euo pipefail
task_root=ROOT_PLACEHOLDER
export HAYDAY_DATA_DIR="$task_root/data"
export PATH="$task_root/current/tools/platform-tools:/usr/bin:/bin:/usr/sbin:/sbin"
export FLET_DESKTOP_FLAVOR=full
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX || true
mkdir -p "$HAYDAY_DATA_DIR/logs"
cd -P "$task_root/current"
exec "$PWD/.venv/bin/python" -I "$PWD/app.py" >> "$HAYDAY_DATA_DIR/logs/launcher.log" 2>&1
'''.replace("ROOT_PLACEHOLDER", shlex.quote(str(root)))
    executable.write_text(launcher, encoding="utf-8", newline="\n")
    executable.chmod(0o755)
    (contents / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": "Hay Day Automation",
        "CFBundleDisplayName": "Hay Day Automation",
        "CFBundleExecutable": "HayDayAutomation",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.1",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    }))
    backup = applications / f".hayday-app-backup-{time.time_ns()}"
    if app.exists():
        app.rename(backup)
    try:
        staging.rename(app)
    except OSError:
        if backup.exists():
            backup.rename(app)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return app


def install(args):
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 16 * 1024**3:
        raise RuntimeError("At least 16 GB of free disk space is required for setup and BlueStacks.")
    memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
    if memory < 8 * 1024**3:
        raise RuntimeError("BlueStacks Air requires at least 8 GB RAM.")
    source = Path(__file__).resolve().parents[1]
    releases = root / "releases"
    releases.mkdir(exist_ok=True)
    release = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=releases))
    shutil.copytree(source, release, dirs_exist_ok=True)
    cache = root / "downloads"
    cache.mkdir(exist_ok=True)
    python = release / ".venv/bin/python"
    print("Installing the project and its macOS libraries...", flush=True)
    run(args.uv, "venv", "--python", sys.executable, release / ".venv")
    run(args.uv, "pip", "install", "--python", python, "--only-binary", ":all:",
        "--require-hashes", "--index-url", "https://pypi.org/simple",
        "-r", release / "requirements-macos.lock")
    run(args.uv, "pip", "check", "--python", python)

    adb_archive = download("adb", cache)
    tools = release / "tools"
    tools.mkdir()
    run("/usr/bin/ditto", "-x", "-k", adb_archive, tools)
    adb = tools / "platform-tools/adb"
    adb.chmod(0o755)
    run(adb, "version", timeout=30)
    flet_archive = download("flet", cache)
    # Provision Flet through its own extraction API, from the verified archive.
    environment = os.environ.copy()
    environment["FLET_CLIENT_URL"] = flet_archive.as_uri()
    environment["FLET_DESKTOP_FLAVOR"] = "full"
    environment["PATH"] = str(adb.parent) + os.pathsep + environment.get("PATH", "")
    environment["HAYDAY_DATA_DIR"] = str(root / "data")
    print("Checking native desktop, image recognition and Mac OCR...", flush=True)
    run(python, "-I", release / "scripts/check_macos.py", env=environment, timeout=180)
    run(python, "-I", release / "scripts/ui_smoke.py", "--hidden", "--seconds-per-route", "0.2",
        "--report", release / "ui-check.json", env=environment, timeout=120)

    bluestacks = install_bluestacks(cache)
    current = root / "current"
    if current.exists() and not current.is_symlink():
        raise RuntimeError(f"The installation link is occupied by a folder: {current}")
    previous = os.readlink(current) if current.is_symlink() else None
    pending = root / f".current-{time.time_ns()}"
    pending.symlink_to(release, target_is_directory=True)
    pending.replace(current)
    try:
        app = write_launcher(root)
    except Exception:
        if previous is None:
            current.unlink()
        else:
            pending.symlink_to(previous, target_is_directory=True)
            pending.replace(current)
        raise

    instructions = root / "Getting Started.txt"
    instructions.write_text(
        "Hay Day Automation for macOS\n\n"
        "Application setup is complete. Finish these first-run game steps:\n"
        "1. Open BlueStacks Air and complete its first-run prompts.\n"
        "2. Sign in to the game store and install Hay Day. Open the game and sign in.\n"
        "3. Enable Android Debug Bridge (ADB) in BlueStacks Settings > Advanced.\n"
        "4. Open Hay Day Automation > BlueStacks, refresh and connect the local endpoint.\n"
        "   Use the ADB port displayed in BlueStacks if it differs from 5555.\n"
        "5. Select your device before starting an automation feature.\n\n"
        "You can reopen the tool from your home folder > Applications > Hay Day Automation.\n"
        "Apple Vision supplies local OCR. Windows desktop popup handling does not apply on Mac.\n"
        "Game gestures depend on BlueStacks' Android touch device; end-to-end game automation\n"
        "must be verified on the target Mac. The installer tests the desktop UI, assets and OCR.\n\n"
        f"App: {app}\nData: {root / 'data'}\nSetup log: {args.log}\n"
        "Run the same setup file again to repair or update. Existing app data is preserved.\n",
        encoding="utf-8",
    )
    print(f"\nApplication setup complete: {app}", flush=True)
    print(f"First-run game instructions: {instructions}", flush=True)
    print("Sign in, install Hay Day in BlueStacks, and enable ADB to finish game setup.", flush=True)
    if not args.no_launch:
        for target in (instructions, bluestacks, app):
            result = subprocess.run(["/usr/bin/open", str(target)], check=False)
            if result.returncode:
                print(f"Could not open automatically; open this manually: {target}")


def main():
    if sys.platform != "darwin":
        raise SystemExit("This installer requires macOS.")
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()
    # OS-managed lock releases even after a crash; no stale PID files to clear.
    with (args.root / "setup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another Hay Day setup is running. Let it finish first.") from None
        try:
            install(args)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise SystemExit(f"Setup failed: {exc}\nSee: {args.log}") from exc


if __name__ == "__main__":
    main()

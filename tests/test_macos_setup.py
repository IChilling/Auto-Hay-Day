"""Cross-platform checks for the installer; native UI/OCR are checked on the Mac."""
import base64
import hashlib
import importlib.util
import io
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from hayday.ad_text import _recognize_macos, parse_ad_text

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_managed_python_policy_with_uv(tmp_path):
    """Exercise uv's real argument parser with the bootstrap's actual policy."""
    uv = shutil.which("uv")
    if not uv:
        local = ROOT / ".setup/macos-build-tools/uv.exe"
        if local.is_file():
            uv = str(local)
    if not uv:
        pytest.skip("uv is required for the bootstrap CLI integration check")
    # Simulate a caller with a conflicting Python preference in their shell.
    environment = os.environ.copy()
    environment.update(UV_PYTHON_PREFERENCE="only-system", UV_NO_MANAGED_PYTHON="1")
    bootstrap = (ROOT / "scripts/macos-bootstrap.sh").read_text(encoding="utf-8")
    for line in bootstrap.splitlines():
        if not line.startswith(("unset ", "export ")):
            continue
        words = shlex.split(line, comments=True)
        if words and words[0] == "unset":
            for name in words[1:]:
                if name == "||":
                    break
                environment.pop(name, None)
        elif words and words[0] == "export":
            name, value = words[1].split("=", 1)
            if name.startswith("UV_") and "$" not in value:
                environment[name] = value
    environment["UV_PYTHON_INSTALL_DIR"] = str(tmp_path / "empty-managed-installations")
    environment["UV_CACHE_DIR"] = str(tmp_path / "cache")
    find_line = next(line for line in bootstrap.splitlines() if line.startswith("task_python="))
    command = shlex.split(find_line.removeprefix('task_python="$(').removesuffix(')"'))
    result = subprocess.run(
        [uv, *command[1:], "--no-python-downloads"], env=environment,
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    # There is deliberately no private Python: lookup must run, reject system
    # Python, and report the missing runtime rather than a CLI-option conflict.
    assert result.returncode != 0
    assert "No interpreter found" in result.stderr, result.stderr
    assert "managed installations" in result.stderr
    assert "cannot be used with" not in result.stderr


def test_single_file_contains_complete_clean_project(tmp_path):
    build = script("build_macos_setup")
    output = tmp_path / "Setup with spaces.command"
    count, digest = build.build(ROOT, output)
    first = output.read_bytes()
    header, encoded = first.split(b"\n__HAYDAY_PAYLOAD_BELOW__\n", 1)
    assert b"\r" not in header
    payload = base64.b64decode(encoded)
    assert hashlib.sha256(payload).hexdigest() == digest
    assert digest.encode() in header
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = archive.getnames()
        assert len(names) == count
        assert "scripts/install_macos.py" in names
        assert "src/hayday/ui.py" in names
        assert "requirements-macos.lock" in names
        assert not any(name.startswith(("images/reference_captures/", "images/wheating/"))
                       for name in names)
        for item in archive:
            assert item.isfile()
            assert not item.name.startswith("/")
            assert not {"..", ".setup", ".venv", "__pycache__", "tests"} & set(Path(item.name).parts)
            assert archive.extractfile(item).read() == (ROOT / item.name).read_bytes()
    build.build(ROOT, output)
    assert output.read_bytes() == first


def test_download_rejects_corruption_and_reuses_verified_cache(tmp_path, monkeypatch):
    installer = script("install_macos")
    good = b"verified test download"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setitem(installer.DOWNLOADS, "adb", ("https://example.invalid/adb.zip", digest))

    def corrupt_download(*args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"corrupt")

    monkeypatch.setattr(installer, "run", corrupt_download)
    with pytest.raises(RuntimeError, match="SHA-256"):
        installer.download("adb", tmp_path)
    assert not (tmp_path / "adb.zip").exists()
    assert not (tmp_path / "adb.zip.partial").exists()
    (tmp_path / "adb.zip").write_bytes(good)
    assert installer.download("adb", tmp_path).read_bytes() == good


def test_mac_ocr_converts_bottom_left_coordinates(monkeypatch):
    candidate = SimpleNamespace(string=lambda: "Reward granted")
    box = SimpleNamespace(origin=SimpleNamespace(x=.1, y=.2),
                          size=SimpleNamespace(width=.5, height=.3))
    observation = SimpleNamespace(topCandidates_=lambda n: [candidate], boundingBox=lambda: box)
    request = SimpleNamespace(
        setRecognitionLevel_=lambda value: None,
        setRecognitionLanguages_=lambda value: None,
        setUsesLanguageCorrection_=lambda value: None,
        results=lambda: [observation],
    )
    handler = SimpleNamespace(performRequests_error_=lambda requests, error: (True, None))
    monkeypatch.setitem(sys.modules, "Vision", SimpleNamespace(
        VNRecognizeTextRequest=SimpleNamespace(alloc=lambda: SimpleNamespace(init=lambda: request)),
        VNRequestTextRecognitionLevelAccurate=1,
        VNImageRequestHandler=SimpleNamespace(alloc=lambda: SimpleNamespace(
            initWithData_options_=lambda data, options: handler)),
    ))
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(
        NSData=SimpleNamespace(dataWithBytes_length_=lambda data, length: data)))
    stream = io.BytesIO()
    Image.new("RGB", (100, 100)).save(stream, "PNG")
    result = _recognize_macos(stream.getvalue())
    assert result["lines"] == [{"text": "Reward granted", "bounds": [10, 50, 50, 30]}]
    # The Mac result feeds the existing reward/countdown parser without special cases.
    from hayday.ad_text import AdTextLine

    parsed = parse_ad_text(tuple(AdTextLine(item["text"], tuple(item["bounds"]))
                                 for item in result["lines"]))
    assert parsed.reward_granted


def test_mac_adb_discovery_and_config(tmp_path, monkeypatch):
    import hayday.adb as adb

    home = tmp_path / "Mac User"
    binary = home / "Applications/BlueStacks.app/Contents/MacOS/hd-adb"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test executable")
    config = home / "Library/BlueStacks/bluestacks.conf"
    config.parent.mkdir(parents=True)
    config.write_text('bst.instance.Air.adb_port="5565"\nbst.instance.Air.display_name="Mac farm"\n')
    monkeypatch.setattr(adb.sys, "platform", "darwin")
    monkeypatch.setattr(adb.Path, "home", lambda: home)
    assert any(item.path == binary.resolve() for item in adb.discover_adb_executables())
    assert any(item.endpoint == "127.0.0.1:5565" and item.display_name == "Mac farm"
               for item in adb.discover_bluestacks_instances())

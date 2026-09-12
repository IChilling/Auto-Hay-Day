"""Build one portable .command containing all application sources and assets.

Run with Python on Windows, macOS or Linux. No Mac is needed to assemble it.
The recipient needs only the generated file and an internet connection.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Setup Hay Day macOS.command"


def payload_files(root: Path):
    fixed = (
        "app.py", "pyproject.toml", "README (IMPORTANT INSTRUCTIONS).txt",
        "requirements-macos.txt", "requirements-macos.lock", "README-macOS.md",
        "scripts/install_macos.py", "scripts/check_macos.py", "scripts/ui_smoke.py",
    )
    for name in fixed:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked installer input: {path}")
        yield path
    for folder in (root / "src/hayday", root / "images"):
        for path in sorted(folder.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            # Training screenshots are not runtime references. Wheating loads
            # its curated references from src/hayday/assets/wheating instead.
            if path.relative_to(root).parts[:2] in {
                ("images", "reference_captures"), ("images", "wheating"),
            }:
                continue
            if path.is_symlink():
                raise ValueError(f"Do not bundle linked files: {path}")
            if path.suffix in {".py", ".png", ".jpg", ".jpeg", ".json", ".svg", ".ico"}:
                yield path


def build(root: Path = ROOT, output: Path = OUTPUT) -> tuple[int, str]:
    files = list(payload_files(root))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                data = path.read_bytes()
                entry = tarfile.TarInfo(path.relative_to(root).as_posix())
                entry.size = len(data)
                entry.mode = 0o644
                entry.mtime = 0
                archive.addfile(entry, io.BytesIO(data))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    header = (root / "scripts/macos-bootstrap.sh").read_text(encoding="utf-8")
    assert header.count("__PAYLOAD_SHA256__") == 1
    assert header.endswith("__HAYDAY_PAYLOAD_BELOW__\n")
    header = header.replace("__PAYLOAD_SHA256__", digest)
    output.write_bytes(header.encode("utf-8") + base64.encodebytes(payload))
    output.chmod(0o755)
    print(f"Created {output} ({output.stat().st_size:,} bytes, {len(files)} embedded files)")
    print(f"SHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
    return len(files), digest


if __name__ == "__main__":
    build()

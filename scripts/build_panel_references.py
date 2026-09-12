r"""Reproduce pixel-exact masked panel references from a 1920 x 1080 capture.

Example (from the project directory)::

    .venv\Scripts\python.exe scripts/build_panel_references.py \
        images/reference_captures/truck_order_panel.png --sync-package

The coordinates describe the observed truck-order panel style. Inspect and revise
them when the game's panel artwork/layout changes; do not resize a new capture
to force it to fit these coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE_SIZE = (1920, 1080)
SPECS = {
    "header_edge": {
        "box": (487, 30, 704, 191),
        "polygon": [(2, 2), (217, 7), (217, 157), (16, 155), (10, 142), (2, 17)],
    },
    "close_button": {
        "box": (1672, 33, 1847, 210),
        "ellipse": (4, 3, 171, 174),
    },
    "board_corner": {
        "box": (313, 918, 435, 1028),
        "polygon": [(0, 0), (40, 0), (40, 56), (53, 70), (122, 72),
                    (122, 110), (35, 110), (16, 103), (3, 84)],
    },
    "detail_corner": {
        "box": (1189, 208, 1300, 278),
        "polygon": [(45, 0), (111, 0), (111, 35), (53, 35), (37, 40),
                    (29, 52), (27, 70), (0, 70), (0, 41), (10, 20), (26, 5)],
    },
}


def build(source: Path, output: Path, sync_package: bool = False) -> list[Path]:
    """Crop original RGB pixels and add hard alpha masks; write only panel_* files."""
    with Image.open(source) as opened:
        if opened.size != REFERENCE_SIZE:
            raise ValueError(
                f"Expected an unscaled {REFERENCE_SIZE[0]} x {REFERENCE_SIZE[1]} "
                f"panel screenshot; got {opened.width} x {opened.height}."
            )
        screenshot = opened.convert("RGB")
    manifest = {
        "reference_size": list(REFERENCE_SIZE),
        "source_name": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provenance": (
            "Static truck-order UI crops from a local screenshot. Original RGB pixels; "
            "binary alpha masks exclude changing text, order contents, and farm scenery. "
            "Crop boxes are [left, top, right, bottom] in the original capture. Mask "
            "coordinates are relative to each crop. Full farm captures are not packaged."
        ),
        "features": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, spec in SPECS.items():
        crop = screenshot.crop(spec["box"]).convert("RGBA")
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        if "polygon" in spec:
            draw.polygon(spec["polygon"], fill=255)
            mask_spec = {"polygon": spec["polygon"]}
        else:
            draw.ellipse(spec["ellipse"], fill=255)
            mask_spec = {"ellipse": spec["ellipse"]}
        crop.putalpha(mask)
        path = output / f"panel_{name}.png"
        crop.save(path)
        paths.append(path)
        manifest["features"][name] = {
            "file": path.name, "box": list(spec["box"]), "mask": mask_spec,
        }
    manifest_path = output / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    paths.append(manifest_path)
    if sync_package:
        bundled = PROJECT / "src" / "hayday" / "assets" / "identifiers"
        bundled.mkdir(parents=True, exist_ok=True)
        for path in paths:
            destination = bundled / path.name
            if destination.resolve() != path.resolve():
                shutil.copy2(path, destination)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original 1920 x 1080 truck-order panel PNG")
    parser.add_argument("--output", type=Path, default=PROJECT / "images")
    parser.add_argument(
        "--sync-package", action="store_true", help="Also refresh installed-build reference copies"
    )
    args = parser.parse_args()
    for path in build(args.source, args.output, args.sync_package):
        print(path)


if __name__ == "__main__":
    main()

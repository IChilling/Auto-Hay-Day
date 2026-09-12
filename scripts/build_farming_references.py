"""Extract masked, original farming UI pixels from authorized local captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
SPECS = {
    "sickle": ("farming_corn_guided.png", (660, 535, 836, 684)),
    "guide_arrow": ("farming_corn_guided.png", (706, 606, 839, 719)),
    "guide_tip": ("farming_corn_guided.png", (764, 650, 839, 719)),
    "empty_plot": ("farming_seed_menu.png", (860, 645, 984, 736)),
    "page_previous": ("farming_seed_menu.png", (274, 775, 397, 895)),
    "page_next": ("farming_seed_menu.png", (611, 775, 735, 895)),
    "growth_bar": ("farming_corn_planted.png", (644, 811, 1260, 918)),
}


def build(captures: Path, output: Path, sync_package: bool = False) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"reference_size": [1920, 1080], "features": {}, "provenance": (
        "Original RGB pixels from authorized local BlueStacks captures. Alpha is the inclusion "
        "mask; transparent pixels are ignored. No generated artwork or fixed farm coordinates. "
        "The guide-relative harvest offset is UI geometry, validated against an observed white "
        "selected-crop highlight. Crop type, counts, timers and premium buttons are excluded."
    )}
    paths = []
    for name, (filename, box) in SPECS.items():
        source = captures / filename
        with Image.open(source) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"{filename} must be the original 1920 x 1080 capture")
            crop = image.convert("RGBA").crop(box)
        rgb = np.asarray(crop)[:, :, :3]
        mask = Image.new("L", crop.size)
        draw = ImageDraw.Draw(mask)
        if name == "sickle":
            # Only metal blade and handle; the arrow and translucent guide are excluded.
            draw.polygon([(15, 112), (67, 86), (83, 102), (29, 143), (11, 141), (6, 133)], fill=255)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            metal = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 130)).astype(np.uint8)*255
            metal[100:, :] = 0
            mask = Image.fromarray(np.maximum(np.asarray(mask), metal))
        elif name == "guide_arrow":
            draw.polygon([(36, 4), (86, 51), (99, 62), (109, 46), (129, 106),
                          (56, 94), (71, 81), (6, 71), (9, 63), (25, 51), (28, 18)], fill=255)
        elif name == "guide_tip":
            draw.polygon([(51, 2), (71, 62), (0, 50), (14, 37), (7, 29), (41, 18)], fill=255)
        elif name == "empty_plot":
            # The selected tile is partly occluded by neighboring plants. Keep the
            # visible white rim and soil only; exclude those changing neighbors.
            draw.polygon([(8, 55), (25, 45), (36, 40), (45, 31), (56, 30),
                          (69, 20), (78, 19), (91, 6), (108, 13), (113, 50),
                          (95, 65), (58, 81), (28, 71)], fill=255)
            r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
            soil = (r > 90) & (r-g > 25) & (g > 35) & (g < 155) & (b < 105)
            white = (r > 210) & (g > 210) & (b > 190) & (r-b < 45)
            mask = Image.fromarray(np.asarray(mask) * (soil | white).astype(np.uint8))
        elif name.startswith("page_"):
            draw.ellipse((5, 4, crop.width-5, crop.height-6), fill=255)
        elif name == "growth_bar":
            # Left cap plus top/bottom rims. The blue fill, text, countdown and
            # neighboring lightning/diamond purchase control can all change.
            draw.rounded_rectangle((3, 4, crop.width+50, crop.height-9), radius=42, fill=255)
            draw.rectangle((65, 15, crop.width, 85), fill=0)
            draw.rectangle((164, 0, 368, 26), fill=0)
            draw.rectangle((164, 63, 485, crop.height), fill=0)
        crop.putalpha(mask)
        path = output / f"{name}.png"
        crop.save(path)
        paths.append(path)
        manifest["features"][name] = {
            "file": path.name, "source": filename, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "box": list(box), "included_pixels": int(np.count_nonzero(np.asarray(mask))),
        }
    manifest["harvest_geometry"] = {"sickle_box": list(SPECS["sickle"][1]),
        "tool_point": [737, 613], "harvest_point": [937, 676], "target_point": [918, 702],
        "target_box": [882, 665, 954, 739], "white_highlight_box": [874, 582, 981, 706]}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    paths.append(path)
    if sync_package:
        packaged = PROJECT / "src/hayday/assets/farming"
        packaged.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, packaged/path.name)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=PROJECT / "images/reference_captures")
    parser.add_argument("--output", type=Path, default=PROJECT / "images/farming")
    parser.add_argument("--sync-package", action="store_true")
    args = parser.parse_args()
    build(args.captures, args.output, args.sync_package)

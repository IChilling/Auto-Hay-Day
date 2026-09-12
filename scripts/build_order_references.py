"""Rebuild order reader references from locally captured, unscaled game screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
SPECS = {
    "item_check": ("orders_double_offer.png", (1478, 414, 1527, 464), "green"),
    "ticket_check": ("orders_double_offer.png", (1089, 656, 1161, 732), "green"),
    "ticket_check_tilted": ("orders_ready.png", (1130, 620, 1193, 697), "green"),
    "quantity_slash": ("orders_ready.png", (1289, 529, 1311, 559), "brown"),
    "send": ("orders_ready.png", (1386, 856, 1658, 1004), "send"),
    "double_offer": ("orders_ready.png", (1656, 802, 1806, 957), "ellipse"),
    "bonus_symbols": ("orders_double_offer.png", (1637, 414, 1830, 488), "rectangle"),
    "bonus_send": ("orders_double_offer.png", (1582, 680, 1880, 774), "bonus"),
    "sent_stamp": ("orders_after_double_send.png", (956, 548, 1095, 686), "ellipse"),
    "ad_close": ("orders_ad_finished.png", (1861, 19, 1902, 59), "ellipse"),
    "navigation": ("resource_icecream_prompt_fresh.png", (1747, 272, 1804, 322), "rectangle"),
}


def build(captures: Path, output: Path, sync_package: bool = False) -> list[Path]:
    manifest = {"reference_size": [1920, 1080], "features": {}, "provenance": (
        "Original RGB pixels from local BlueStacks screenshots captured during authorized "
        "order testing. No user explanation images are used. Alpha masks exclude changing "
        "backgrounds. Coordinates are [left, top, right, bottom] in source pixels."
    )}
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (filename, box, mode) in SPECS.items():
        source = captures / filename
        with Image.open(source) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"{filename} must be an unscaled 1920 x 1080 capture")
            crop = image.convert("RGBA").crop(box)
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode in {"green", "brown"}:
            rgb = np.asarray(crop)[:, :, :3].astype(np.int16)
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            if mode == "green":
                keep = (g-r > 12) & (g-b > 16) & (g > 75)
            else:
                keep = (r > 70) & (r < 160) & (g > 35) & (g < 125) & (r-g > 18)
            mask = Image.fromarray(keep.astype(np.uint8)*255)
        elif mode == "ellipse":
            draw.ellipse((2, 2, crop.width-3, crop.height-3), fill=255)
        elif mode == "send":
            draw.polygon([(55, 0), (272, 0), (272, 148), (65, 148),
                          (20, 127), (0, 87), (5, 47), (28, 14)], fill=255)
        elif mode == "bonus":
            draw.rectangle((0, 0, 90, 93), fill=255)
            draw.rectangle((90, 23, 298, 84), fill=255)
            # Recognition depends on the TV/action frame, not localized SEND text.
            draw.rectangle((142, 25, 230, 73), fill=0)
        else:
            draw.rectangle((0, 0, crop.width, crop.height), fill=255)
        crop.putalpha(mask)
        path = output / f"order_{name}.png"
        crop.save(path)
        paths.append(path)
        manifest["features"][name] = {
            "file": path.name, "source": filename, "source_sha256":
            hashlib.sha256(source.read_bytes()).hexdigest(), "box": list(box), "mask": mode,
        }
    manifest_path = output / "order_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    paths.append(manifest_path)
    if sync_package:
        package = PROJECT / "src" / "hayday" / "assets" / "identifiers"
        package.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, package / path.name)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=PROJECT / "images" / "reference_captures")
    parser.add_argument("--output", type=Path, default=PROJECT / "images")
    parser.add_argument("--sync-package", action="store_true")
    args = parser.parse_args()
    for path in build(args.captures, args.output, args.sync_package):
        print(path)


if __name__ == "__main__":
    main()

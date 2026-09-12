"""Build original-pixel Raspberry orchard references from the saved live shop view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "images" / "reference_captures"
SOURCE = CAPTURES / "orchard_raspberry_catalog.png"
INVALID_SOURCE = CAPTURES / "orchard_invalid_placement.png"
ALTERNATE_SOURCE = CAPTURES / "orchard_raspberry_catalog_alternate.png"
ANIMATED_SOURCE = CAPTURES / "orchard_raspberry_catalog_animation.png"
PLACEMENT_SOURCES = (
    (CAPTURES / "orchard_raspberry_placement_before.png",
     "Stable catalog frame immediately before the one live purchase swipe."),
    (CAPTURES / "orchard_raspberry_placement_after_1.png",
     "First fresh frame showing the new bush and the exact 220-coin charge."),
    (CAPTURES / "orchard_raspberry_placement_after_2.png",
     "Second fresh frame showing the same newly placed bush."),
)
DESIRED = ROOT / "images" / "learned_items" / "0d9c7223250cd3328d65.png"
BOARD_SOURCE = CAPTURES / 'orchard_raspberry_board_order.png'
BOARD_ICON_BOX = [1245, 455, 1322, 516]
OUTPUTS = (ROOT / "images" / "orchard", ROOT / "src" / "hayday" / "assets" / "orchard")

# All boxes are LTRB coordinates in SOURCE. Runtime code derives positions from
# newly matched controls; these coordinates are provenance only.
BOXES = {
    "shop_button": [917, 870, 1075, 1075],
    "header": [309, 10, 716, 91],
    "raspberry_card": [337, 210, 525, 330],
    "card_arrow": [467, 300, 565, 395],
    "tree_tab": [918, 555, 1055, 699],
}
INVALID_BOX = [1021, 122, 1752, 217]
ROTATE_BOX = [1111, 205, 1302, 361]


def _encoded(crop: np.ndarray, mask: np.ndarray) -> bytes:
    if crop.shape[:2] != mask.shape or np.count_nonzero(mask) < 40:
        raise ValueError("Orchard reference masks must retain enough original pixels.")
    ok, encoded = cv2.imencode(".png", np.dstack((crop, mask.astype(np.uint8))))
    if not ok:
        raise OSError("OpenCV could not encode an orchard reference.")
    return encoded.tobytes()


def _shop_button(crop: np.ndarray) -> tuple[np.ndarray, str]:
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.ellipse(mask, (79, 103), (77, 96), 0, 0, 360, 255, -1)
    return mask, "Binary ellipse includes the observed round shop control and excludes farm scenery."


def _header(crop: np.ndarray) -> tuple[np.ndarray, str]:
    low = crop.min(axis=2)
    high = crop.max(axis=2)
    mask = ((high < 105) | ((low > 205) & (high - low < 55))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, "Original near-black outline and neutral white title pixels; binary 3x3 close."


def _raspberry_card(crop: np.ndarray) -> tuple[np.ndarray, str]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 28) & (hue <= 96) & (saturation >= 65) & (value >= 45)
    berry = (((hue <= 14) | (hue >= 145)) & (saturation >= 90) & (value >= 65))
    mask = (green | berry).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    # The lower-right corner contains part of the generic drag arrow.
    mask[91:, 135:] = 0
    return mask, (
        "Original green foliage and saturated pink Raspberry pixels; binary 3x3 close/dilate; "
        "the overlapping generic arrow corner is excluded."
    )


def _card_arrow(crop: np.ndarray) -> tuple[np.ndarray, str]:
    mask = np.zeros(crop.shape[:2], np.uint8)
    polygon = np.array([
        [15, 7], [39, 31], [49, 17], [91, 88], [86, 94], [23, 75], [32, 59],
        [3, 58], [6, 43], [0, 35],
    ])
    cv2.fillPoly(mask, [polygon], 255)
    return mask, "Binary polygon follows the observed white/gold catalog drag arrow."


def _tree_tab(crop: np.ndarray) -> tuple[np.ndarray, str]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((hue <= 28) & (saturation >= 90) & (value >= 45) & (value <= 205)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros_like(mask)
    # Keep only sizeable dark-orange components near the tree glyph, excluding
    # the drawer border and farm pixels along the crop edges.
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area >= 20 and 15 < x < 122 and 15 < y < 133:
            kept[labels == index] = 255
    kept = cv2.dilate(kept, np.ones((3, 3), np.uint8))
    return kept, "Dark-orange tree glyph components inside the selected category tab; binary dilation."


def _desired_raspberry(image: np.ndarray) -> tuple[np.ndarray, str]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    berry = (((hue <= 14) | (hue >= 145)) & (saturation >= 80) & (value >= 55))
    green = (hue >= 28) & (hue <= 96) & (saturation >= 65) & (value >= 40)
    mask = (berry | green).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask, "Original Raspberry artwork pixels selected by saturated berry/leaf hue; binary close/dilate."


MASKS = {
    "shop_button": _shop_button,
    "header": _header,
    "raspberry_card": _raspberry_card,
    "card_arrow": _card_arrow,
    "tree_tab": _tree_tab,
}


def build() -> None:
    source_png = SOURCE.read_bytes()
    source = cv2.imdecode(np.frombuffer(source_png, np.uint8), cv2.IMREAD_COLOR)
    if source is None or source.shape != (1080, 1920, 3):
        raise ValueError("The original 1920x1080 Raspberry shop capture is required.")
    desired_png = DESIRED.read_bytes()
    desired = cv2.imdecode(np.frombuffer(desired_png, np.uint8), cv2.IMREAD_COLOR)
    if desired is None:
        raise ValueError("The learned Raspberry item crop is required.")
    invalid_png = INVALID_SOURCE.read_bytes()
    invalid = cv2.imdecode(np.frombuffer(invalid_png, np.uint8), cv2.IMREAD_COLOR)
    if invalid is None or invalid.shape != source.shape:
        raise ValueError("The original invalid-placement capture is required.")
    alternate_png = ALTERNATE_SOURCE.read_bytes()
    alternate = cv2.imdecode(np.frombuffer(alternate_png, np.uint8), cv2.IMREAD_COLOR)
    if alternate is None or alternate.shape != source.shape:
        raise ValueError("The original alternate Raspberry catalog capture is required.")
    animated_png = ANIMATED_SOURCE.read_bytes()
    animated = cv2.imdecode(np.frombuffer(animated_png, np.uint8), cv2.IMREAD_COLOR)
    if animated is None or animated.shape != source.shape:
        raise ValueError("The animated Raspberry catalog capture is required.")
    placement_supporting = []
    for path, purpose in PLACEMENT_SOURCES:
        data = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != source.shape:
            raise ValueError(f"The live placement capture is invalid: {path.name}")
        placement_supporting.append({
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_size": [1920, 1080],
            "purpose": purpose,
        })
    for output in OUTPUTS:
        output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source_png).hexdigest(),
        "source_size": [1920, 1080],
        "supporting_sources": [
            {
                "source": str(INVALID_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": hashlib.sha256(invalid_png).hexdigest(),
                "source_size": [1920, 1080],
                "purpose": "Explicit no-space negative placement outcome.",
            },
            {
                "source": str(ALTERNATE_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": hashlib.sha256(alternate_png).hexdigest(),
                "source_size": [1920, 1080],
                "purpose": "Same Raspberry catalog after a camera pan; independent layout replay.",
            },
            {
                "source": str(ANIMATED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": hashlib.sha256(animated_png).hexdigest(),
                "source_size": [1920, 1080],
                "purpose": "Same Raspberry card in a different live idle-animation pose.",
            },
            *placement_supporting,
        ],
        "coordinate_format": "Source feature boxes are LTRB; runtime positions are always freshly detected.",
        "species": {
            "raspberry": {
                "desired_file": "desired_raspberry.png",
                "desired_source": str(DESIRED.relative_to(ROOT)).replace("\\", "/"),
                "desired_source_sha256": hashlib.sha256(desired_png).hexdigest(),
                "desired_source_size": [desired.shape[1], desired.shape[0]],
            }
        },
        "features": {},
        "geometry": {
            "header_box": BOXES["header"],
            "raspberry_card_box": BOXES["raspberry_card"],
            "card_arrow_box": BOXES["card_arrow"],
            "tree_tab_box": BOXES["tree_tab"],
            "shop_button_box": BOXES["shop_button"],
            "drag_point": [426, 270],
            "panel_right": 918,
            "minimum_clearance_at_1080p": 42,
        },
        "transaction": (
            "The worker writes placement_attempted before its only charge-capable swipe. "
            "A restart never repeats that swipe. Growing requires two fresh stable local observations."
        ),
        "provenance": (
            "Every reference retains the original captured RGB. Alpha is a binary inclusion mask over stable "
            "shop artwork; source coordinates are provenance and never runtime farm coordinates."
        ),
    }

    for name, box in BOXES.items():
        x0, y0, x1, y1 = box
        crop = source[y0:y1, x0:x1]
        mask, description = MASKS[name](crop)
        data = _encoded(crop, mask)
        for output in OUTPUTS:
            (output / f"{name}.png").write_bytes(data)
        manifest["features"][name] = {
            "file": f"{name}.png",
            "box": box,
            "mask": description,
            "opaque_pixels": int(np.count_nonzero(mask)),
        }

    desired_mask, description = _desired_raspberry(desired)
    desired_data = _encoded(desired, desired_mask)
    for output in OUTPUTS:
        (output / "desired_raspberry.png").write_bytes(desired_data)
    manifest["species"]["raspberry"]["mask"] = description
    manifest["species"]["raspberry"]["opaque_pixels"] = int(np.count_nonzero(desired_mask))

    board_png = BOARD_SOURCE.read_bytes()
    board = cv2.imdecode(np.frombuffer(board_png, np.uint8), cv2.IMREAD_COLOR)
    if board is None or board.shape != source.shape:
        raise ValueError('The original Raspberry order-board capture is required.')
    x0, y0, x1, y1 = BOARD_ICON_BOX
    crop = board[y0:y1, x0:x1]
    mask, description = _desired_raspberry(crop)
    data = _encoded(crop, mask)
    for output in OUTPUTS:
        (output / 'desired_raspberry_board.png').write_bytes(data)
    manifest['species']['raspberry']['desired_board_file'] = 'desired_raspberry_board.png'
    manifest['features']['desired_raspberry_board'] = {
        'file': 'desired_raspberry_board.png', 'box': BOARD_ICON_BOX,
        'source': BOARD_SOURCE.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(board_png).hexdigest(),
        'mask': description, 'opaque_pixels': int(np.count_nonzero(mask)),
    }

    x0, y0, x1, y1 = INVALID_BOX
    invalid_crop = invalid[y0:y1, x0:x1]
    low = invalid_crop.min(axis=2)
    high = invalid_crop.max(axis=2)
    invalid_mask = ((high < 95) | ((low > 205) & (high-low < 55))).astype(np.uint8)*255
    invalid_mask = cv2.morphologyEx(invalid_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    invalid_data = _encoded(invalid_crop, invalid_mask)
    for output in OUTPUTS:
        (output / "invalid_space.png").write_bytes(invalid_data)
    manifest["features"]["invalid_space"] = {
        "file": "invalid_space.png",
        "box": INVALID_BOX,
        "source": str(INVALID_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(invalid_png).hexdigest(),
        "mask": "Original black outline and neutral white NOT ENOUGH SPACE HERE text; binary 3x3 close.",
        "opaque_pixels": int(np.count_nonzero(invalid_mask)),
    }
    x0, y0, x1, y1 = ROTATE_BOX
    rotate_crop = invalid[y0:y1, x0:x1]
    hsv = cv2.cvtColor(rotate_crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    rotate_mask = (((hue <= 30) & (saturation >= 85) & (value >= 70))
                   | ((saturation <= 45) & (value >= 205))
                   | (value <= 75)).astype(np.uint8)*255
    ellipse = np.zeros_like(rotate_mask)
    cv2.ellipse(ellipse, (95, 78), (91, 76), 0, 0, 360, 255, -1)
    rotate_mask &= ellipse
    rotate_data = _encoded(rotate_crop, rotate_mask)
    for output in OUTPUTS:
        (output / "placement_rotate.png").write_bytes(rotate_data)
    manifest["features"]["placement_rotate"] = {
        "file": "placement_rotate.png",
        "box": ROTATE_BOX,
        "source": str(INVALID_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(invalid_png).hexdigest(),
        "mask": "Original orange arrows, neutral white button, and dark outline inside the observed rotate control ellipse.",
        "opaque_pixels": int(np.count_nonzero(rotate_mask)),
    }

    encoded_manifest = json.dumps(manifest, indent=2) + "\n"
    for output in OUTPUTS:
        (output / "manifest.json").write_text(encoded_manifest, encoding="utf-8")


if __name__ == "__main__":
    build()

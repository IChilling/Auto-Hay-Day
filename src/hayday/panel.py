"""Read-only, layout-aware verification of the truck-order modal.

The shipped references contain static frame pixels, never order text or tickets.
They describe the observed modal style, not every possible future game redesign.
Matching a generic close button or observing a changed screen is insufficient.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Bounds = tuple[int, int, int, int]


@dataclass(frozen=True)
class PanelFeature:
    name: str
    score: float
    bounds: Bounds


@dataclass(frozen=True)
class PanelResult:
    verified: bool
    reason: str
    score: float = 0.0
    features: tuple[PanelFeature, ...] = ()
    bounds: Bounds | None = None


@dataclass(frozen=True)
class _Reference:
    name: str
    image: np.ndarray
    mask: np.ndarray
    x: int
    y: int


class PanelVerifier:
    """Verify independent modal features at one consistent scale and translation.

    Bounds are ``(x, y, width, height)`` in original screenshot pixels.
    The result does not describe truck availability or whether an order is ready.
    """

    def __init__(self, assets_dir: Path | None = None):
        project = Path(__file__).resolve().parents[2]
        default_assets = (
            project / "images" if (project / "pyproject.toml").is_file()
            else Path(__file__).parent / "assets" / "identifiers"
        )
        self.assets_dir = Path(assets_dir) if assets_dir is not None else default_assets
        self._references: dict[str, _Reference] = {}
        self._asset_error = ""
        try:
            manifest = json.loads((self.assets_dir / "panel_manifest.json").read_text())
            if not isinstance(manifest, dict) or manifest.get("reference_size") != [1920, 1080]:
                raise ValueError("Panel references must describe the 1920 x 1080 source layout")
            for name in ("header_edge", "close_button", "board_corner", "detail_corner"):
                entry = manifest["features"][name]
                box = entry["box"]
                if not isinstance(box, list) or len(box) != 4 or not all(
                    type(value) is int for value in box
                ):
                    raise ValueError(f"{name} has invalid source coordinates")
                if not (0 <= box[0] < box[2] <= 1920 and 0 <= box[1] < box[3] <= 1080):
                    raise ValueError(f"{name} source coordinates are outside the reference")
                raw = cv2.imdecode(
                    np.frombuffer((self.assets_dir / entry["file"]).read_bytes(), np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
                    raise ValueError(f"{name} must be an RGBA image")
                if raw.shape[:2] != (box[3]-box[1], box[2]-box[0]):
                    raise ValueError(f"{name} dimensions disagree with its source coordinates")
                mask = (raw[:, :, 3] >= 200).astype(np.uint8) * 255
                if np.count_nonzero(mask) < 40:
                    raise ValueError(f"{name} has too few visible pixels")
                self._references[name] = _Reference(
                    name, raw[:, :, :3], mask, *box[:2]
                )
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self._asset_error = f"Truck-order panel references unavailable: {exc}"

    @property
    def reference_error(self) -> str:
        """Empty when references loaded; callers can validate before issuing input."""
        return self._asset_error

    @staticmethod
    def _scaled(reference: _Reference, scale: float) -> tuple[np.ndarray, np.ndarray]:
        size = (
            max(2, round(reference.image.shape[1] * scale)),
            max(2, round(reference.image.shape[0] * scale)),
        )
        # Premultiply before resampling so arbitrary hidden RGB cannot enter a
        # visible edge. Keep only pixels with predominantly opaque source support.
        alpha = reference.mask.astype(np.float32) / 255
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        weight = cv2.resize(alpha, size, interpolation=interpolation)
        rgb = cv2.resize(
            reference.image.astype(np.float32) * alpha[:, :, None],
            size, interpolation=interpolation,
        )
        image = np.clip(rgb / np.maximum(weight[:, :, None], 1e-6), 0, 255).astype(np.uint8)
        mask = (weight >= 0.85).astype(np.uint8) * 255
        return image, mask

    @staticmethod
    def _match(image: np.ndarray, template: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    def verify(self, png: bytes, cancel: Callable[[], bool] | None = None) -> PanelResult:
        """Inspect a screenshot only; errors and cancellation return unverified."""
        stopped = cancel or (lambda: False)
        if stopped():
            return PanelResult(False, "Panel verification cancelled.")
        if self._asset_error:
            return PanelResult(False, self._asset_error)
        try:
            raw = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, cv2.error):
            raw = None
        if raw is None:
            return PanelResult(False, "Panel verification requires a valid screenshot.")
        height, width = raw.shape[:2]
        if min(height, width) < 240:
            return PanelResult(False, "Screenshot is too small to verify the order panel.")
        resize = min(1.0, 960 / width, 720 / height)
        frame = cv2.resize(raw, (round(width * resize), round(height * resize)))
        height, width = frame.shape[:2]
        nominal = min(width / 1920, height / 1080)
        # Panel size is governed by viewport fit, not the farm camera's zoom.
        scales = nominal * np.unique(np.r_[np.geomspace(0.70, 1.16, 15), 1.0])
        header = self._references["header_edge"]
        candidates: list[tuple[float, float, int, int]] = []
        search = frame[: round(height * 0.43)]
        for scale in scales:
            if stopped():
                return PanelResult(False, "Panel verification cancelled.")
            template, mask = self._scaled(header, float(scale))
            if min(mask.shape) < 12 or any(
                a < b for a, b in zip(search.shape, template.shape, strict=True)
            ):
                continue
            scores = self._match(search, template, mask)
            for _ in range(2):
                _, score, _, point = cv2.minMaxLoc(scores)
                if score < 0.75:
                    break
                candidates.append((score, float(scale), *point))
                x, y = point
                radius = max(12, round(50 * scale))
                scores[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1

        best_score = 0.0
        best_features: tuple[PanelFeature, ...] = ()
        for header_score, scale, x, y in sorted(candidates, reverse=True)[:12]:
            if stopped():
                return PanelResult(False, "Panel verification cancelled.")
            tx, ty = x - header.x * scale, y - header.y * scale
            # The entire observed panel must be visible; don't extrapolate offscreen.
            if tx + 125*scale < -3 or ty + 25*scale < -3:
                continue
            if tx + 1850*scale > width + 3 or ty + 1040*scale > height + 3:
                continue
            features = [PanelFeature(
                header.name, header_score,
                self._bounds(x, y, header.image.shape[1]*scale,
                             header.image.shape[0]*scale, resize),
            )]
            for name in ("close_button", "board_corner", "detail_corner"):
                reference = self._references[name]
                template, mask = self._scaled(reference, scale)
                th, tw = template.shape[:2]
                expected_x, expected_y = tx + reference.x*scale, ty + reference.y*scale
                tolerance = max(3, round(13*scale))
                left, top = max(0, round(expected_x)-tolerance), max(0, round(expected_y)-tolerance)
                right = min(width, round(expected_x)+tw+tolerance+1)
                bottom = min(height, round(expected_y)+th+tolerance+1)
                roi = frame[top:bottom, left:right]
                if roi.shape[0] < th or roi.shape[1] < tw:
                    continue
                _, score, _, point = cv2.minMaxLoc(self._match(roi, template, mask))
                if score >= (0.82 if name == "close_button" else 0.76):
                    features.append(PanelFeature(
                        name, score,
                        self._bounds(left+point[0], top+point[1], tw, th, resize),
                    ))
            score = sum(feature.score for feature in features) / 4
            if score > best_score:
                best_score, best_features = score, tuple(features)
            names = {feature.name for feature in features}
            if not {"header_edge", "close_button", "board_corner"}.issubset(names):
                continue
            if header_score < 0.82 or not self._surfaces_agree(frame, scale, tx, ty):
                continue
            return PanelResult(
                True, "Truck-order panel verified by its header, close control, board frame, "
                "and matching wood/cream panel layout.",
                min(feature.score for feature in features), tuple(features),
                self._bounds(tx+125*scale, ty+25*scale, 1725*scale, 1015*scale, resize),
            )
        return PanelResult(
            False, "Truck-order panel was not verified: its independent frame features "
            "and panel layout did not agree.", best_score, best_features,
        )

    @staticmethod
    def _bounds(x: float, y: float, width: float, height: float, resize: float) -> Bounds:
        return tuple(round(value / resize) for value in (x, y, width, height))

    @staticmethod
    def _surfaces_agree(frame: np.ndarray, scale: float, tx: float, ty: float) -> bool:
        def region(box: tuple[int, int, int, int]) -> np.ndarray:
            x1, y1, x2, y2 = box
            return frame[
                max(0, round(ty+y1*scale)):round(ty+y2*scale),
                max(0, round(tx+x1*scale)):round(tx+x2*scale),
            ].astype(np.int16)

        def fraction(pixels: np.ndarray, kind: str) -> float:
            if pixels.size == 0:
                return 0.0
            b, g, r = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
            if kind == "gold":
                found = (r > 195) & (g > 120) & (g < 240) & (b < 110) & (r-g > 12)
            elif kind == "wood":
                found = (r > 105) & (r < 225) & (r-g > 18) & (g-b > 4) & (b < 155)
            else:
                found = (r > 220) & (g > 205) & (b > 150) & (r-b > 8)
            return float(np.mean(found))

        return (
            fraction(region((1164, 300, 1184, 755)), "gold") > 0.70
            and fraction(region((445, 1006, 1690, 1018)), "gold") > 0.70
            and fraction(region((357, 196, 1144, 985)), "wood") > 0.15
            and fraction(region((1205, 235, 1719, 810)), "cream") > 0.43
        )

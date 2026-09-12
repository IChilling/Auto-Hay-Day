"""Locate the truck order board using only the opaque parts of its PNG reference.

The detector has no game coordinates and sends no input. The four reference
quadrants remain at their original relative positions during verification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureMatch:
    name: str
    score: float


@dataclass(frozen=True)
class BoardMatch:
    x: int
    y: int
    width: int
    height: int
    scale: float
    score: float
    features: tuple[FeatureMatch, ...]
    tap_x: int
    tap_y: int


@dataclass(frozen=True)
class DetectionReport:
    match: BoardMatch | None
    reason: str
    candidates: tuple[BoardMatch, ...] = ()


@dataclass(frozen=True)
class _Template:
    color: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    scale: float


class BoardDetector:
    """Conservative, cancellable multiscale masked board detector.

    Scale is relative to the cropped original PNG, not to a particular emulator
    resolution. Extremely small, clipped or heavily obscured boards are rejected.
    Scores are similarity measures, not calibrated statistical probabilities.
    """

    MIN_SCALE = 0.12
    MAX_SCALE = 2.0
    FEATURE_THRESHOLD = 0.64
    SCORE_THRESHOLD = 0.72

    def __init__(self, template_path: Path | None = None):
        if template_path is None:
            source = Path(__file__).resolve().parents[2] / "images" / "quest_board_identifier.png"
            template_path = source if source.is_file() else (
                Path(__file__).parent / "assets" / "identifiers" / "quest_board_identifier.png"
            )
        self.template_path = Path(template_path)
        try:
            data = np.frombuffer(self.template_path.read_bytes(), np.uint8)
            rgba = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        except OSError as exc:
            raise ValueError(f"Cannot read board identifier: {self.template_path}") from exc
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("Board identifier must be a PNG with an alpha transparency channel.")
        alpha = rgba[:, :, 3]
        ys, xs = np.nonzero(alpha)
        if len(xs) < 64 or not np.any(alpha == 0):
            raise ValueError("Board identifier needs both visible features and transparent areas.")
        self.alpha_bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        x0, y0, x1, y1 = self.alpha_bounds
        self._color = rgba[y0:y1, x0:x1, :3].astype(np.float32)
        self._alpha = rgba[y0:y1, x0:x1, 3].astype(np.float32) / 255.0
        self.height, self.width = self._alpha.shape

    def _resize(self, scale: float) -> _Template:
        width, height = max(4, round(self.width * scale)), max(4, round(self.height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        alpha = cv2.resize(self._alpha, (width, height), interpolation=interpolation)
        # Premultiplication keeps arbitrary RGB values under transparent pixels
        # from bleeding into visible colors when a reference is scaled down.
        premultiplied = cv2.resize(
            self._color * self._alpha[:, :, None], (width, height), interpolation=interpolation
        )
        color = premultiplied / np.maximum(alpha[:, :, None], 1e-6)
        color = np.clip(color, 0, 255).astype(np.uint8)
        mask = (alpha >= 0.85).astype(np.uint8) * 255
        return _Template(color, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), mask, scale)

    @staticmethod
    def _score(color: np.ndarray, template: _Template) -> tuple[float, tuple[FeatureMatch, ...]]:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
        expected = template.gray.astype(np.float32)
        height, width = gray.shape
        regions = (
            ("upper left frame", slice(0, height // 2), slice(0, width // 2)),
            ("upper right frame", slice(0, height // 2), slice(width // 2, width)),
            ("lower left support", slice(height // 2, height), slice(0, width // 2)),
            ("lower right ledge", slice(height // 2, height), slice(width // 2, width)),
        )

        def similarity(ys: slice, xs: slice) -> float:
            selected = template.mask[ys, xs] > 0
            if np.count_nonzero(selected) < 12:
                return 0.0
            a, b = gray[ys, xs][selected], expected[ys, xs][selected]
            a, b = a - a.mean(), b - b.mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator < 1e-5 or float(a.std()) < 3.0:
                return 0.0
            correlation = max(0.0, float(np.dot(a, b) / denominator))
            difference = np.abs(
                color[ys, xs][selected].astype(np.float32)
                - template.color[ys, xs][selected].astype(np.float32)
            ).mean()
            color_score = max(0.0, 1.0 - float(difference) / 96.0)
            return 0.72 * correlation + 0.28 * color_score

        features = tuple(FeatureMatch(name, similarity(ys, xs)) for name, ys, xs in regions)
        whole = similarity(slice(0, height), slice(0, width))
        # One feature may be obscured; the other three still span both axes and
        # must agree at one common position and scale, with no independent shifts.
        strongest = sorted((feature.score for feature in features), reverse=True)[:3]
        return 0.55 * whole + 0.45 * float(np.mean(strongest)), features

    @staticmethod
    def _same_board(first: BoardMatch, second: BoardMatch) -> bool:
        # Neighboring sizes and offsets of a single peak are not two boards.
        distance_x = abs((first.x + first.width / 2) - (second.x + second.width / 2))
        distance_y = abs((first.y + first.height / 2) - (second.y + second.height / 2))
        return distance_x < min(first.width, second.width) * 0.4 and (
            distance_y < min(first.height, second.height) * 0.4
        )

    @staticmethod
    def _decode(png: bytes) -> np.ndarray:
        try:
            frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise ValueError("Board search requires a valid screenshot image.") from exc
        if frame is None:
            raise ValueError("Board search requires a valid screenshot image.")
        return frame

    def _refine(
        self, frame: np.ndarray, start_x: int, start_y: int, scale: float,
        cancelled: Callable[[], bool], adjustments: np.ndarray, margin: int,
    ) -> BoardMatch | None:
        frame_height, frame_width = frame.shape[:2]
        best: BoardMatch | None = None
        for adjustment in adjustments:
            if cancelled():
                return None
            template = self._resize(scale * float(adjustment))
            height, width = template.gray.shape
            if width > frame_width or height > frame_height:
                continue
            center_x = start_x + self.width * scale / 2
            center_y = start_y + self.height * scale / 2
            x, y = round(center_x - width / 2), round(center_y - height / 2)
            left, top = max(0, x - margin), max(0, y - margin)
            right = min(frame_width, x + width + margin)
            bottom = min(frame_height, y + height + margin)
            if right - left < width or bottom - top < height:
                continue
            region = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            response = cv2.matchTemplate(
                region, template.gray, cv2.TM_CCOEFF_NORMED, mask=template.mask
            )
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, _, _, location = cv2.minMaxLoc(response)
            match_x, match_y = left + location[0], top + location[1]
            patch = frame[match_y:match_y+height, match_x:match_x+width]
            score, features = self._score(patch, template)
            candidate = BoardMatch(
                match_x, match_y, width, height, template.scale, score, features,
                match_x + width // 2, match_y + height // 2,
            )
            if best is None or score > best.score:
                best = candidate
        return best

    def _accepted(self, candidate: BoardMatch) -> bool:
        return candidate.score >= self.SCORE_THRESHOLD and sum(
            feature.score >= self.FEATURE_THRESHOLD for feature in candidate.features
        ) >= 3

    def revalidate(
        self, png: bytes, match: BoardMatch, cancel: Callable[[], bool] | None = None,
    ) -> DetectionReport:
        """Check the board again immediately before input, near its last location.

        Searches within six pixels at three nearby sizes; it cannot switch to a
        different distant target. The caller must also check screenshot dimensions
        and device identity against the screenshot used for the original search.
        """
        cancelled = cancel or (lambda: False)
        if cancelled():
            return DetectionReport(None, "Board search cancelled.")
        frame = self._decode(png)
        refined = self._refine(
            frame, match.x, match.y, match.scale, cancelled,
            np.array([0.97, 1.0, 1.03]), margin=6,
        )
        if cancelled():
            return DetectionReport(None, "Board search cancelled.")
        if refined is None or not self._accepted(refined):
            return DetectionReport(None, "The board changed or moved before the tap; no tap was sent.")
        return DetectionReport(refined, "The same board is still visible at the expected location.", (refined,))

    def detect(self, png: bytes, cancel: Callable[[], bool] | None = None) -> DetectionReport:
        """Find an unambiguous visible board, or explain why no tap is justified."""
        cancelled = cancel or (lambda: False)
        if cancelled():
            return DetectionReport(None, "Board search cancelled.")
        frame = self._decode(png)
        frame_height, frame_width = frame.shape[:2]
        ratio = min(1.0, 960.0 / max(frame_height, frame_width))
        small = cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        max_scale = min(
            self.MAX_SCALE, frame_width / self.width * 0.98, frame_height / self.height * 0.95
        )
        if max_scale < self.MIN_SCALE:
            return DetectionReport(None, "Screenshot is too small to resolve the board features.")
        scales = np.geomspace(self.MIN_SCALE, max_scale, max(2, int(np.ceil(
            np.log(max_scale / self.MIN_SCALE) / np.log(1.1)
        )) + 1))
        coarse: list[tuple[float, int, int, float]] = []
        seen_dimensions: set[tuple[int, int]] = set()
        for scale in scales:
            if cancelled():
                return DetectionReport(None, "Board search cancelled.")
            template = self._resize(float(scale) * ratio)
            height, width = template.gray.shape
            if (width, height) in seen_dimensions or np.count_nonzero(template.mask) < 48:
                continue
            seen_dimensions.add((width, height))
            response = cv2.matchTemplate(gray, template.gray, cv2.TM_CCOEFF_NORMED, mask=template.mask)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                if score < 0.42:
                    break
                coarse.append((score, round(x / ratio), round(y / ratio), float(scale)))
                radius = max(3, min(width, height) // 3)
                response[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1.0

        # Refine only plausible local maxima. Spatial deduplication occurs after
        # refinement so a nearby coarse scale can rescue a blurred first peak.
        coarse.sort(reverse=True)
        refined: list[BoardMatch] = []
        for _, start_x, start_y, scale in coarse[:18]:
            if cancelled():
                return DetectionReport(None, "Board search cancelled.")
            best = self._refine(
                frame, start_x, start_y, scale, cancelled,
                np.linspace(0.95, 1.05, 7), margin=max(4, round(3 / ratio)),
            )
            if best is not None:
                refined.append(best)
        if cancelled():
            return DetectionReport(None, "Board search cancelled.")
        refined.sort(key=lambda candidate: candidate.score, reverse=True)
        distinct: list[BoardMatch] = []
        for candidate in refined:
            if not any(self._same_board(candidate, previous) for previous in distinct):
                distinct.append(candidate)
        # At close zoom, a coarse scale interval spans several pixels. A final
        # pass around plausible distinct peaks keeps the returned box and tap
        # location accurate without doing a dense whole-screen scale search.
        for index, candidate in enumerate(distinct[:4]):
            if candidate.score < 0.65:
                continue
            fine = self._refine(
                frame, candidate.x, candidate.y, candidate.scale, cancelled,
                np.linspace(0.975, 1.025, 21), margin=6,
            )
            if fine is not None and fine.score > candidate.score:
                distinct[index] = fine
        if cancelled():
            return DetectionReport(None, "Board search cancelled.")
        distinct.sort(key=lambda candidate: candidate.score, reverse=True)
        accepted = [candidate for candidate in distinct if self._accepted(candidate)]
        if not accepted:
            return DetectionReport(
                None, "Board not found with enough agreeing frame and support features. "
                "Keep the whole board visible and try a closer zoom.", tuple(distinct[:3])
            )
        if len(accepted) > 1 and accepted[0].score - accepted[1].score < 0.07:
            return DetectionReport(
                None, "Board search is ambiguous: multiple locations have similar feature agreement.",
                tuple(accepted[:3]),
            )
        return DetectionReport(
            accepted[0], "Board located using its frame and supports; the truck is not part of the match.",
            tuple(distinct[:3]),
        )

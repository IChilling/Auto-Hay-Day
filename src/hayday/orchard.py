"""Recognize and place one Raspberry bush through the observed orchard catalog.

The catalog is opened from a freshly matched shop control. A placement intent is
durably written before the only charge-capable swipe. Observed coordinates may be
kept as transaction evidence, but are never reused to authorize later input.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.resources import ResourceChanged, ResourceResult, _save_json


@dataclass(frozen=True)
class OrchardObservation:
    """Fresh, resolution-scaled shop and placement observations for one species."""

    species: str
    shop_button: VisualTarget | None
    header: VisualTarget | None
    card: VisualTarget | None
    card_arrow: VisualTarget | None
    tree_tab: VisualTarget | None
    drag_start: VisualTarget | None
    clear_targets: tuple[VisualTarget, ...]
    score: float
    panel_right: int | None = None

    @property
    def shop_open(self) -> bool:
        return all((self.header, self.card, self.card_arrow, self.tree_tab, self.drag_start))


class OrchardVision:
    """Read the Raspberry orchard catalog and find full-footprint clear grass."""

    _FEATURES = (
        "shop_button", "header", "raspberry_card", "card_arrow", "tree_tab",
        "invalid_space", "placement_rotate",
    )

    def __init__(self, reference_path: Path | None = None):
        checkout = Path(__file__).resolve().parents[2] / "images" / "orchard"
        self.reference_path = Path(reference_path) if reference_path else (
            checkout if checkout.is_dir() else Path(__file__).parent / "assets" / "orchard"
        )
        try:
            self.manifest = json.loads((self.reference_path / "manifest.json").read_text("utf-8"))
            if self.manifest.get("version") != 1:
                raise ValueError("Unsupported orchard reference version.")
            self.references = {
                name: _decode((self.reference_path / f"{name}.png").read_bytes(), True)
                for name in self._FEATURES
            }
            desired_file = self.manifest["species"]["raspberry"]["desired_file"]
            self.desired = _decode((self.reference_path / desired_file).read_bytes(), True)
            board_file = self.manifest['species']['raspberry'].get('desired_board_file')
            self.desired_variants = (self.desired,) + (
                (_decode((self.reference_path / board_file).read_bytes(), True),) if board_file else ())
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Orchard references are missing or invalid.") from exc
        for reference in (*self.references.values(), *self.desired_variants):
            if (reference.ndim != 3 or reference.shape[2] != 4
                    or np.count_nonzero(reference[:, :, 3] >= 200) < 40):
                raise ValueError("Orchard references require original pixels and alpha masks.")
        self._matcher = ResourceVision()

    @staticmethod
    def _check(cancel: Callable[[], bool]) -> None:
        if cancel():
            raise ResourceChanged("Orchard recognition cancelled.")

    def _search(self, image: np.ndarray, name: str, threshold: float,
                cancel: Callable[[], bool], *, peaks: int = 6) -> tuple[VisualTarget, ...]:
        self._check(cancel)
        base = image.shape[0] / 1080
        scales = np.unique(np.r_[np.geomspace(base * .72, base * 1.32, 17), base])
        result = self._matcher._search(
            image, self.references[name], scales, threshold, cancel, max_peaks=peaks,
        )
        self._check(cancel)
        return result

    @staticmethod
    def _unambiguous(matches: tuple[VisualTarget, ...], margin: float = .025) -> VisualTarget | None:
        if not matches or (len(matches) > 1 and matches[0].score - matches[1].score < margin):
            return None
        return matches[0]

    def _species(self, desired_icon: bytes, cancel: Callable[[], bool]) -> str | None:
        self._check(cancel)
        image = _decode(desired_icon)
        scales = np.unique(np.r_[np.geomspace(.58, 1.60, 19), 1.0])
        for reference in self.desired_variants:
            found = self._matcher._search(image, reference, scales, .90, cancel, max_peaks=3)
            self._check(cancel)
            if not found:
                continue
            first = found[0]
            height, width = image.shape[:2]
            centered = (abs(first.center[0] - width / 2) <= width * .30
                        and abs(first.center[1] - height / 2) <= height * .30)
            ambiguous = len(found) > 1 and first.score - found[1].score < .025
            if centered and not ambiguous:
                return 'raspberry'
        return None

    @staticmethod
    def _relative_center(source_box, target: VisualTarget, point) -> tuple[float, float]:
        scale = target.width / max(1, source_box[2] - source_box[0])
        return (target.x + (point[0] - source_box[0]) * scale,
                target.y + (point[1] - source_box[1]) * scale)

    @staticmethod
    def _near(target: VisualTarget, point: tuple[float, float], tolerance: float) -> bool:
        return math.dist(target.center, point) <= tolerance

    @staticmethod
    def _clear_grass(image: np.ndarray, panel_right: int, radius: int) -> tuple[VisualTarget, ...]:
        """Return separated maxima whose complete circular footprint is clear grass."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (24, 65, 55), (90, 255, 255))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 125)
        grass[cv2.dilate(edges, np.ones((13, 13), np.uint8)) > 0] = 0
        allowed = np.zeros_like(grass)
        left = max(panel_right + round(45 * height / 1080), round(width * .16))
        right = round(width * .84)
        top, bottom = round(height * .19), round(height * .77)
        if right <= left or bottom <= top:
            return ()
        allowed[top:bottom, left:right] = 255
        grass &= allowed
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        distance = cv2.distanceTransform(grass, cv2.DIST_L2, 5)
        working = distance.copy()
        targets = []
        for _ in range(10):
            _, clearance, _, (x, y) = cv2.minMaxLoc(working)
            if clearance < radius:
                break
            score = min(1.0, float(clearance) / max(1.0, radius * 1.25))
            targets.append(VisualTarget(x - radius, y - radius, radius * 2, radius * 2, score))
            cv2.circle(working, (x, y), radius * 2, 0, -1)
        return tuple(targets)

    def observe(self, png: bytes, desired_icon: bytes,
                cancel: Callable[[], bool] | None = None) -> OrchardObservation | None:
        cancelled = cancel or (lambda: False)
        species = self._species(desired_icon, cancelled)
        if species is None:
            return None
        image = _decode(png)
        height, width = image.shape[:2]
        buttons = tuple(target for target in self._search(image, "shop_button", .87, cancelled)
                        if target.center[1] >= height * .72 and target.center[0] <= width * .60)
        shop_button = self._unambiguous(buttons, .018)

        headers = self._search(image, "header", .90, cancelled, peaks=3)
        header = self._unambiguous(headers)
        if header is None:
            return OrchardObservation(species, shop_button, None, None, None, None, None, (),
                                      shop_button.score if shop_button else 0.0)

        geometry = self.manifest["geometry"]
        header_box = geometry["header_box"]
        scale = header.width / (header_box[2] - header_box[0])
        card_box = geometry["raspberry_card_box"]
        card_source_center = ((card_box[0] + card_box[2]) / 2,
                              (card_box[1] + card_box[3]) / 2)
        card_expected = self._relative_center(header_box, header, card_source_center)
        # Bush cards bob vertically and flex their foliage while the catalog is
        # open. Species identity is already established from the selected order
        # icon, so allow that observed pose variation here while retaining the
        # independent header, arrow, tab, and relative-position gates below.
        cards = tuple(target for target in self._search(image, "raspberry_card", .84, cancelled)
                      if self._near(target, card_expected, max(22, 34 * scale)))
        card = self._unambiguous(cards)
        if card is None:
            return OrchardObservation(species, shop_button, header, None, None, None, None, (),
                                      min(header.score, shop_button.score if shop_button else 1.0))

        arrow_box = geometry["card_arrow_box"]
        arrow_source_center = ((arrow_box[0] + arrow_box[2]) / 2,
                               (arrow_box[1] + arrow_box[3]) / 2)
        arrow_expected = self._relative_center(card_box, card, arrow_source_center)
        arrows = tuple(target for target in self._search(image, "card_arrow", .89, cancelled)
                       if self._near(target, arrow_expected, max(12, 18 * scale)))
        arrow = self._unambiguous(arrows)

        tab_box = geometry["tree_tab_box"]
        tab_source_center = ((tab_box[0] + tab_box[2]) / 2,
                             (tab_box[1] + tab_box[3]) / 2)
        tab_expected = self._relative_center(header_box, header, tab_source_center)
        tabs = tuple(target for target in self._search(image, "tree_tab", .88, cancelled)
                     if self._near(target, tab_expected, max(14, 22 * scale)))
        tab = self._unambiguous(tabs)

        drag = None
        if arrow is not None and tab is not None:
            x, y = self._relative_center(card_box, card, geometry["drag_point"])
            extent = max(8, round(12 * scale))
            drag = VisualTarget(round(x - extent / 2), round(y - extent / 2), extent, extent,
                                min(card.score, arrow.score))
        panel_right = round(header.x + (geometry["panel_right"] - header_box[0]) * scale)
        radius = max(18, round(geometry["minimum_clearance_at_1080p"] * height / 1080))
        clear = self._clear_grass(image, panel_right, radius) if drag is not None else ()
        evidence = [item.score for item in (shop_button, header, card, arrow, tab) if item is not None]
        return OrchardObservation(species, shop_button, header, card, arrow, tab, drag, clear,
                                  min(evidence) if evidence else 0.0, panel_right)

    def invalid_placement(self, png: bytes,
                          cancel: Callable[[], bool] | None = None) -> bool:
        cancelled = cancel or (lambda: False)
        image = _decode(png)
        return bool(self._search(image, "invalid_space", .90, cancelled, peaks=3))

    def placement_mode(self, png: bytes,
                       cancel: Callable[[], bool] | None = None) -> bool:
        cancelled = cancel or (lambda: False)
        image = _decode(png)
        return bool(self._search(image, "placement_rotate", .90, cancelled, peaks=3))

    @staticmethod
    def _affine(before: np.ndarray, after: np.ndarray, panel_right: int) -> np.ndarray | None:
        """Estimate the farm view transform while ignoring the catalog and HUD."""
        height, width = before.shape[:2]
        orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
        before_mask = np.zeros((height, width), np.uint8)
        before_mask[round(height * .16):round(height * .80),
                    max(panel_right + 12, round(width * .15)):round(width * .87)] = 255
        after_mask = np.zeros_like(before_mask)
        after_mask[round(height * .16):round(height * .80),
                   round(width * .10):round(width * .87)] = 255
        first_points, first_desc = orb.detectAndCompute(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), before_mask)
        second_points, second_desc = orb.detectAndCompute(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), after_mask)
        if first_desc is None or second_desc is None or len(first_points) < 16 or len(second_points) < 16:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first_desc, second_desc, k=2)
        good = [one for one, two in pairs if one.distance < .72 * two.distance]
        if len(good) < 12:
            return None
        source = np.float32([first_points[match.queryIdx].pt for match in good])
        destination = np.float32([second_points[match.trainIdx].pt for match in good])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=3,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < 10:
            return None
        scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        angle = abs(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
        if (not .88 <= scale <= 1.12 or angle > math.radians(6)
                or translation > math.hypot(width, height) * .25):
            return None
        return matrix

    @staticmethod
    def _shrub_change(before: np.ndarray, aligned: np.ndarray,
                      destination: VisualTarget) -> bool:
        x, y, width, height = destination.box
        margin = max(6, round(width * .15))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1 = min(before.shape[1], x + width + margin)
        y1 = min(before.shape[0], y + height + margin)
        old, new = before[y0:y1, x0:x1], aligned[y0:y1, x0:x1]
        if old.size == 0 or old.shape != new.shape:
            return False
        old_hsv, new_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV), cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
        # Ordinary Hay Day grass is saturated but fairly bright. The newly
        # placed shrub introduces a large mass of genuinely dark green leaves.
        old_dark = ((old_hsv[:, :, 0] >= 25) & (old_hsv[:, :, 0] <= 95)
                    & (old_hsv[:, :, 1] >= 80) & (old_hsv[:, :, 2] <= 160))
        new_dark = ((new_hsv[:, :, 0] >= 25) & (new_hsv[:, :, 0] <= 95)
                    & (new_hsv[:, :, 1] >= 80) & (new_hsv[:, :, 2] <= 160))
        old_edges = cv2.Canny(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), 55, 125) > 0
        new_edges = cv2.Canny(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), 55, 125) > 0
        difference = np.abs(old.astype(np.int16) - new.astype(np.int16)).mean(axis=2)
        return bool(
            new_dark.mean() >= old_dark.mean() + .045
            and new_edges.mean() >= old_edges.mean() + .025
            and np.mean(difference >= 24) >= .10
        )

    def placement_confirmed(self, before_png: bytes, after_pngs: tuple[bytes, bytes],
                            destination: VisualTarget, panel_right: int,
                            cancel: Callable[[], bool] | None = None) -> bool:
        """Require the same new green, structured local object in two fresh frames."""
        cancelled = cancel or (lambda: False)
        self._check(cancelled)
        before = _decode(before_png)
        confirmations = []
        for png in after_pngs:
            self._check(cancelled)
            after = _decode(png)
            if after.shape != before.shape:
                return False
            matrix = self._affine(before, after, panel_right)
            if matrix is None:
                return False
            aligned = cv2.warpAffine(
                after, matrix, (before.shape[1], before.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            confirmations.append(self._shrub_change(before, aligned, destination))
        self._check(cancelled)
        return bool(all(confirmations))


class OrchardWorker:
    """Place at most one Raspberry bush and preserve any uncertain purchase intent."""

    _STAGES = {"placement_attempted", "growing"}

    def __init__(self, client, capture, cancel_event, progress, state_path, *, vision=None):
        self.client = client
        self.capture = capture
        self.cancel_event = cancel_event
        self.progress = progress
        self.state_path = Path(state_path)
        self.vision = vision or OrchardVision()
        self.serial = client.serial
        self._size: tuple[int, int] | None = None
        self.state = {"version": 1, "serial": self.serial, "items": {}, "history": []}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text("utf-8"))
            if (not isinstance(self.state, dict) or self.state.get("version") != 1
                    or self.state.get("serial") != self.serial
                    or not isinstance(self.state.get("items"), dict)
                    or not isinstance(self.state.get("history", []), list)
                    or any(not isinstance(entry, dict) or entry.get("stage") not in self._STAGES
                           for entry in self.state["items"].values())):
                raise ValueError("Saved orchard placement intent is invalid or belongs to another device.")
            self.state.setdefault("history", [])

    def _check(self) -> None:
        if self.cancel_event.is_set() or not self.serial or self.client.serial != self.serial:
            raise ResourceChanged("Orchard work was cancelled or its selected device changed.")

    def _wait(self, seconds: float = .4) -> None:
        self._check()
        self.cancel_event.wait(seconds)
        self._check()

    def _frame(self):
        self._check()
        frame = self.capture()
        self._check()
        if self._size and self._size != (frame.width, frame.height):
            raise ResourceChanged("Device resolution changed during orchard work.")
        self._size = frame.width, frame.height
        return frame

    def _record(self, key: str, stage: str, **details) -> None:
        entry = self.state["items"].setdefault(key, {})
        entry.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **details)
        _save_json(self.state_path, self.state)

    def _reconcile_saved_placement(self, key: str, entry: dict, icon: bytes) -> bool:
        """Promote a guarded attempt from immutable before/after evidence only.

        This path never captures or sends device input. It lets improved visual
        validation resolve an earlier conservative result without ever buying a
        second bush.
        """
        names = {
            label: entry.get(f"{label}_capture")
            for label in ("before", "after_1", "after_2")
        }
        if any(not isinstance(name, str) or Path(name).name != name for name in names.values()):
            return False
        evidence = self.state_path.parent / "orchard_evidence"
        payloads = {}
        for label, name in names.items():
            path = evidence / name
            expected = entry.get(f"{label}_sha256")
            if not path.is_file() or not isinstance(expected, str):
                return False
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                return False
            payloads[label] = data
        self._check()
        before = self.vision.observe(
            payloads["before"], icon, cancel=self.cancel_event.is_set,
        )
        self._check()
        if before is None or not before.shop_open or before.panel_right is None:
            return False
        after_payloads = (payloads["after_1"], payloads["after_2"])
        if any(self.vision.invalid_placement(data, cancel=self.cancel_event.is_set)
               or self.vision.placement_mode(data, cancel=self.cancel_event.is_set)
               for data in after_payloads):
            self._check()
            return False

        stored = entry.get("destination_evidence")
        # Current transactions save the exact observed target as evidence.
        # Legacy guarded attempts predate that field; their before-frame still
        # deterministically identifies the first ranked target the worker used.
        candidates = before.clear_targets[:1]
        if (isinstance(stored, list) and len(stored) == 4
                and all(type(value) is int for value in stored)):
            candidates = (VisualTarget(*stored, 1.0),)
        panel_right = entry.get("panel_right_evidence", before.panel_right)
        if type(panel_right) is not int or panel_right < 0:
            panel_right = before.panel_right
        confirmed = []
        for destination in candidates:
            if self.vision.placement_confirmed(
                payloads["before"], after_payloads, destination,
                panel_right,
                cancel=self.cancel_event.is_set,
            ):
                confirmed.append(destination)
        self._check()
        if len(confirmed) != 1:
            return False
        after = self.vision.observe(
            payloads["after_2"], icon, cancel=self.cancel_event.is_set,
        )
        self._check()
        self._record(
            key,
            "growing",
            placement_confirmed_at=datetime.now(UTC).isoformat(),
            evidence_reconciled=True,
            destination_evidence=list(confirmed[0].box),
            panel_right_evidence=before.panel_right,
            catalog_open=bool(after and after.shop_open),
            placement_mode=False,
        )
        return True

    def observe_inventory(self, key, status, available, required, observation_id=None) -> bool:
        """Retire a placement after two fresh positive board-stock observations.

        The original requirement and available count are recorded before the
        purchase gesture. A fulfilled status from a different requirement is
        deliberately insufficient evidence. Returning ``True`` means this call
        retired the pending placement and a later exhausted bush may be replaced.
        """
        self._check()
        entry = self.state["items"].get(key)
        if not isinstance(entry, dict) or entry.get("stage") not in self._STAGES:
            return False
        before = entry.get("inventory_before")
        if not isinstance(before, dict):
            return False
        previous = before.get("available")
        same_requirement = (
            type(required) is int and required > 0 and before.get("required") == required
        )
        increased = (
            same_requirement
            and type(available) is int
            and type(previous) is int
            and available > previous
        )
        became_fulfilled = (
            same_requirement
            and before.get("status") == "missing"
            and status == "fulfilled"
        )
        positive = increased or became_fulfilled
        confirmation = entry.get("stock_confirmation")
        if not positive:
            if confirmation is not None:
                entry.pop("stock_confirmation", None)
                entry["updated_at"] = datetime.now(UTC).isoformat()
                _save_json(self.state_path, self.state)
            return False
        if observation_id is None or observation_id == before.get("captured_at"):
            return False
        if isinstance(confirmation, dict) and confirmation.get("observation_id") == observation_id:
            return False
        evidence = [available, required, status]
        count = (
            int(confirmation.get("count", 0)) + 1
            if isinstance(confirmation, dict) and confirmation.get("evidence") == evidence
            else 1
        )
        observed_at = datetime.now(UTC).isoformat()
        stock_confirmation = {
            "evidence": evidence,
            "observation_id": observation_id,
            "count": count,
        }
        if count < 2:
            self._record(
                key,
                entry["stage"],
                stock_confirmation=stock_confirmation,
            )
            return False
        retired = {
            "item": key,
            "species": entry.get("species", "raspberry"),
            "operation": entry.get("operation"),
            "previous_stage": entry["stage"],
            "reason": "positive_stock_confirmed",
            "stock_evidence": evidence,
            "retired_at": observed_at,
        }
        self.state["history"] = [*self.state.get("history", []), retired][-100:]
        del self.state["items"][key]
        _save_json(self.state_path, self.state)
        return True

    def _observe(self, frame, icon) -> OrchardObservation | None:
        self._check()
        observed = self.vision.observe(frame.png, icon, cancel=self.cancel_event.is_set)
        self._check()
        return observed

    @staticmethod
    def _same(first: VisualTarget | None, second: VisualTarget | None,
              frame, tolerance: float = .10) -> bool:
        if first is None or second is None:
            return False
        return (math.dist(first.center, second.center) <= max(5, first.width * tolerance)
                and abs(first.width - second.width) <= max(3, first.width * .08)
                and abs(first.height - second.height) <= max(3, first.height * .08)
                and 0 <= first.x < frame.width and 0 <= first.y < frame.height)

    @classmethod
    def _same_shop(cls, first: OrchardObservation, second: OrchardObservation, frame) -> bool:
        return (first.species == second.species and first.shop_open and second.shop_open
                and cls._same(first.header, second.header, frame)
                and cls._same(first.card, second.card, frame)
                and cls._same(first.card_arrow, second.card_arrow, frame)
                and cls._same(first.tree_tab, second.tree_tab, frame))

    @classmethod
    def _stable_destination(cls, first: OrchardObservation, second: OrchardObservation,
                            frame) -> VisualTarget | None:
        candidates = [(left, right) for left in first.clear_targets for right in second.clear_targets
                      if cls._same(left, right, frame, .40)]
        if not candidates:
            return None
        left, right = max(candidates, key=lambda pair: (min(pair[0].score, pair[1].score),
                                                        -math.dist(pair[0].center, pair[1].center)))
        return right if right.score >= left.score * .75 else left

    def _flags(self, frame, icon) -> tuple[bool, bool]:
        observation = self._observe(frame, icon)
        catalog = bool(observation and observation.shop_open)
        mode = self.vision.placement_mode(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        return catalog, mode

    def _cancel_mode(self, frame, icon, *, invalid_seen: bool = False):
        """Use only a twice-observed shop control to leave an invalid placement mode."""
        first = self._observe(frame, icon)
        if first is None or first.shop_open or first.shop_button is None:
            catalog, mode = self._flags(frame, icon)
            return frame, catalog, mode or (invalid_seen and not catalog)
        fresh = self._frame()
        second = self._observe(fresh, icon)
        mode = self.vision.placement_mode(fresh.png, cancel=self.cancel_event.is_set)
        invalid = self.vision.invalid_placement(fresh.png, cancel=self.cancel_event.is_set)
        self._check()
        if (second is None or second.shop_open or not (mode or (invalid_seen and invalid))
                or not self._same(first.shop_button, second.shop_button, fresh)):
            catalog, remaining = self._flags(fresh, icon)
            return fresh, catalog, remaining or (invalid_seen and not catalog)
        self.progress("Closing the confirmed orchard placement mode through the shop control.")
        self.client.tap(*second.shop_button.center, width=fresh.width, height=fresh.height)
        self._wait(.6)
        opened = self._frame()
        opened_observation = self._observe(opened, icon)
        if opened_observation is None or not opened_observation.shop_open:
            catalog, remaining = self._flags(opened, icon)
            # The tap may have reached the device even when the resulting UI
            # is not recognizable. Keep the placement guard raised unless the
            # opened catalog itself proves the purchase ghost was cancelled.
            return opened, catalog, remaining or not catalog
        self._wait(.35)
        checked = self._frame()
        checked_observation = self._observe(checked, icon)
        if (checked_observation is None or not self._same_shop(opened_observation, checked_observation, checked)
                or not self._same(opened_observation.shop_button, checked_observation.shop_button, checked)):
            catalog, remaining = self._flags(checked, icon)
            return checked, catalog, remaining
        self.client.tap(*checked_observation.shop_button.center, width=checked.width, height=checked.height)
        self._wait(.45)
        final = self._frame()
        catalog, remaining = self._flags(final, icon)
        return final, catalog, remaining

    def work_if_recognized(self, frame, icon, key, baseline=None) -> ResourceResult | None:
        self._check()
        self._size = frame.width, frame.height
        first = self._observe(frame, icon)
        if first is None:
            return None
        entry = self.state["items"].get(key)
        if entry is None:
            # Recipe and board crops can have separate learned identities. A
            # positively identified species shares its existing placement intent.
            pending = [(identity, saved) for identity, saved in self.state['items'].items()
                       if saved.get('species') == first.species]
            if len(pending) > 1:
                return ResourceResult('unsupported', 'Multiple saved orchard placements need reconciliation.')
            if pending:
                key, entry = pending[0]
        if entry:
            if entry["stage"] == "placement_attempted":
                self._reconcile_saved_placement(key, entry, icon)
                entry = self.state["items"][key]
            stage = entry["stage"]
            return ResourceResult("waiting",
                "A Raspberry bush placement is pending verification." if stage == "placement_attempted" else
                "The newly placed Raspberry bush is growing; no duplicate bush will be purchased.",
                {"item": key, "species": "raspberry", "orchard_pending": True,
                 "pending_stage": stage, "defer_session": True,
                 "catalog_open": first.shop_open,
                 "placement_mode": self.vision.placement_mode(
                     frame.png, cancel=self.cancel_event.is_set),
                 "operation": entry.get("operation")})

        shop_frame, shop = frame, first
        if not shop.shop_open:
            if shop.shop_button is None:
                return ResourceResult("unsupported",
                    "Raspberry was recognized, but the shop control was not uniquely verified.",
                    {"item": key, "species": "raspberry", "catalog_open": False,
                     "placement_mode": False})
            fresh = self._frame()
            checked = self._observe(fresh, icon)
            if checked is None or not self._same(shop.shop_button, checked.shop_button, fresh):
                return ResourceResult("changed",
                    "The shop control changed before orchard navigation; no input was sent.",
                    {"item": key, "species": "raspberry", "catalog_open": False,
                     "placement_mode": False})
            self.progress("Opening the freshly verified Trees & Bushes catalog for Raspberry.")
            self.client.tap(*checked.shop_button.center, width=fresh.width, height=fresh.height)
            for _ in range(5):
                self._wait(.35)
                shop_frame = self._frame()
                shop = self._observe(shop_frame, icon)
                if shop is not None and shop.shop_open:
                    break
            else:
                catalog, mode = self._flags(shop_frame, icon)
                return ResourceResult("unsupported",
                    "The Raspberry catalog did not become fully recognizable after one shop tap.",
                    {"item": key, "species": "raspberry", "catalog_open": catalog,
                     "placement_mode": mode})

        self._wait(.35)
        fresh = self._frame()
        checked = self._observe(fresh, icon)
        if checked is None or not self._same_shop(shop, checked, fresh):
            catalog, mode = self._flags(fresh, icon)
            return ResourceResult("changed",
                "The Raspberry card or Trees & Bushes header changed before placement; no swipe was sent.",
                {"item": key, "species": "raspberry", "catalog_open": catalog,
                 "placement_mode": mode})
        destination = self._stable_destination(shop, checked, fresh)
        if destination is None:
            return ResourceResult("unsupported",
                "No full Raspberry-bush footprint was clear grass in two stable catalog frames.",
                {"item": key, "species": "raspberry", "catalog_open": True,
                 "placement_mode": False})

        operation = uuid.uuid4().hex
        evidence = self.state_path.parent / "orchard_evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        before_path = evidence / f"{operation}_before.png"
        before_path.write_bytes(fresh.png)
        # This write precedes the only input that can charge coins. A failed ADB
        # command remains uncertain because the device may have received it.
        self._record(key, "placement_attempted", operation=operation, species="raspberry",
                     inventory_before=dict(baseline) if isinstance(baseline, dict) else {},
                     before_capture=before_path.name,
                     before_sha256=hashlib.sha256(fresh.png).hexdigest(),
                     destination_evidence=list(destination.box),
                     panel_right_evidence=checked.panel_right or 0)
        self._check()
        self.progress("Placing one Raspberry bush on freshly verified clear grass.")
        self.client.swipe(*checked.drag_start.center, *destination.center,
                          width=fresh.width, height=fresh.height, duration_ms=850)
        self._wait(.65)
        after_first = self._frame()
        self._wait(.40)
        after_second = self._frame()
        for suffix, observed in (("after_1", after_first), ("after_2", after_second)):
            path = evidence / f"{operation}_{suffix}.png"
            path.write_bytes(observed.png)
            self._record(key, "placement_attempted", **{
                f"{suffix}_capture": path.name,
                f"{suffix}_sha256": hashlib.sha256(observed.png).hexdigest(),
            })

        invalid = (self.vision.invalid_placement(after_first.png, cancel=self.cancel_event.is_set)
                   or self.vision.invalid_placement(after_second.png, cancel=self.cancel_event.is_set))
        self._check()
        confirmed = False if invalid else self.vision.placement_confirmed(
            fresh.png, (after_first.png, after_second.png), destination,
            checked.panel_right or 0, cancel=self.cancel_event.is_set,
        )
        self._check()
        mode = self.vision.placement_mode(after_second.png, cancel=self.cancel_event.is_set)
        after_observation = self._observe(after_second, icon)
        catalog_open = bool(after_observation and after_observation.shop_open)
        placement_mode = mode
        if invalid or mode:
            _, catalog_open, placement_mode = self._cancel_mode(
                after_second, icon, invalid_seen=invalid,
            )

        if confirmed and not placement_mode:
            self._record(key, "growing", placement_confirmed_at=datetime.now(UTC).isoformat(),
                         catalog_open=catalog_open, placement_mode=placement_mode)
            return ResourceResult("waiting",
                "One Raspberry bush was placed and its new growth was confirmed in two frames.",
                {"item": key, "species": "raspberry", "orchard_pending": True,
                 "pending_stage": "growing", "operation": operation, "defer_session": True,
                 "catalog_open": catalog_open, "placement_mode": placement_mode,
                 "invalid_placement": invalid, "placement_confirmed": confirmed})
        reason = ("The game reported that the chosen footprint lacked space. " if invalid else
                  "The new bush or its safe exit state could not be positively confirmed. ")
        return ResourceResult("waiting", reason +
            "The saved placement intent blocks another purchase until it is inspected.",
            {"item": key, "species": "raspberry", "orchard_pending": True,
             "pending_stage": "placement_attempted", "operation": operation,
             "defer_session": True, "catalog_open": catalog_open,
             "placement_mode": placement_mode, "invalid_placement": invalid,
             "placement_confirmed": confirmed})

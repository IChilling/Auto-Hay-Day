"""Read-only order observations in the coordinate system of a verified order panel.

Quantity glyph geometry locates every visible item row; no three-item assumption
or farm coordinate is used. Controls are observations, never permission to act.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.panel import Bounds, PanelResult, PanelVerifier
from hayday.quantities import read_quantity

Point = tuple[int, int]
TICKET_FINGERPRINT_SCHEME = 'reward_rows_v2'


@dataclass(frozen=True)
class ControlObservation:
    kind: str
    bounds: Bounds
    center: Point
    score: float


@dataclass(frozen=True)
class TicketObservation:
    slot_id: int
    bounds: Bounds
    center: Point
    ready: bool
    score: float
    fingerprint: str
    sent: bool = False
    sent_fingerprint: str = ""


@dataclass(frozen=True)
class ItemSlot:
    index: int
    bounds: Bounds
    center: Point
    status: str
    red_quantity: bool
    check_score: float
    fingerprint: str
    quantity_bounds: Bounds
    icon_bounds: Bounds
    available: int | None = None
    required: int | None = None


@dataclass(frozen=True)
class SelectedOrder:
    items: tuple[ItemSlot, ...]
    fingerprint: str
    ready: bool
    complete: bool
    obstructed: bool = False


@dataclass(frozen=True)
class OrderObservation:
    panel: PanelResult
    tickets: tuple[TicketObservation, ...] = ()
    selected: SelectedOrder | None = None
    send_button: ControlObservation | None = None
    double_offer: ControlObservation | None = None
    reason: str = ""
    bonus_prompt: ControlObservation | None = None
    navigation: ControlObservation | None = None


@dataclass(frozen=True)
class _Marker:
    image: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class _Found:
    box: Bounds
    score: float


def fingerprints_match(first: str, second: str, max_distance: int = 12) -> bool:
    """Compare perceptual visual IDs; different lengths/empty IDs never match."""
    if not first or len(first) != len(second):
        return False
    try:
        if len(first) % 16:
            return False
        distances = [(int(first[i:i+16], 16) ^ int(second[i:i+16], 16)).bit_count()
                     for i in range(0, len(first), 16)]
        return max(distances) <= max_distance and sum(distances)/len(distances) <= max_distance/2
    except ValueError:
        return False


class OrderReader:
    """Classify known visual states, failing closed on unknown/occluded layouts."""

    def __init__(self, assets_dir: Path | None = None, verifier=None):
        self.verifier = verifier if verifier is not None else PanelVerifier(assets_dir)
        self.assets_dir = Path(assets_dir) if assets_dir is not None else self.verifier.assets_dir
        self._markers: dict[str, _Marker] = {}
        self._error = ""
        try:
            manifest = json.loads((self.assets_dir / "order_manifest.json").read_text())
            if not isinstance(manifest, dict) or manifest.get("reference_size") != [1920, 1080]:
                raise ValueError("Order references have an unsupported source layout")
            required = ("item_check", "ticket_check", "ticket_check_tilted", "quantity_slash", "send", "double_offer",
                        "bonus_symbols", "bonus_send", "sent_stamp", "ad_close", "navigation")
            for name in required:
                entry = manifest["features"][name]
                raw = cv2.imdecode(np.frombuffer(
                    (self.assets_dir / entry["file"]).read_bytes(), np.uint8
                ), cv2.IMREAD_UNCHANGED)
                if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
                    raise ValueError(f"{name} must contain RGBA reference pixels")
                mask = (raw[:, :, 3] >= 200).astype(np.uint8)*255
                if np.count_nonzero(mask) < 40:
                    raise ValueError(f"{name} has too few included pixels")
                self._markers[name] = _Marker(raw[:, :, :3], mask)
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self._error = f"Order references unavailable: {exc}"

    @property
    def reference_error(self) -> str:
        return self._error or getattr(self.verifier, "reference_error", "")

    @staticmethod
    def _decode(png: bytes) -> np.ndarray | None:
        try:
            return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, cv2.error):
            return None

    @staticmethod
    def _variant(marker: _Marker, scale: float, angle: float) -> _Marker:
        alpha = marker.mask.astype(np.float32)/255
        rgb = marker.image.astype(np.float32)*alpha[:, :, None]
        size = (max(3, round(alpha.shape[1]*scale)), max(3, round(alpha.shape[0]*scale)))
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if angle:
            h, w = alpha.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            target = (round(w*1.45), round(h*1.45))
            matrix[:, 2] += ((target[0]-w)/2, (target[1]-h)/2)
            alpha = cv2.warpAffine(alpha, matrix, target)
            rgb = cv2.warpAffine(rgb, matrix, target)
        image = np.clip(rgb/np.maximum(alpha[:, :, None], 1e-6), 0, 255).astype(np.uint8)
        return _Marker(image, (alpha >= 0.85).astype(np.uint8)*255)

    def _find(
        self, name: str, frame: np.ndarray, region: tuple[int, int, int, int], *,
        threshold: float = 0.84, count: int = 1, scales=(1.0,), angles=(0.0,),
    ) -> list[_Found]:
        x1, y1, x2, y2 = region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        found: list[_Found] = []
        for scale in scales:
            for angle in angles:
                marker = self._variant(self._markers[name], scale, angle)
                height, width = marker.mask.shape
                if roi.shape[0] < height or roi.shape[1] < width:
                    continue
                scores = cv2.matchTemplate(
                    roi, marker.image, cv2.TM_CCOEFF_NORMED, mask=marker.mask
                )
                scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
                for _ in range(count):
                    _, score, _, point = cv2.minMaxLoc(scores)
                    if score < threshold:
                        break
                    x, y = point
                    found.append(_Found((x+x1, y+y1, width, height), min(1.0, score)))
                    radius = max(12, min(width, height)//2)
                    scores[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1
        unique = []
        for candidate in sorted(found, key=lambda item: item.score, reverse=True):
            x, y, width, height = candidate.box
            center = (x+width/2, y+height/2)
            if any(abs(center[0]-(other.box[0]+other.box[2]/2)) < max(width, other.box[2])*0.55
                   and abs(center[1]-(other.box[1]+other.box[3]/2)) < max(height, other.box[3])*0.55
                   for other in unique):
                continue
            unique.append(candidate)
            if len(unique) >= count:
                break
        return unique

    @staticmethod
    def _fingerprint(image: np.ndarray) -> str:
        if image.size == 0:
            return ""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        frequency = cv2.dct(tiny)[:8, :8]
        bits = frequency > np.median(frequency.reshape(-1)[1:])
        bits[0, 0] = False
        return np.packbits(bits).tobytes().hex()

    @classmethod
    def _quantity_fingerprint(cls, image: np.ndarray) -> str:
        mask, _ = cls._quantity_masks(image)
        points = cv2.findNonZero(mask)
        if points is None:
            return "0"*16
        x, y, width, height = cv2.boundingRect(points)
        normalized = cv2.copyMakeBorder(mask[y:y+height, x:x+width], 3, 3, 3, 3,
                                       cv2.BORDER_CONSTANT, value=0)
        coverage = cv2.resize(normalized, (8, 8), interpolation=cv2.INTER_AREA)
        return np.packbits(coverage > 85).tobytes().hex()

    @classmethod
    def _ticket_fingerprint(cls, frame: np.ndarray, cx: int, cy: int) -> str:
        crop = frame[max(187, cy-125):min(997, cy+125), max(345, cx-135):min(1187, cx+150)]
        channels = crop.astype(np.int16)
        # Neutral black excludes the blue coin shadow and XP-star edges.
        ink = ((channels.max(axis=2) < 110) &
               (channels.max(axis=2)-channels.min(axis=2) < 30)).astype(np.uint8)*255
        _, labels, stats, centers = cv2.connectedComponentsWithStats(ink)
        characters = [(index, centers[index][1]) for index, (_, _, width, height, area)
                      in enumerate(stats[1:], 1) if area > 180 and 35 <= height <= 100 and
                      10 <= width <= 175 and 20 < centers[index][0] < 210]
        # Reward text may connect as one word or separate into individual digits
        # as the ticket tilts. Group complete rows before deskewing their ink.
        rows = []
        for index, center_y in sorted(characters, key=lambda item: item[1]):
            row = next((row for row in rows if abs(center_y-np.mean([y for _, y in row])) < 32), None)
            if row is None:
                rows.append([(index, center_y)])
            else:
                row.append((index, center_y))
        if len(rows) < 2:
            return ""
        signatures = []
        for row in rows[:2]:
            word = np.isin(labels, [index for index, _ in row]).astype(np.uint8)*255
            # Retain small enclosed counters, e.g. the interior of 9, without
            # admitting unrelated components outside the actual reward row.
            wx, wy, ww, wh = cv2.boundingRect(cv2.findNonZero(word))
            for index, (ix, iy, iw, ih, area) in enumerate(stats[1:], 1):
                if area < 180 and wx <= ix and wy <= iy and ix+iw <= wx+ww and iy+ih <= wy+wh:
                    word[labels == index] = 255
            corners = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(word))).astype(np.float32)
            sums, differences = corners.sum(axis=1), np.diff(corners, axis=1).ravel()
            ordered = np.array([corners[np.argmin(sums)], corners[np.argmin(differences)],
                                corners[np.argmax(sums)], corners[np.argmax(differences)]], np.float32)
            matrix = cv2.getPerspectiveTransform(
                ordered, np.array([[2, 2], [125, 2], [125, 61], [2, 61]], np.float32)
            )
            # Deskew the reward glyphs themselves. Neighboring paper can overlap
            # during the selection animation and must not control orientation.
            normalized = cv2.warpPerspective(word, matrix, (128, 64))
            signatures.append(cls._fingerprint(cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)))
        return "".join(signatures)

    @staticmethod
    def _quantity_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
        brown = ((r > 70) & (r < 170) & (g > 35) & (g < 135) &
                 (b > 10) & (b < 110) & (r-g > 18) & (g-b > 12))
        red = (r > 215) & (g > 35) & (g < 135) & (b < 110) & (r-g > 100)
        return (brown | red).astype(np.uint8)*255, red

    def _quantities(self, frame: np.ndarray) -> list[tuple[Bounds, int, bool]]:
        mask, red = self._quantity_masks(frame)
        roi = mask[408:740, 1205:1727]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
        components = [
            (index, int(x)+1205, int(y)+408, int(w), int(h), int(area))
            for index, (x, y, w, h, area) in enumerate(stats[1:], start=1)
            if 13 <= h <= 44 and 5 <= w <= 40 and area >= 35
        ]
        expected = cv2.resize(self._markers["quantity_slash"].mask, (22, 30)) > 0
        found = []
        for label, x, y, width, height, _ in components:
            if not 0.46 < width/height < 0.9:
                continue
            component = (labels[y-408:y-408+height, x-1205:x-1205+width] == label).astype(np.uint8)
            resized = cv2.resize(component, (22, 30), interpolation=cv2.INTER_NEAREST) > 0
            overlap = np.sum(expected & resized)/max(1, np.sum(expected | resized))
            if overlap < 0.69:
                continue
            neighbors = [item for item in components if abs(item[2]-y) <= max(4, height*.16)
                         and abs(item[4]-height) <= max(4, height*.20)]
            left, right = x, x+width
            left_count = right_count = 0
            for item in sorted(neighbors, key=lambda item: item[1], reverse=True):
                _, nx, _, nw, _, _ = item
                if nx+nw <= left and 0 <= left-(nx+nw) <= max(8, height*.38):
                    left, left_count = nx, left_count+1
            for item in sorted(neighbors, key=lambda item: item[1]):
                _, nx, _, nw, _, _ = item
                if nx >= right and 0 <= nx-right <= max(8, height*.38):
                    right, right_count = nx+nw, right_count+1
            if not left_count or not right_count:
                continue
            quantity = (left, y-2, right-left, height+4)
            missing = int(np.sum(red[y-2:y+height+2, left:x])) >= max(12, height*2)
            if not any(abs(left-other[0][0]) < 20 and abs(y-other[0][1]) < 12 for other in found):
                found.append((quantity, x+width, missing))
        return sorted(found, key=lambda item: (round(item[0][1]/35), item[0][0]))

    def observe(self, png: bytes, cancel: Callable[[], bool] | None = None) -> OrderObservation:
        stopped = cancel or (lambda: False)
        if self.reference_error:
            return OrderObservation(PanelResult(False, self.reference_error), reason=self.reference_error)
        panel = self.verifier.verify(png, cancel=cancel)
        if not panel.verified or panel.bounds is None or stopped():
            return OrderObservation(panel, reason="Order panel unavailable or reading cancelled.")
        original = self._decode(png)
        if original is None:
            return OrderObservation(panel, reason="Screenshot cannot be decoded.")
        x, y, width, height = panel.bounds
        scale = (width/1725 + height/1015)/2
        tx, ty = x-125*scale, y-25*scale
        canonical = cv2.warpAffine(original, np.array(
            [[1/scale, 0, -tx/scale], [0, 1/scale, -ty/scale]], np.float32
        ), (1920, 1080))

        def bounds(box: Bounds) -> Bounds:
            bx, by, bw, bh = box
            return round(tx+bx*scale), round(ty+by*scale), round(bw*scale), round(bh*scale)

        def control(kind: str, found: _Found | None) -> ControlObservation | None:
            if found is None:
                return None
            box = bounds(found.box)
            return ControlObservation(kind, box, (box[0]+box[2]//2, box[1]+box[3]//2), found.score)

        def single(name: str, region, **kwargs):
            matches = self._find(name, canonical, region, **kwargs)
            return matches[0] if matches else None

        offer = single("double_offer", (1600, 750, 1840, 995), threshold=0.82,
                       scales=(0.9, 0.94, 0.97, 1.0, 1.03, 1.06, 1.09, 1.12))
        bonus_send = single("bonus_send", (1530, 638, 1915, 810), threshold=0.86)
        bonus_symbols = single("bonus_symbols", (1560, 380, 1915, 525), threshold=0.85)
        bonus_prompt = control("bonus_send", bonus_send) if bonus_send and bonus_symbols else None
        send = single("send", (1340, 812, 1735, 1030), threshold=0.82,
                      scales=(0.94, 0.97, 1.0, 1.03, 1.06, 1.09))
        navigation = single("navigation", (1180, 100, 1918, 820), threshold=0.88)
        if stopped():
            return OrderObservation(panel, reason="Order reading cancelled.")
        checks = self._find(
            "ticket_check", canonical, (360, 190, 1195, 1000), threshold=0.81, count=9,
            angles=(-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25),
        )
        checks += self._find(
            "ticket_check_tilted", canonical, (360, 190, 1195, 1000), threshold=0.86, count=9,
        )
        stamps = self._find("sent_stamp", canonical, (355, 190, 1160, 1000),
                            threshold=0.88, count=9)
        tickets = []
        # These are coordinates within the verified modal, independent of farm position.
        for row in range(3):
            for column in range(3):
                cx, cy = 511+255*column, 329+267*row
                patch = canonical[cy-100:cy+103, cx-75:cx+82]
                b, g, r = (patch[:, :, i].astype(np.int16) for i in range(3))
                paper = (r > 175) & (g > 165) & (b > 100) & (r-b > 15)
                if np.mean(paper) < 0.38:
                    continue
                nearby = [item for item in checks if
                          abs(item.box[0]+item.box[2]/2-(cx+100)) < 100 and
                          abs(item.box[1]+item.box[3]/2-(cy+90)) < 105]
                stamp = [item for item in stamps if
                         abs(item.box[0]+item.box[2]/2-cx) < 100 and
                         abs(item.box[1]+item.box[3]/2-cy) < 115]
                box = bounds((cx-123, cy-123, 246, 246))
                stamp_fingerprint = ""
                if stamp:
                    sx, sy, sw, sh = stamp[0].box
                    stamp_fingerprint = self._fingerprint(canonical[sy:sy+sh, sx:sx+sw])
                tickets.append(TicketObservation(
                    row*3+column, box, (round(tx+cx*scale), round(ty+cy*scale)),
                    bool(nearby) and not bool(stamp),
                    max((item.score for item in nearby+stamp), default=0.0),
                    self._ticket_fingerprint(canonical, cx, cy), bool(stamp), stamp_fingerprint,
                ))
        quantities = self._quantities(canonical)
        item_checks = self._find(
            "item_check", canonical, (1205, 403, 1729, 740), threshold=0.85, count=12,
            scales=(0.65, 0.8, 1.0),
        )
        items = []
        signatures = []
        assigned_checks = set()
        for index, (quantity, required_x, missing) in enumerate(quantities):
            qx, qy, qw, qh = quantity
            # Quantized font size resists one-pixel resampling differences while
            # retaining smaller layouts when the game displays more item rows.
            ratio = max(15, round((qh-4)/5)*5)/30
            center_x = qx+qw/2
            nearby = [(i, match) for i, match in enumerate(item_checks) if
                      abs((match.box[0]+match.box[2]/2)-(center_x+37*ratio)) < 32*ratio and
                      abs((match.box[1]+match.box[3]/2)-(qy-88*ratio)) < 28*ratio]
            check_score = max((match.score for _, match in nearby), default=0.0)
            assigned_checks.update(i for i, _ in nearby)
            status = "missing" if missing else "fulfilled" if nearby else "unknown"
            icon_box = (round(center_x-65*ratio), round(qy-118*ratio),
                        round(130*ratio), round(150*ratio))
            box = bounds(icon_box)
            # Icon center excludes top-right check, top-left flame, and stock counts.
            icon = canonical[round(qy-71*ratio):round(qy-10*ratio),
                             round(center_x-48*ratio):round(center_x+29*ratio)]
            required = canonical[qy:qy+qh, required_x:qx+qw]
            fingerprint = self._fingerprint(icon)+self._quantity_fingerprint(required)
            signatures.append(fingerprint)
            parsed_quantity = read_quantity(png, bounds(quantity))
            items.append(ItemSlot(
                index, box, (round(tx+center_x*scale), round(ty+(qy-58*ratio)*scale)),
                status, missing, check_score, fingerprint, bounds(quantity),
                bounds((round(center_x-48*ratio), round(qy-71*ratio),
                        round(77*ratio), round(61*ratio))),
                parsed_quantity.available if parsed_quantity else None,
                parsed_quantity.required if parsed_quantity else None,
            ))
        # A visible item whose quantity could not be read must not silently vanish
        # from an otherwise fulfilled order. Inspect occupied cells independently.
        if quantities:
            font_ratio = max(15, round(np.median([q[0][3]-4 for q in quantities])/5)*5)/30
            rows = []
            for quantity, _, _ in quantities:
                if not any(abs(quantity[1]-row) < 20 for row in rows):
                    rows.append(quantity[1])
            rows.sort()
            step = float(np.median(np.diff(rows))) if len(rows) > 1 else 160*font_ratio
            if step > 50:
                while rows[0]-step >= 440:
                    rows.insert(0, round(rows[0]-step))
                while rows[-1]+step <= 725:
                    rows.append(round(rows[-1]+step))
            for row_y in rows:
                for center_x in (1300, 1465, 1630):
                    if any(abs((q[0]+q[2]/2)-center_x) < 65 and abs(q[1]-row_y) < 25
                           for q, _, _ in quantities):
                        continue
                    left, top = round(center_x-50*font_ratio), round(row_y-95*font_ratio)
                    right, bottom = round(center_x+45*font_ratio), round(row_y-12*font_ratio)
                    crop = canonical[top:bottom, left:right]
                    if not crop.size:
                        continue
                    b, g, r = (crop[:, :, channel] for channel in range(3))
                    foreground = (r < 210) | (g < 185) | (b < 130)
                    if np.mean(foreground) < 0.075:
                        continue
                    icon_box = bounds((left, top, right-left, bottom-top))
                    items.append(ItemSlot(
                        len(items), icon_box,
                        (round(tx+center_x*scale), round(ty+(row_y-58*font_ratio)*scale)),
                        "unknown", False, 0.0, self._fingerprint(crop),
                        bounds((round(center_x-40*font_ratio), row_y,
                                round(80*font_ratio), round(34*font_ratio))), icon_box,
                    ))
        obstructed = bonus_prompt is not None or navigation is not None
        complete = (bool(items) and all(item.status != "unknown" for item in items)
                    and len(assigned_checks) == len(item_checks) and not obstructed)
        ready = complete and all(item.status == "fulfilled" for item in items)
        ready = ready and any(ticket.ready for ticket in tickets) and send is not None
        selected = None
        if items:
            stable_header = canonical[240:387, 1280:1680]
            fingerprint = self._fingerprint(stable_header)+"".join(signatures)
            selected = SelectedOrder(tuple(items), fingerprint, ready, complete, obstructed)
        return OrderObservation(
            panel, tuple(tickets), selected, control("send", send), control("double_offer", offer),
            "Order read from verified panel." if complete else "No complete unobstructed order selected.",
            bonus_prompt, control("navigation", navigation),
        )

    @staticmethod
    def item_icon(png: bytes, slot: ItemSlot) -> bytes:
        """Return native item-body pixels for contextual resource matching."""
        frame = OrderReader._decode(png)
        if frame is None:
            raise ValueError("Item icon requires a valid screenshot")
        x, y, width, height = slot.icon_bounds
        if x < 0 or y < 0 or x+width > frame.shape[1] or y+height > frame.shape[0]:
            raise ValueError("Item icon is outside the screenshot")
        success, encoded = cv2.imencode(".png", frame[y:y+height, x:x+width])
        if not success:
            raise ValueError("Item icon could not be encoded")
        return encoded.tobytes()

    def ad_close(self, png: bytes) -> ControlObservation | None:
        """Recognize only the observed ad SDK gray circular X in its screen corner.

        The caller must establish an active rewarded-ad state and enforce its own
        wait and consecutive-frame checks. This method does not inspect ad content.
        """
        if self.reference_error:
            return None
        frame = self._decode(png)
        if frame is None:
            return None
        height, width = frame.shape[:2]
        scale = min(width/1920, height/1080)
        matches = self._find(
            "ad_close", frame, (round(width*.90), 0, width, round(height*.12)),
            threshold=0.92, scales=(scale*.9, scale, scale*1.1),
        )
        if not matches:
            return None
        match = matches[0]
        x, y, w, h = match.box
        # A gray neutral surface distinguishes the SDK X from colored game controls.
        pixel = frame[y+h//2, x+w//4].astype(np.int16)
        if int(pixel.max()-pixel.min()) > 18 or not 30 < float(np.mean(pixel)) < 155:
            return None
        return ControlObservation("ad_close", match.box, (x+w//2, y+h//2), match.score)

"""Read resource navigation and recipe controls without sending game input.

References come from observed game UI pixels. World-icon matches are candidates,
not evidence of harvestability or collection; inventory changes must verify work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class VisualTarget:
    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def box(self) -> Box:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class RecipeRow:
    box: Box
    item_box: Box
    quantity_box: Box
    missing: bool | None
    icon_png: bytes
    navigation: VisualTarget


@dataclass(frozen=True)
class ResourcePopup:
    box: Box
    title_png: bytes
    rows: tuple[RecipeRow, ...]
    navigation: tuple[VisualTarget, ...]
    recipe_complete: bool = False

    @property
    def title_fingerprint(self) -> str:
        """Exact captured-title identity; use visual matching across resolutions."""
        return hashlib.sha256(self.title_png).hexdigest()[:16]


@dataclass(frozen=True)
class ResourceScene:
    popups: tuple[ResourcePopup, ...]
    empty_slots: tuple[VisualTarget, ...]
    arrows: tuple[VisualTarget, ...]


def _decode(png: bytes, alpha: bool = False) -> np.ndarray:
    try:
        image = cv2.imdecode(
            np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR
        )
    except cv2.error as exc:
        raise ValueError("Resource vision requires a valid screenshot or icon PNG.") from exc
    if image is None:
        raise ValueError("Resource vision requires a valid screenshot or icon PNG.")
    return image


def _png(image: np.ndarray) -> bytes:
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode a resource image crop.")
    return data.tobytes()


def _contains(box: Box, point: tuple[int, int]) -> bool:
    x, y, width, height = box
    return x <= point[0] < x + width and y <= point[1] < y + height


class ResourceVision:
    def __init__(self, reference_path: Path | None = None):
        if reference_path is None:
            source = Path(__file__).resolve().parents[2] / "images" / "resources"
            reference_path = source if source.is_dir() else Path(__file__).parent / "assets/resources"
        self.reference_path = Path(reference_path)
        try:
            self._arrow = _decode((self.reference_path / "navigation_arrow.png").read_bytes(), True)
            self._empty = _decode((self.reference_path / "empty_label.png").read_bytes(), True)
        except OSError as exc:
            raise ValueError("Resource navigation or EMPTY label reference is missing.") from exc
        for reference in (self._arrow, self._empty):
            if reference.ndim != 3 or reference.shape[2] != 4 or np.count_nonzero(
                reference[:, :, 3] >= 200
            ) < 30:
                raise ValueError("Resource references require original pixels and alpha masks.")
        self._products = []
        registry = self.reference_path/'products/manifest.json'
        if registry.is_file():
            manifest = json.loads(registry.read_text('utf-8'))
            if manifest.get('version') != 1:
                raise ValueError('Unsupported production reference manifest.')
            for product in manifest['products']:
                files = [product[name] for name in ('image', 'identifier', 'title')]
                if any(not isinstance(name, str) or Path(name).name != name for name in files):
                    raise ValueError('Production reference names must be local filenames.')
                artwork, identifier, title = [(registry.parent/name).read_bytes() for name in files]
                rgba = _decode(artwork, True)
                if rgba.ndim != 3 or rgba.shape[2] != 4 or np.count_nonzero(rgba[:, :, 3]) < 100:
                    raise ValueError('Production artwork requires an alpha inclusion mask.')
                height = product['reference_height']
                if type(height) is not int or height <= 0:
                    raise ValueError('Production reference height must be a positive integer.')
                self._products.append((rgba, identifier, title, height))

    def production_reference(self, icon: bytes, title: bytes | None, screen_height: int) -> bytes | None:
        """Use complete product artwork after identifying the requested item.

        Title evidence takes priority. Without a title, a strong match against
        the registered board/ingredient identifier is required. Farm position
        and machine position never participate in identity or matching.
        """
        matches = []
        for rgba, identifier, expected_title, reference_height in self._products:
            if title is not None:
                identified = self.titles_match(title, expected_title)
            else:
                found = self.find_item(icon, identifier, min_scale=.5, max_scale=1.8)
                identified = bool(found and found[0].score >= .95)
            if identified:
                color, mask = self._scaled(rgba, screen_height/reference_height)
                matches.append(_png(np.dstack((color, mask))))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _scaled(rgba: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        height, width = rgba.shape[:2]
        shape = max(5, round(width * scale)), max(5, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        alpha = rgba[:, :, 3].astype(np.float32) / 255
        scaled_alpha = cv2.resize(alpha, shape, interpolation=interpolation)
        color = cv2.resize(
            rgba[:, :, :3].astype(np.float32) * alpha[:, :, None], shape,
            interpolation=interpolation,
        ) / np.maximum(scaled_alpha[:, :, None], 1e-6)
        return np.clip(color, 0, 255).astype(np.uint8), (scaled_alpha >= 0.8).astype(np.uint8) * 255

    def _search(
        self, frame: np.ndarray, rgba: np.ndarray, scales: np.ndarray,
        threshold: float, cancel: Callable[[], bool], max_peaks: int = 8,
    ) -> tuple[VisualTarget, ...]:
        height, width = frame.shape[:2]
        ratio = min(1.0, 960.0 / max(width, height))
        small = cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        targets = []
        dimensions = set()
        for scale in scales:
            if cancel():
                return ()
            color, mask = self._scaled(rgba, float(scale) * ratio)
            th, tw = mask.shape
            if (tw, th) in dimensions or th > gray.shape[0] or tw > gray.shape[1]:
                continue
            dimensions.add((tw, th))
            if np.count_nonzero(mask) < 12:
                continue
            response = cv2.matchTemplate(
                gray, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED, mask=mask
            )
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            for _ in range(max_peaks):
                _, correlation, _, (x, y) = cv2.minMaxLoc(response)
                if correlation < threshold - 0.24:
                    break
                patch = small[y:y+th, x:x+tw]
                difference = float(np.abs(
                    patch[mask > 0].astype(np.float32) - color[mask > 0].astype(np.float32)
                ).mean())
                score = 0.8 * correlation + 0.2 * max(0.0, 1.0-difference/100.0)
                if score >= threshold - 0.18:
                    targets.append(VisualTarget(
                        round(x/ratio), round(y/ratio), round(tw/ratio), round(th/ratio), score
                    ))
                rx, ry = max(3, tw//2), max(3, th//2)
                response[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1] = -1
        targets.sort(key=lambda item: item.score, reverse=True)
        distinct = []
        for target in targets:
            if not any(
                abs(target.center[0]-prior.center[0]) < min(target.width, prior.width)*0.5
                and abs(target.center[1]-prior.center[1]) < min(target.height, prior.height)*0.5
                for prior in distinct
            ):
                distinct.append(target)
        # Downsampled white lettering is phase-sensitive. Validate local peaks
        # against original pixels before accepting a control or item candidate.
        refined = []
        for target in distinct[:20]:
            if cancel():
                return ()
            best = target
            base_scale = target.width / rgba.shape[1]
            for adjustment in np.linspace(0.95, 1.05, 5):
                color, mask = self._scaled(rgba, base_scale*float(adjustment))
                th, tw = mask.shape
                cx, cy = target.center
                left, top = max(0, cx-tw//2-4), max(0, cy-th//2-4)
                right, bottom = min(width, cx+(tw+1)//2+4), min(height, cy+(th+1)//2+4)
                local = frame[top:bottom, left:right]
                if local.shape[0] < th or local.shape[1] < tw:
                    continue
                response = cv2.matchTemplate(
                    cv2.cvtColor(local, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED, mask=mask,
                )
                response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
                _, correlation, _, (x, y) = cv2.minMaxLoc(response)
                patch = local[y:y+th, x:x+tw]
                difference = float(np.abs(
                    patch[mask > 0].astype(np.float32)-color[mask > 0].astype(np.float32)
                ).mean())
                score = 0.8*correlation + 0.2*max(0.0, 1.0-difference/100.0)
                if score > best.score:
                    best = VisualTarget(left+x, top+y, tw, th, score)
            if best.score >= threshold:
                refined.append(best)
        return tuple(sorted(refined, key=lambda item: item.score, reverse=True))

    @staticmethod
    def _cream(image: np.ndarray) -> np.ndarray:
        b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
        return ((b > 155) & (g > 190) & (r > 210) & (r-g < 35) & (g-b < 78))

    @staticmethod
    def _row_missing(image: np.ndarray) -> bool | None:
        b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
        red = (r > 190) & (g < 145) & (b < 120) & (r-g > 65)
        brown = (r > 65) & (r < 185) & (g > 40) & (g < 145) & (b < 115) & (r-g > 15) & (g-b > 10)
        red_count, brown_count = int(red.sum()), int(brown.sum())
        minimum = max(6, image.shape[0] * image.shape[1] * 0.012)
        if red_count >= minimum:
            return True
        if brown_count >= minimum:
            return False
        return None

    @staticmethod
    def _title_crop(image: np.ndarray) -> np.ndarray:
        """Separate the text header from the barn icon and stock bubble beside it."""
        ink = np.max(image, axis=2) < 70
        columns = np.flatnonzero(ink.sum(axis=0) > 1)
        if len(columns) < 8:
            return image
        breaks = np.flatnonzero(np.diff(columns) > max(8, image.shape[0]//4))+1
        groups = np.split(columns, breaks)
        group = max(groups, key=lambda values: values[-1]-values[0])
        left, right = int(group[0]), int(group[-1])+1
        ys = np.flatnonzero(ink[:, left:right].any(axis=1))
        if not len(ys):
            return image
        return image[max(0, int(ys[0])-2):int(ys[-1])+3, max(0, left-2):right+2]

    @staticmethod
    def titles_match(first_png: bytes, second_png: bytes) -> bool:
        """Compare captured item-name artwork without OCR or stock quantities."""
        first, second = _decode(first_png), _decode(second_png)
        aspect_first, aspect_second = first.shape[1]/first.shape[0], second.shape[1]/second.shape[0]
        if abs(aspect_first/aspect_second-1) > 0.12:
            return False
        first = cv2.resize(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (256, 48)).astype(np.float32)
        second = cv2.resize(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), (256, 48)).astype(np.float32)
        first, second = first-first.mean(), second-second.mean()
        denominator = float(np.linalg.norm(first)*np.linalg.norm(second))
        return denominator > 1e-5 and float(np.sum(first*second)/denominator) >= 0.83

    @staticmethod
    def _popup_palette_gain(frame: np.ndarray, arrows: tuple[VisualTarget, ...]) -> float:
        """Calibrate UI brightness from the white interiors of matched arrows.

        The original captures have white at 239; current BlueStacks captures
        also render the same controls at 255. Keep the narrow cream palette and
        original crop pixels, adjusting only its expected brightness. A generic
        pale-color mask would merge the popover with the order panel behind it.
        """
        histogram = np.zeros(256, dtype=np.int64)
        for arrow in arrows:
            x, y, width, height = arrow.box
            patch = frame[y:y+height, x:x+width].astype(np.int16)
            neutral = (patch.min(axis=2) >= 205) & (np.ptp(patch, axis=2) <= 3)
            histogram += np.bincount(patch.max(axis=2)[neutral], minlength=256)
        white = int(histogram.argmax())
        minimum = max(12, round(20 * (frame.shape[0] / 1080) ** 2))
        if histogram[white] < minimum or not 215 <= white <= 255:
            return 1.0
        return white / 239.0

    def _popups(self, frame: np.ndarray, arrows: tuple[VisualTarget, ...]) -> tuple[ResourcePopup, ...]:
        height, width = frame.shape[:2]
        scale = height / 1080
        gain = self._popup_palette_gain(frame, arrows)
        # The resource popover uses a flat cream distinct from most adjacent
        # panels. A broad "pale color" mask joins it to unrelated UI controls.
        mask = np.all(
            np.abs(frame.astype(np.float32)-np.array([205, 233, 239])*gain) < 8*gain, axis=2
        ).astype(np.uint8)
        kernel = np.ones((max(1, round(3*scale)),)*2, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        popups = []
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            box = x, y, w, h
            navigation = tuple(arrow for arrow in arrows if _contains(box, arrow.center))
            if not navigation or not 145*scale < h < height*0.85:
                continue
            # A stock bubble can bridge the popover's cream into the order
            # panel behind it (observed for White Sugar). Its lateral extension
            # is shallow; the actual popup retains tall cream side borders.
            component = labels[y:y+h, x:x+w] == label
            borders = np.flatnonzero(component.sum(axis=0) >= h*.65)
            if len(borders) >= 2:
                padding = max(1, round(2*scale))
                left = max(0, int(borders[0])-padding)
                right = min(w, int(borders[-1])+padding+1)
                trimmed = x+left, y, right-left, h
                kept = tuple(arrow for arrow in navigation if _contains(trimmed, arrow.center))
                if right-left < w*.85 and kept:
                    area = int(component[:, left:right].sum())
                    x, y, w, h = box = trimmed
                    navigation = kept
            if not 130*scale < w < width*0.43:
                continue
            if area < w*h*0.28:
                continue
            popup = frame[y:y+h, x:x+w]
            title_height = min(h//3, max(16, round(75*scale)))
            title = popup[max(0, round(2*scale)):title_height, :]
            # Require actual black-outlined header artwork, not a bare pale shape.
            dark = np.max(title, axis=2) < 70
            if np.count_nonzero(dark) < max(20, title.size//100):
                continue
            title = self._title_crop(title)
            row_background = np.all(
                np.abs(popup.astype(np.float32)-np.array([178, 224, 239])*gain) < 18*gain,
                axis=2,
            )
            row_mask = cv2.morphologyEx(row_background.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            row_count, _, row_stats, _ = cv2.connectedComponentsWithStats(row_mask)
            rows = []
            plausible = [stats for stats in row_stats[1:] if (
                stats[2] > w*0.5 and 30*scale < stats[3] < 125*scale
            )]
            row_width = max((int(stats[2]) for stats in plausible), default=0)
            row_navigation = []
            for candidate in plausible:
                rx, ry, rw, rh, row_area = (int(value) for value in candidate)
                if row_area < rw*rh*0.35:
                    continue
                # An icon can cover the left edge of its row's background. All
                # observed rows share a right edge and the widest detected row
                # supplies the missing left extent, without fixed coordinates.
                rx, rw = max(0, rx+rw-row_width), row_width
                row_box = x+rx, y+ry, rw, rh
                row_arrows = [arrow for arrow in navigation if _contains(row_box, arrow.center)]
                if len(row_arrows) != 1:
                    continue
                row_navigation.append(row_arrows[0])
                # Recipe rows start with a substantial item icon. Text-only
                # "Produce in …" location buttons do not qualify as recipes.
                item_x = max(x, x+rx-round(8*scale))
                item_y = max(y, y+ry-round(5*scale))
                item_width = min(round(rw*0.33), x+w-item_x)
                item_height = min(rh+round(10*scale), y+h-item_y)
                item = frame[item_y:item_y+item_height, item_x:item_x+item_width]
                icon_pixels = ~self._cream(item)
                # Compact ingredients such as cotton occupy slightly less than
                # one fifth of this deliberately generous crop. Location text
                # remains well below this threshold (about 9.8 percent in the
                # original Produce-in prompt replay).
                if float(icon_pixels.mean()) < 0.15:
                    continue
                # Two lines of location text can cross the occupancy threshold
                # as brightness changes. An ingredient also needs a substantial
                # connected illustration; individual text glyphs are much smaller.
                _, _, parts, _ = cv2.connectedComponentsWithStats(icon_pixels.astype(np.uint8))
                if not any(
                    pw >= item_width*0.3 and ph >= item_height*0.3
                    and area >= icon_pixels.size*0.06
                    for _, _, pw, ph, area in parts[1:]
                ):
                    continue
                quantity = (
                    x+rx+round(rw*0.32), y+ry+round(rh*0.20),
                    round(rw*0.34), round(rh*0.66),
                )
                qx, qy, qw, qh = quantity
                missing = self._row_missing(frame[qy:qy+qh, qx:qx+qw])
                rows.append(RecipeRow(
                    row_box, (item_x, item_y, item_width, item_height), quantity,
                    missing, _png(item), row_arrows[0],
                ))
            rows.sort(key=lambda row: row.box[1])
            # Count the broad recipe-row backgrounds, not every arrow inside
            # the connected cream component. A popup can overlap the machine's
            # LEVEL panel and absorb that unrelated arrow into its bounding box.
            complete = bool(rows) and len(rows) == len(plausible) and all(
                row.missing is not None for row in rows
            )
            # Likewise, a location prompt's arrow must belong to its one broad
            # action row. An unrelated arrow merely enclosed by an inflated
            # popup component cannot become a navigation action.
            location = tuple(row_navigation) if (
                not rows and len(plausible) == 1 and len(row_navigation) == 1
            ) else ()
            popups.append(ResourcePopup(box, _png(title), tuple(rows), location, complete))
        popups.sort(key=lambda popup: popup.box[2]*popup.box[3])
        return tuple(popups)

    def observe(self, png: bytes, cancel: Callable[[], bool] | None = None) -> ResourceScene:
        cancelled = cancel or (lambda: False)
        if cancelled():
            return ResourceScene((), (), ())
        frame = _decode(png)
        screen_scale = frame.shape[0] / 1080
        scales = np.linspace(0.8, 1.2, 9) * screen_scale
        arrows = self._search(frame, self._arrow, scales, 0.82, cancelled)
        if cancelled():
            return ResourceScene((), (), ())
        popups = self._popups(frame, arrows)
        empty = self._search(frame, self._empty, scales, 0.84, cancelled)
        # A popup can cover queue controls; its contents are never drop targets.
        empty = tuple(target for target in empty if not any(
            _contains(popup.box, target.center) for popup in popups
        ))
        return ResourceScene(popups, empty, arrows)

    def find_item(
        self, png: bytes, icon_png: bytes, *, min_scale: float = 0.3,
        max_scale: float = 3.0, region: Box | None = None,
        cancel: Callable[[], bool] | None = None, threshold: float = .84,
    ) -> tuple[VisualTarget, ...]:
        """Find visual candidates; matching a world decoration is not collection.

        Accepts the cropped native item image from OrderReader.item_icon. Cream
        background connected to the crop boundary is excluded automatically; pale
        pixels enclosed inside an icon are retained. Caller verifies resulting game
        state, and must not use a visual candidate as a claim that goods are ready.
        """
        cancelled = cancel or (lambda: False)
        if cancelled():
            return ()
        if not (0 < min_scale <= max_scale <= 8):
            raise ValueError("Item search scales must satisfy 0 < min <= max <= 8.")
        if not .5 <= threshold <= 1:
            raise ValueError('Item match threshold must be between .5 and 1.')
        frame, icon = _decode(png), _decode(icon_png, True)
        if icon.ndim != 3 or icon.shape[2] not in (3, 4):
            raise ValueError("Item icon must be RGB or RGBA.")
        if icon.shape[2] == 3:
            cream = self._cream(icon).astype(np.uint8)
            _, labels = cv2.connectedComponents(cream)
            edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            background = np.isin(labels, edge_labels[edge_labels != 0])
            alpha = (~background).astype(np.uint8)*255
            # Ignore the antialiased border between UI paper and item artwork.
            alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
            icon = np.dstack((icon, alpha))
        ys, xs = np.nonzero(icon[:, :, 3] >= 200)
        if len(xs) < 30:
            return ()
        icon = icon[ys.min():ys.max()+1, xs.min():xs.max()+1]
        origin_x = origin_y = 0
        if region is not None:
            origin_x, origin_y, rw, rh = region
            if origin_x < 0 or origin_y < 0 or rw <= 0 or rh <= 0 or (
                origin_x+rw > frame.shape[1] or origin_y+rh > frame.shape[0]
            ):
                raise ValueError("Item search region must fit inside the screenshot.")
            frame = frame[origin_y:origin_y+rh, origin_x:origin_x+rw]
        count = max(1, int(np.ceil(np.log(max_scale/min_scale)/np.log(1.1)))+1)
        found = self._search(frame, icon, np.geomspace(min_scale, max_scale, count), threshold, cancelled, 4)
        return tuple(VisualTarget(
            target.x+origin_x, target.y+origin_y, target.width, target.height, target.score
        ) for target in found)

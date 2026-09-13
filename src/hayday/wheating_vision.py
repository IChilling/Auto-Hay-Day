"""Read wheat fields and the player's shop; all input lives in the workers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.ad_text import AdTextReader
from hayday.farming import FarmingWorker
from hayday.game_scene import farm_scene_vision
from hayday.quantities import read_count
from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.wheating_numbers import read_number
from hayday.wheating_shop_building import ShopBuildingVision
from hayday.wheating_sold import SoldReceiptVision


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


@dataclass(frozen=True)
class SaleSlot:
    kind: str
    target: VisualTarget
    advertised: bool = False


@dataclass(frozen=True)
class ShopView:
    kind: str = 'unknown'
    close: VisualTarget | None = None
    slots: tuple[SaleSlot, ...] = ()
    wheat: VisualTarget | None = None
    stock: int | None = None
    quantity: int | None = None
    price: int | None = None
    plus_quantity: VisualTarget | None = None
    minus_quantity: VisualTarget | None = None
    plus_price: VisualTarget | None = None
    max_price: VisualTarget | None = None
    submit: VisualTarget | None = None
    newspaper: VisualTarget | None = None
    ad_free: bool = False
    ad_checked: bool = False
    cooldown: int | None = None
    silo_tab: VisualTarget | None = None
    inventory: VisualTarget | None = None
    layout_verified: bool = False


class WheatingVision:
    def __init__(self, cancel=lambda: False, text_reader=None):
        self.cancel = cancel
        self.root = Path(__file__).parent/'assets/wheating'
        self.specs = json.loads((self.root/'manifest.json').read_text('utf-8'))['features']
        self.refs = {path.stem: _decode(path.read_bytes(), True) for path in self.root.glob('*.png')}
        # Cropping tools may save the title as RGB. The matcher requires an
        # alpha mask; include every pixel when the title has no transparency.
        header = self.refs.get('shop_header')
        if header is not None and header.shape[2] == 3:
            self.refs['shop_header'] = cv2.cvtColor(header, cv2.COLOR_BGR2BGRA)
        if not {'shop', 'wheat_seed', 'soil', 'wheat_ripe'} <= self.refs.keys():
            raise ValueError('Wheating reference images are missing.')
        self.wheat_icon = (self.root/'wheat_seed.png').read_bytes()
        self.matcher = ResourceVision()
        self.text = text_reader or AdTextReader(cancel=cancel)
        self._png = None
        self._cache = {}
        self._image = None
        self._composer_anchor = None
        self._composer_layouts = {}
        self._composer_candidates = {}
        self._inventory_wheat = None
        self._edit_layout = None
        self._edit_status = None
        self._overview_layout = None
        self._overview_controls = []
        self._slot_cache = []
        self._slot_features = {}
        self._sold_receipts = SoldReceiptVision(cancel)
        self._shop_building = ShopBuildingVision(cancel)
        self._crop_controls = None

    def _image_for(self, png):
        if self._png != png:
            self._png, self._cache = png, {}
            self._image = _decode(png)
        return self._image

    def matches(self, png, name, threshold=.91, world=False, region=None):
        decoded = self._image_for(png)
        key = name, threshold, world, region
        if key not in self._cache:
            image = decoded
            base = image.shape[0]/(1080 if name in {'soil', 'wheat_ripe', 'wheat_seed'} else 1045)
            scales = np.unique(np.r_[np.geomspace(base*(.35 if world else .80), base*1.35, 19), base, 1.])
            x = y = 0
            if region:
                x, y, w, h = region
                image = image[y:y+h, x:x+w]
            # Try native UI scale first. Mixing near-identical coarse scales can
            # suppress the exact peak for thin outlined lettering.
            native = 1080 if name in {'soil', 'wheat_ripe', 'wheat_seed'} else 1041 if name in {'shop_header', 'shop_close', 'empty_sale', 'sold'} else 1045
            native = self.specs.get(name, {}).get('reference_height', native)
            exact = np.array([decoded.shape[0]/native])
            peaks = 24 if name in {'empty_sale', 'sold', 'sold_live', 'wheat_sale'} else 8
            hits = self.matcher._search(image, self.refs[name], exact, threshold, self.cancel, peaks)
            if not hits:
                hits = self.matcher._search(image, self.refs[name], scales, threshold, self.cancel, peaks)
            self._cache[key] = tuple(VisualTarget(t.x+x, t.y+y, t.width, t.height, t.score) for t in hits)
        return self._cache[key]

    def one(self, png, name, **kwargs):
        hits = self.matches(png, name, **kwargs)
        return hits[0] if hits and (len(hits) == 1 or hits[0].score-hits[1].score > .045) else None

    def farm(self, frame):
        self._image_for(frame.png)
        if 'farm' not in self._cache:
            self._cache['farm'] = farm_scene_vision().ready(frame.png)
        return self._cache['farm']

    def shop_building(self, frame):
        image = self._image_for(frame.png)
        if 'shop_building' not in self._cache:
            self._cache['shop_building'] = self._shop_building.find(image)
        return self._cache['shop_building']

    def crop_controls(self, frame):
        """Recognize a crop overlay before dismissing it on the way to the shop."""
        self._image_for(frame.png)
        if 'crop_controls' not in self._cache:
            if self._crop_controls is None:
                from hayday.wheating_crop import WheatFarmingVision
                self._crop_controls = WheatFarmingVision(cancel=self.cancel)
            vision = self._crop_controls
            harvest = vision.harvest(frame.png)
            seed = None if harvest else vision.page_next(frame.png)
            growing = None if harvest or seed else vision.growing(frame.png)
            self._cache['crop_controls'] = (('harvest', harvest.tool) if harvest else
                ('seed', seed) if seed else ('growing', growing) if growing else None)
        return self._cache['crop_controls']

    def shop_is_open(self, frame):
        """Verify the overview controls without decoding inventory or every sale slot."""
        header_region = (round(frame.width*.25), 0, round(frame.width*.50), round(frame.height*.21))
        header = self.one(frame.png, 'shop_header', region=header_region)
        if header is None:
            return False
        close_region = (round(frame.width*.75), 0, round(frame.width*.20), round(frame.height*.23))
        close = self.one(frame.png, 'shop_close', region=close_region)
        return close is not None and close.x > header.x+header.width

    def shop_layout_current(self, frame):
        """The overview's fixed controls still occupy their settled positions."""
        return (self._overview_layout is not None
                and self._overview_layout[2] == (frame.width, frame.height)
                and self._pixels_match(self._image_for(frame.png), self._overview_controls))

    def dialog_close(self, frame):
        if self.farm(frame):
            return None
        target = self.one(frame.png, 'shop_close', threshold=.94)
        if (target and frame.width*.5 < target.center[0] < frame.width*.94
                and frame.height*.02 < target.center[1] < frame.height*.30):
            return target
        return None

    def plots(self, frame, kind):
        self._image_for(frame.png)
        key = 'plots', kind
        if key not in self._cache:
            self._cache[key] = self._plots(frame, kind)
        return self._cache[key]

    def _plots(self, frame, kind):
        """Candidates only. Selected farming controls authorize the actual gesture."""
        region = (round(frame.width*.10), round(frame.height*.20),
                  round(frame.width*.79), round(frame.height*.62))
        if kind == 'ripe':
            textures = self._ripe_textures(frame)
            if textures:
                return self._ripe_bases(frame, textures)
            if not self._wheat_regions(frame):
                return ()
        found = self.matches(frame.png, 'soil_texture' if kind == 'empty' else 'ripe_live',
                             threshold=.90, world=True, region=region)
        return found or self.matches(frame.png, 'soil' if kind == 'empty' else 'wheat_ripe',
                                    threshold=.91, world=True, region=region)

    def _ripe_textures(self, frame):
        self._image_for(frame.png)
        if 'ripe_textures' not in self._cache:
            self._cache['ripe_textures'] = self._find_ripe_textures(frame)
        return self._cache['ripe_textures']

    def _wheat_regions(self, frame):
        _, stats = self._wheat_data(frame)
        base = frame.height/1080
        parts = [s for s in stats[1:] if s[4] >= 400*base**2 and s[4]/(s[2]*s[3]) >= .25]
        if not parts:
            return ()
        largest = max(s[4] for s in parts)
        if largest >= 15000*base**2:
            # Follow the connected field, keeping nearby crop clumps but leaving
            # small yellow farm decorations out of the expensive texture search.
            nearby = [max(parts, key=lambda s: s[4])]
            remaining = [s for s in parts if s is not nearby[0] and s[4] >= 900*base**2]
            while remaining:
                joined = [s for s in remaining if any(np.hypot(
                    max(0, s[0]-p[0]-p[2], p[0]-s[0]-s[2]),
                    max(0, s[1]-p[1]-p[3], p[1]-s[1]-s[3])) < 70*base for p in nearby)]
                if not joined:
                    break
                nearby.extend(joined)
                remaining = [s for s in remaining if all(s is not p for p in joined)]
            parts = nearby
        pad = round(55*base)
        # Keep the existing farm-only search boundary. Padding into the upper
        # HUD/picker region can mistake enlarged seed artwork for wheat texture.
        left = max(round(frame.width*.10), min(s[0] for s in parts)-pad)
        top = max(round(frame.height*.20), min(s[1] for s in parts)-pad)
        right = min(round(frame.width*.89), max(s[0]+s[2] for s in parts)+pad)
        bottom = min(round(frame.height*.82), max(s[1]+s[3] for s in parts)+pad)
        if right <= left or bottom <= top:
            return ()
        return (int(left), int(top), int(right-left), int(bottom-top))

    def _find_ripe_textures(self, frame, *, broad=False):
        names = ('ripe_texture', 'ripe_texture_dense', 'ripe_full_a', 'ripe_full_b')
        region = self._wheat_regions(frame)
        if not region:
            return ()
        # Color only narrows the search. Original wheat textures still establish
        # identity; several weaker hits require the same dense field near the shop.
        x, y, w, h = region
        crop = self._image_for(frame.png)[y:y+h, x:x+w]
        if broad:
            native = tuple(t for name in names for t in self.matches(
                frame.png, name, threshold=.82, world=True, region=region))
        else:
            native = tuple(VisualTarget(t.x+x, t.y+y, t.width, t.height, t.score)
                           for name in names for t in self.matcher._search(crop, self.refs[name],
                               np.array([frame.height/1080]), .82, self.cancel, 8))
        textures = tuple(t for t in native if t.score >= .90)
        if textures:
            return textures
        shop = self.shop_building(frame)
        # Swaying wheat at an intermediate zoom can miss the strict template
        # threshold. Several separate texture matches inside one dense field
        # beside the shop provide a candidate; selected crop controls still
        # gate harvesting. A yellow patch or a single weak match is insufficient.
        textures = native
        labels, stats, identified = self._wheat_components(frame, textures)
        accepted = []
        for component in identified:
            if shop is None:
                break
            x, y, w, h, area = stats[component]
            gap = np.hypot(max(0, x-shop.x-shop.width, shop.x-x-w),
                           max(0, y-shop.y-shop.height, shop.y-y-h))
            if area < 1500*(frame.height/1080)**2 or area/(w*h) < .35 or gap > frame.width*.20:
                continue
            separate = []
            hits = []
            for target in textures:
                cx, cy = target.center
                if labels[cy, cx] != component:
                    continue
                hits.append(target)
                if all(np.linalg.norm(np.subtract(target.center, p)) > 24*frame.height/1080 for p in separate):
                    separate.append(target.center)
            if len(separate) >= 3:
                accepted.extend(hits)
        if broad and not accepted and max(self._wheat_data(frame)[1][1:, 4], default=0) >= 15000*(frame.height/1080)**2:
            # Cropping changes the coarse sampling phase of very small texture
            # references. Retain the original strict search for that rare case.
            original = (round(frame.width*.10), round(frame.height*.20),
                        round(frame.width*.79), round(frame.height*.62))
            return tuple(t for name in names for t in self.matches(
                frame.png, name, threshold=.90, world=True, region=original))
        if accepted or broad:
            return tuple(accepted)
        return self._find_ripe_textures(frame, broad=True)

    def _wheat_data(self, frame):
        image = self._image_for(frame.png)
        if 'wheat_data' in self._cache:
            return self._cache['wheat_data']
        # Texture establishes wheat identity. Connected foliage locates its
        # actual lower edge, so taps do not land on the soil behind stalk tips.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 155, 180), (31, 255, 255))
        # Gold HUD frames can touch low wheat and merge into its component.
        # Remove fixed screen controls before following connected stalks.
        mask[round(frame.height*.84):] = 0
        mask[:round(frame.height*.16)] = 0
        mask[:, :round(frame.width*.10)] = 0
        mask[:, round(frame.width*.88):] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        self._cache['wheat_data'] = labels, stats
        return labels, stats

    def _wheat_components(self, frame, textures):
        labels, stats = self._wheat_data(frame)
        identified = {}
        for texture in textures:
            values = labels[texture.y:texture.y+texture.height, texture.x:texture.x+texture.width]
            ids, counts = np.unique(values[values > 0], return_counts=True)
            if len(ids):
                label = int(ids[np.argmax(counts)])
                if stats[label, 4] > 400*(frame.height/1080)**2:
                    identified[label] = max(texture.score, identified.get(label, 0))
        return labels, stats, identified

    def _ripe_bases(self, frame, textures):
        labels, stats, identified = self._wheat_components(frame, textures)
        targets = []
        for label, score in identified.items():
            x, y, w, h, _ = stats[label]
            if y+h < frame.height*.24:
                continue
            component = (labels == label).astype(np.uint8)
            distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
            _, radius, _, (cx, cy) = cv2.minMaxLoc(distance)
            if radius < 4*frame.height/1080:
                continue
            size = round(30*frame.height/1080)
            targets.append(VisualTarget(cx-size//2, cy-size//2, size, size, score))
        return tuple(targets)

    def field_bounds(self, frame):
        textures = self._ripe_textures(frame)
        _, stats, identified = self._wheat_components(frame, textures)
        parts = [stats[label] for label in identified if stats[label, 4] > 1500*(frame.height/1080)**2]
        if not parts:
            return None
        left, top = min(s[0] for s in parts), min(s[1] for s in parts)
        right, bottom = max(s[0]+s[2] for s in parts), max(s[1]+s[3] for s in parts)
        return VisualTarget(int(left), int(top), int(right-left), int(bottom-top), 1.)

    def harvest_sweep(self, frame):
        self._image_for(frame.png)
        if 'harvest_sweep' not in self._cache:
            self._cache['harvest_sweep'] = self._harvest_sweep(frame)
        return list(self._cache['harvest_sweep'])

    def wheat_outside(self, frame, points):
        """Check live, texture-confirmed foliage beyond a cached plot route."""
        textures = self._ripe_textures(frame)
        labels, stats, identified = self._wheat_components(frame, textures)
        if not identified:
            return False
        coverage = np.zeros(labels.shape, np.uint8)
        scale = frame.height/1080
        for point in points:
            cv2.ellipse(coverage, tuple(map(int, point)), (round(50*scale), round(65*scale)),
                        0, 0, 360, 255, -1)
        # Ignore isolated edge pixels caused by stalk sway. This tests coverage,
        # not maturity, and never adds unverified color blobs to the plot map.
        outside = (np.isin(labels, tuple(identified)) & (coverage == 0)).astype(np.uint8)
        _, _, parts, _ = cv2.connectedComponentsWithStats(outside)
        # Gold order-board trim can touch the field's color component. Its
        # sparse outline must not force a complete retrace on every cycle.
        return any(area > 300*scale**2 and area/(w*h) > .40
                   for x, y, w, h, area in parts[1:])

    def _harvest_sweep(self, frame):
        """One serpentine path across wheat in the observed field."""
        bases = self.plots(frame, 'ripe')
        if not bases:
            return []
        labels, stats = self._wheat_data(frame)
        selected = set()
        for base in bases:
            x, y = base.center
            label = int(labels[y, x])
            if label:
                selected.add(label)
        if not selected:
            return []
        # Include nearby wheat clumps belonging to the same field even when a
        # swaying clump lacks a template peak in this individual frame.
        changed = True
        while changed:
            changed = False
            for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
                if (label in selected or area < 900*(frame.height/1080)**2 or w/h < .65
                        or x < frame.width*.12 or x+w > frame.width*.88
                        or y < frame.height*.2 or y+h > frame.height*.90):
                    continue
                for known in tuple(selected):
                    sx, sy, sw, sh, _ = stats[known]
                    gap_x = max(0, x-sx-sw, sx-x-w)
                    gap_y = max(0, y-sy-sh, sy-y-h)
                    if np.hypot(gap_x, gap_y) < 70*frame.height/1080:
                        selected.add(label)
                        changed = True
                        break
        field = np.isin(labels, tuple(selected))
        ys, _ = np.nonzero(field)
        if not len(ys):
            return []
        step = max(10, round(24*frame.height/1080))
        rows = list(range(int(ys.min()+step//2), int(ys.max()), step))
        if len(rows)*2+2 > 98:
            return []
        path = []
        for row, y in enumerate(rows):
            xs = np.nonzero(field[y])[0]
            if len(xs) < 8:
                continue
            endpoints = [(int(xs[2]), y), (int(xs[-3]), y)]
            path.extend(reversed(endpoints) if row % 2 else endpoints)
        return path

    def wheat_at(self, frame, target):
        """Check ripe wheat texture locally; yellow color alone is insufficient."""
        return any(np.linalg.norm(np.subtract(t.center, target.center)) < max(t.width, target.width)
                   for t in self.plots(frame, 'ripe'))

    @staticmethod
    def _line_target(line):
        return VisualTarget(*line.bounds, 1.) if line.bounds else None

    @staticmethod
    def _enabled(image, target):
        if target is None:
            return False
        x, y, w, h = target.box
        pad = max(2, h//2)
        patch = image[max(0, y-pad):min(len(image), y+h+pad), x:x+w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return float(np.mean((hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 120))) > .25

    @staticmethod
    def _checked(image, newspaper):
        if newspaper is None:
            return False
        x, y, w, h = newspaper.box
        patch = image[max(0, y-h//3):y+h, x:min(image.shape[1], x+2*w)]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return np.mean(cv2.inRange(hsv, (35, 100, 90), (85, 255, 255)) > 0) > .07

    @staticmethod
    def _plus_buttons(image, region):
        """White plus shapes inside round controls; never the premium bottom row."""
        x0, y0, w, h = region
        crop = image[y0:y0+h, x0:x0+w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 200), (179, 65, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(white)
        result = []
        for x, y, bw, bh, area in stats[1:]:
            if not (12*image.shape[0]/1080 <= min(bw, bh) and .65 < bw/bh < 1.45 and .3 < area/(bw*bh) < .8):
                continue
            shape = white[y:y+bh, x:x+bw] > 0
            if (shape[bh//3:2*bh//3].mean() > .68
                    and shape[:, bw//3:2*bw//3].mean() > .68
                    and shape[:bh//4, :bw//4].mean() < .2):
                result.append(VisualTarget(int(x+x0), int(y+y0), int(bw), int(bh), 1.))
        return tuple(sorted(result, key=lambda p: p.y))

    def shop(self, frame):
        image = self._image_for(frame.png)
        cached = self._cache.get('shop_view')
        if cached is not None and cached[0] is frame:
            return cached[1]
        view = self._shop(frame, image)
        # Recovery and the shop worker inspect the same capture. Parse it once.
        self._cache['shop_view'] = frame, view
        return view

    def _shop(self, frame, image):
        png = frame.png
        if not self._shop_close_possible(image):
            return ShopView()
        fast = self._overview_fast(frame, image)
        if fast is not None:
            return fast
        if (self._edit_layout is not None and self._edit_layout['size'] == (frame.width, frame.height)
                and self._pixels_match(image, self._edit_layout['guards'])):
            return self._edit_values(frame, image, self._edit_layout, verified=True)
        for layout in (*self._composer_layouts.values(), *self._composer_candidates.values()):
            if layout['size'] == (frame.width, frame.height) and self._pixels_match(image, layout['guards']):
                self._composer_layouts[layout['selected']] = layout
                return self._composer_values(frame, image, layout, verified=True)
        header_region = (round(frame.width*.25), 0, round(frame.width*.50), round(frame.height*.21))
        close_region = (round(frame.width*.75), 0, round(frame.width*.20), round(frame.height*.23))
        # Opening the building has already verified these features on this
        # exact frame. Consume that proof before searching for absent dialogs.
        headers = self._cache.get(('shop_header', .91, False, header_region), ())
        closes = self._cache.get(('shop_close', .91, False, close_region), ())
        if len(headers) == len(closes) == 1 and closes[0].x > headers[0].x+headers[0].width:
            header, close = headers[0], closes[0]
            return self._overview_observed(frame, image, header, close)
        if self._composer_anchor is not None:
            a = self._composer_anchor
            region = (max(0, a.x-8), max(0, a.y-8), min(frame.width-a.x+8, a.width+16), min(frame.height-a.y+8, a.height+16))
            adjust = self.one(png, 'composer_adjust', region=region)
            if adjust:
                self._composer_anchor = adjust
                return self._composer_observed(frame, image, adjust)
        adjust = self.one(png, 'composer_adjust', region=(round(frame.width*.48), round(frame.height*.20),
                          round(frame.width*.47), round(frame.height*.40)))
        if adjust is not None:
            self._composer_anchor = adjust
            return self._composer_observed(frame, image, adjust)
        edit = self.one(png, 'edit_title', region=(round(frame.width*.28), round(frame.height*.05),
                        round(frame.width*.45), round(frame.height*.25)))
        if edit is not None:
            return self._edit(frame, image, edit)
        header = self.one(png, 'shop_header', region=header_region)
        close = self.one(png, 'shop_close', region=close_region) if header is not None else None
        if header is not None and close is not None:
            return self._overview_observed(frame, image, header, close)
        return ShopView()

    def _overview_observed(self, frame, image, header, close):
        png = frame.png
        scale = header.width/self.refs['shop_header'].shape[1]
        slots, features = [], {}
        region = (round(frame.width*.10), round(frame.height*.23), round(frame.width*.80), round(frame.height*.56))
        for name, kind in (('empty_sale', 'empty'), ('sold', 'sold'), ('sold_live', 'sold')):
            for match in self.matches(png, name):
                # Top/bottom rows are discovered from artwork, not slot indexes.
                if match.y <= header.y+header.height or match.y > frame.height*.82:
                    continue
                cx, cy = match.center
                if kind == 'sold':
                    cy += round(100*scale)
                candidate = SaleSlot(kind, VisualTarget(cx-round(115*scale), cy-round(120*scale),
                                                        round(230*scale), round(240*scale), match.score))
                if not any(np.linalg.norm(np.subtract(s.target.center, candidate.target.center)) < 80*scale for s in slots):
                    slots.append(candidate)
                    features[candidate.target.box] = self._feature_pixels(image, match)
        for match in self._sold_badges(image):
            candidate = self._sold_slot(match, scale)
            if not any(self._same_slot(slot, candidate) for slot in slots):
                slots.append(candidate)
                features[candidate.target.box] = self._feature_pixels(image, match)
        wheat = self.matches(png, 'wheat_sale', region=region, threshold=.90)
        for match in wheat:
            cx, cy = match.center
            target = VisualTarget(cx-round(110*scale), cy-round(100*scale), round(220*scale), round(230*scale), match.score)
            if not any(np.linalg.norm(np.subtract(slot.target.center, target.center)) < 100*scale for slot in slots):
                marker = self._ad_marker(image, self._slot_region(image, target))
                advertised = bool(marker and marker.center[0] < target.x+target.width*.5)
                slots.append(SaleSlot('wheat', target, advertised))
                features[target.box] = self._feature_pixels(image, match)
        slots = tuple(sorted(slots, key=lambda s: (s.target.y, s.target.x)))
        same_layout = (self._overview_layout is not None
                       and self._overview_layout[2] == (frame.width, frame.height)
                       and FarmingWorker._same_target(self._overview_layout[0], header, frame))
        covered = all(any(np.linalg.norm(np.subtract(old.target.center, new.target.center)) < old.target.width*.38
                          for new in slots) for old, _ in self._slot_cache)
        # Coin/receipt animations can temporarily hide a slot's artwork. Return
        # only recognized slots, but retain its location for the next frame.
        if not same_layout or covered:
            self._overview_layout = header, close, (frame.width, frame.height)
            self._overview_controls = [(t, image[t.y:t.y+t.height, t.x:t.x+t.width].copy()) for t in (header, close)]
            self._slot_cache = [(slot, self._slot_crop(image, slot.target).copy()) for slot in slots]
            self._slot_features = features
        return ShopView('overview', close=close, slots=slots)

    @staticmethod
    def _feature_pixels(image, target):
        return target, image[target.y:target.y+target.height, target.x:target.x+target.width].copy()

    def _sold_badges(self, image):
        if 'sold_badges' not in self._cache:
            self._cache['sold_badges'] = self._sold_receipts.find(image)
        return self._cache['sold_badges']

    @staticmethod
    def _sold_slot(match, scale):
        cx, cy = match.center
        return SaleSlot('sold', VisualTarget(cx-round(115*scale), cy-round(20*scale),
                                            round(230*scale), round(240*scale), match.score))

    @staticmethod
    def _same_slot(first, second):
        return (np.linalg.norm(np.subtract(first.target.center, second.target.center))
                < min(first.target.width, second.target.width)*.38)

    @staticmethod
    def _slot_region(image, target):
        left, top = max(0, target.x-45), max(0, target.y-55)
        right = min(image.shape[1], target.x+target.width+45)
        bottom = min(image.shape[0], target.y+target.height+70)
        return left, top, right-left, bottom-top

    @classmethod
    def _slot_crop(cls, image, target):
        x, y, w, h = cls._slot_region(image, target)
        return image[y:y+h, x:x+w]

    def _local(self, image, name, region, threshold=.91):
        x, y, w, h = region
        native = self.specs[name].get('reference_height', 1041 if name in {'shop_header', 'shop_close', 'empty_sale', 'sold'} else 1080)
        scale = image.shape[0]/native
        hits = self.matcher._search(image[y:y+h, x:x+w], self.refs[name],
            np.array([scale]), threshold, self.cancel, 1)
        if not hits:
            hits = self.matcher._search(image[y:y+h, x:x+w], self.refs[name],
                np.array([.95, 1.05, 1.10])*scale, threshold, self.cancel, 1)
        if not hits:
            return None
        t = hits[0]
        return VisualTarget(t.x+x, t.y+y, t.width, t.height, t.score)

    def _overview_fast(self, frame, image):
        if self._overview_layout is None or not self._slot_cache:
            return None
        header, close, size = self._overview_layout
        if size != (frame.width, frame.height):
            return None
        # Both unobscured controls must match freshly. A dimmed shop behind a
        # sale dialog must never qualify for cached-slot input.
        if not self._pixels_match(image, self._overview_controls):
            # Flying coins can cross a control without moving the shop. Recheck
            # its original template locally before paying for full discovery.
            # Both controls still need the normal recognition threshold and
            # must occupy their previously verified positions.
            for name, (target, pixels) in zip(('shop_header', 'shop_close'), self._overview_controls, strict=True):
                if self._pixels_match(image, [(target, pixels)]):
                    continue
                left, top = max(0, target.x-8), max(0, target.y-8)
                right, bottom = min(frame.width, target.x+target.width+8), min(frame.height, target.y+target.height+8)
                matched = self._local(image, name, (left, top, right-left, bottom-top))
                if not FarmingWorker._same_target(target, matched, frame):
                    return None
        slots, updated = [], []
        identities = dict(self._slot_features)
        scale = header.width/self.refs['shop_header'].shape[1]
        badges = [(self._sold_slot(match, scale), match) for match in self._sold_badges(image)]
        for previous, pixels in self._slot_cache:
            if self.cancel():
                return None
            crop = self._slot_crop(image, previous.target)
            if np.array_equal(crop, pixels):
                slots.append(previous)
                updated.append((previous, pixels))
                continue
            region = self._slot_region(image, previous.target)
            feature = self._slot_features.get(previous.target.box)
            kind = None
            badge = next((match for slot, match in badges if self._same_slot(previous, slot)), None)
            if badge is not None:
                kind = 'sold'
                identities[previous.target.box] = self._feature_pixels(image, badge)
            elif feature is not None:
                target, identity_pixels = feature
                current = image[target.y:target.y+target.height, target.x:target.x+target.width]
                # Only exactly identical identity artwork can bypass a search.
                # Coins and moving neighbors outside it do not change the item.
                if np.array_equal(current, identity_pixels):
                    kind = previous.kind
            if kind is None:
                features = [('empty_sale', 'empty'), ('sold_live', 'sold'), ('sold', 'sold'), ('wheat_sale', 'wheat')]
                features.sort(key=lambda item: item[1] != previous.kind)
                for name, candidate in features:
                    matched = self._local(image, name, region)
                    if matched is not None:
                        kind = candidate
                        identities[previous.target.box] = self._feature_pixels(image, matched)
                        break
            if kind is None:
                # Receipt/coin animations can hide one crate. Keep its location,
                # omit it from input targets, and continue recognizing the rest.
                updated.append((previous, pixels))
                continue
            marker = self._ad_marker(image, region) if kind == 'wheat' else None
            advertised = bool(marker and marker.center[0] < previous.target.x+previous.target.width*.5)
            slot = SaleSlot(kind, previous.target, advertised)
            slots.append(slot)
            updated.append((slot, crop.copy()))
        # Receipts can appear in crates omitted by an earlier partial scan.
        # Discover them even when every previously cached crate is unchanged.
        for slot, match in badges:
            if not any(self._same_slot(previous, slot) for previous, _ in updated):
                slots.append(slot)
                updated.append((slot, self._slot_crop(image, slot.target).copy()))
                identities[slot.target.box] = self._feature_pixels(image, match)
        if len(slots) < max(1, len(self._slot_cache)//2):
            # A changed crate grid needs discovery, rather than being treated
            # as one temporary receipt animation in an otherwise stable grid.
            return None
        self._slot_cache = updated
        self._slot_features = identities
        return ShopView('overview', close=close, slots=tuple(slots), layout_verified=True)

    def _shop_close_possible(self, image):
        # A cheap rejection gate for leaving the shop. This never identifies a
        # control or authorizes a tap; actual shop artwork must still match.
        small = cv2.resize(image, (640, round(640*image.shape[0]/image.shape[1])), interpolation=cv2.INTER_AREA)
        crop = small[:round(len(small)*.24), round(640*.45):round(640*.97)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for name, native in (('composer_close', 1080), ('edit_close', 1045)):
            for factor in (.8, .9, 1., 1.1, 1.2):
                color, mask = self.matcher._scaled(self.refs[name], image.shape[0]/native*640/image.shape[1]*factor)
                h, w = mask.shape
                if h > len(gray) or w > gray.shape[1]:
                    continue
                response = cv2.matchTemplate(gray, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED, mask=mask)
                response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                difference = cv2.absdiff(crop[y:y+h, x:x+w], color)[mask > 0].mean()
                if .8*score+.2*max(0., 1.-difference/100) >= .70:
                    return True
        return False

    @staticmethod
    def _pixels_match(image, controls):
        for target, pixels in controls:
            crop = image[target.y:target.y+target.height, target.x:target.x+target.width]
            if crop.shape != pixels.shape:
                return False
            difference = cv2.absdiff(crop, pixels)
            if difference.mean() > 2 or np.mean(difference > 8) > .05:
                return False
        return True

    def _ad_marker(self, image, region):
        marker = self._local(image, 'ad_marker', region)
        if marker:
            return marker
        # The first crate can clip the paper's left edge against the shop rim.
        # Match its unchanged newsprint interior, inside an observed wheat slot.
        x, y, w, h = region
        body = self.refs['ad_marker'][12:64, 22:48]
        hits = self.matcher._search(image[y:y+h, x:x+w], body,
            np.array([.95, 1., 1.05, 1.10])*(image.shape[0]/1080), .94, self.cancel, 1)
        if not hits:
            return None
        target = hits[0]
        return VisualTarget(target.x+x, target.y+y, target.width, target.height, target.score)

    def _edit(self, frame, image, anchor):
        close = self.one(frame.png, 'edit_close', region=self.relative(
            anchor, (1090, 35, 210, 210), source=(800, 164, 1020, 231)).box)
        if close is None:
            return ShopView('edit')
        layout = dict(anchor=anchor, close=close, size=(frame.width, frame.height),
                      guards=[(t, image[t.y:t.y+t.height, t.x:t.x+t.width].copy()) for t in (anchor, close)])
        self._edit_layout = layout
        return self._edit_values(frame, image, layout)

    def _edit_values(self, frame, image, layout, verified=False):
        png = frame.png
        anchor = layout['anchor']

        def region(box):
            return self.relative(anchor, box, source=(800, 164, 1020, 231)).box

        newspaper = self.one(png, 'newspaper', region=region((990, 430, 190, 170)), threshold=.88)
        tick = self.one(png, 'ad_check', region=region((990, 430, 220, 180)), threshold=.90)
        if newspaper is None:
            newspaper = tick
        button = self.one(png, 'advertise_button_live', region=region((675, 650, 490, 135)), threshold=.91)
        wheat = self.matches(png, 'wheat_inventory', region=region((680, 260, 290, 185)))
        free = self.one(png, 'advertise_now', region=region((675, 470, 330, 120))) is not None
        cooldown = None
        if not free:
            x, y, w, h = region((650, 430, 550, 220))
            status = image[y:y+h, x:x+w]
            if self._edit_status is not None and np.array_equal(status, self._edit_status[0]):
                cooldown = self._edit_status[1]
            else:
                # Free-ad artwork is decisive. OCR is needed only for a timer,
                # and only its small status panel is sent to the local reader.
                reading = self.text.read(cv2.imencode('.png', status)[1].tobytes(), cancel=self.cancel)
                lines = reading.lines if not reading.error else ()
                text = normalized(' '.join(line.text for line in lines))
                clock = re.search(r'\b([0-5]):([0-5][0-9])\b', text)
                clock = clock or re.search(r'\b([0-5])\s*min\s*([0-5]?[0-9])\s*sec\b', text)
                cooldown = int(clock[1])*60+int(clock[2]) if clock else None
                self._edit_status = status.copy(), cooldown
        return ShopView('edit', close=layout['close'], wheat=wheat[0] if len(wheat) == 1 else None,
            submit=button if self._enabled(image, button) else None, newspaper=newspaper,
            ad_free=free, ad_checked=bool(tick), cooldown=cooldown, layout_verified=verified)

    @staticmethod
    def relative(anchor, box, source=(1260, 389, 1476, 430)):
        scale = anchor.width/(source[2]-source[0])
        x, y, w, h = box
        return VisualTarget(round(anchor.x+(x-source[0])*scale), round(anchor.y+(y-source[1])*scale),
                            max(1, round(w*scale)), max(1, round(h*scale)), anchor.score)

    def near(self, frame, anchor, name):
        box = self.specs[name]['box']
        predicted = self.relative(anchor, (box[0], box[1], box[2]-box[0], box[3]-box[1]))
        margin = max(5, round(predicted.height*.12))
        left, top = max(0, predicted.x-margin), max(0, predicted.y-margin)
        right, bottom = min(frame.width, predicted.x+predicted.width+margin), min(frame.height, predicted.y+predicted.height+margin)
        if right <= left or bottom <= top:
            return None
        return self.one(frame.png, name, region=(left, top, right-left, bottom-top), threshold=.90)

    def _composer_observed(self, frame, image, anchor):
        close = self.near(frame, anchor, 'composer_close')
        title = self.near(frame, anchor, 'composer_wheat_title')
        new = self.near(frame, anchor, 'composer_title')
        submit = self.near(frame, anchor, 'composer_submit')
        if close is None or submit is None or not (title or new):
            return ShopView('composer', close=close)
        region = self.relative(anchor, (380, 200, 680, 820)).box
        if min(region) < 0 or region[0]+region[2] > frame.width or region[1]+region[3] > frame.height:
            return ShopView('composer', close=close)
        plus = self.near(frame, anchor, 'quantity_plus')
        maximum = self.near(frame, anchor, 'price_max')
        guards = [anchor, close, title or new, submit, plus, maximum]
        layout = dict(anchor=anchor, close=close, selected=bool(title), region=region,
                      submit=submit, plus=plus, maximum=maximum,
                      size=(frame.width, frame.height),
                      guards=[(t, image[t.y:t.y+t.height, t.x:t.x+t.width].copy())
                              for t in guards if t is not None])
        # Opening animations must not overwrite an already settled position.
        # Promote a new candidate only when a subsequent capture matches it.
        self._composer_candidates[bool(title)] = layout
        if bool(title) not in self._composer_layouts:
            self._composer_layouts[bool(title)] = layout
        return self._composer_values(frame, image, layout)

    def _inventory_icon(self, frame, region):
        # Stock sorting moves wheat between cells. Verify its last location in
        # a small crop first; search the inventory only when that check misses.
        previous = self._inventory_wheat
        wheat = None
        if previous is not None:
            x, y, w, h = region
            left, top = max(x, previous.x-8), max(y, previous.y-8)
            right = min(x+w, previous.x+previous.width+8)
            bottom = min(y+h, previous.y+previous.height+8)
            if right > left and bottom > top:
                wheat = self.one(frame.png, 'wheat_inventory', region=(left, top, right-left, bottom-top))
        if wheat is None:
            icons = self.matches(frame.png, 'wheat_inventory', region=region)
            wheat = icons[0] if len(icons) == 1 else None
        self._inventory_wheat = wheat
        return wheat

    def _composer_values(self, frame, image, layout, verified=False):
        anchor, region, title = layout['anchor'], layout['region'], layout['selected']
        wheat = self._inventory_icon(frame, region)
        stock = None
        if wheat:
            scale = wheat.width/117
            stock_box = (round(wheat.x+42*scale), round(wheat.y+56*scale), round(137*scale), round(87*scale))
            stock = read_number(frame.png, stock_box, image=image)
        quantity = read_number(frame.png, self.relative(anchor, (1257, 266, 95, 73)).box, suffix=True, image=image) if title else None
        price = read_number(frame.png, self.relative(anchor, (1338, 439, 78, 80)).box, image=image) if title else None
        checked_region = self.relative(anchor, (1130, 770, 230, 165)).box
        tick = self.one(frame.png, 'ad_check', region=checked_region, threshold=.90)
        # Composer has a larger newspaper than Edit Sale; its observed position
        # is usable only when an actual checked icon is matched there.
        newspaper = tick
        return ShopView('composer', close=layout['close'],
            inventory=VisualTarget(*region, anchor.score),
            wheat=self.relative(anchor, (1356, 237, 112, 125)) if title else wheat,
            stock=stock, quantity=quantity, price=price,
            plus_quantity=layout['plus'], max_price=layout['maximum'],
            minus_quantity=self._quantity_minus(image, anchor) if title else None,
            submit=layout['submit'] if self._enabled(image, layout['submit']) else None,
            newspaper=newspaper, ad_checked=bool(tick),
            silo_tab=self.near(frame, anchor, 'silo_tab') if wheat is None else None,
            layout_verified=verified)

    def _quantity_minus(self, image, anchor):
        """Verify the fixed quantity button's gold face and white minus glyph."""
        target = self.relative(anchor, (1135, 242, 119, 120))
        x, y, w, h = target.box
        if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
            return None
        hsv = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, (15, 120, 160), (42, 255, 255))
        if (gold > 0).mean() < .25:
            return None
        glyph = hsv[round(h*.18):round(h*.62), round(w*.16):round(w*.84)]
        white = cv2.inRange(glyph, (0, 0, 205), (179, 60, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(white)
        bars = [s for s in stats[1:] if .40*w <= s[2] <= .65*w
                and .09*h <= s[3] <= .24*h and s[4]/(s[2]*s[3]) > .75]
        return target if len(bars) == 1 else None

    def seed_stock(self, frame, seed):
        box = FarmingWorker._stock_box(frame, seed)
        return read_count(frame.png, box) if box else None

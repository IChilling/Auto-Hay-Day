"""Follow the game's resource links and perform one verified unit of resource work.

The order runner rereads inventory after every attempt. Production is never
assumed complete because a gesture succeeded or an icon resembled a product.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.order_vision import OrderReader
from hayday.resource_vision import ResourceVision
from hayday.work_scheduling import retry_delay, utc_timestamp


@dataclass(frozen=True)
class ResourceResult:
    status: str
    message: str
    details: dict = field(default_factory=dict)


class ResourceChanged(RuntimeError):
    pass


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ResourceWorker:
    """Fixed-device, cancellable production traversal with a bounded dependency depth."""

    def __init__(self, client, *, capture=None, cancel_event=None, progress=None,
                 state_path: Path, vision=None, fields=None, fruits=None, orchards=None,
                 max_depth=6):
        self.client = client
        self.capture = capture or client.capture
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress = progress or (lambda message: None)
        self.state_path = Path(state_path)
        self.vision = vision or ResourceVision()
        self.fields = fields
        self.fruits = fruits
        self.orchards = orchards
        self.max_depth = max_depth
        if type(max_depth) is not int or not 0 <= max_depth <= 12:
            raise ValueError('Resource dependency depth must be an integer between 0 and 12.')
        self.serial = client.serial
        self._size = None
        self._menu_open = False
        self._state = {'version': 1, 'serial': self.serial, 'items': {}, 'events': []}
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if (not isinstance(self._state, dict) or self._state.get('version') != 1
                    or self._state.get('serial') != self.serial
                    or not isinstance(self._state.get('items'), dict)
                    or not isinstance(self._state.get('events'), list)):
                raise ValueError('Saved resource work is invalid; no resource input can be sent.')
            if any(not isinstance(entry, dict) for entry in self._state['items'].values()):
                raise ValueError('Saved resource item records are invalid; no resource input can be sent.')
        checkout = Path(__file__).resolve().parents[2]
        self.references = (checkout / 'images' / 'learned_items' if (checkout / 'pyproject.toml').is_file()
                           else self.state_path.parent / 'images' / 'learned_items')
        if self.state_path.with_name('fruit.json').is_file():
            self._ensure_fruit_worker()
        if self.state_path.with_name('orchard.json').is_file():
            self._ensure_orchard_worker()

    def _ensure_fruit_worker(self):
        if self.fruits is None:
            from hayday.fruit import FruitWorker
            self.fruits = FruitWorker(self.client, self._capture, self.cancel_event, self.progress,
                                      self.state_path.with_name('fruit.json'))

    def _ensure_orchard_worker(self):
        if self.orchards is None:
            from hayday.orchard import OrchardWorker
            self.orchards = OrchardWorker(
                self.client, self._capture, self.cancel_event, self.progress,
                self.state_path.with_name('orchard.json'),
            )

    def _check(self):
        if self.cancel_event.is_set():
            raise ResourceChanged('Resource work cancelled.')
        if not self.serial or self.client.serial != self.serial:
            raise ResourceChanged('Device selection changed during resource work.')

    def _wait(self, seconds):
        self._check()
        self.cancel_event.wait(seconds)
        self._check()

    def _capture(self):
        self._check()
        frame = self.capture()
        self._check()
        if self._size and self._size != (frame.width, frame.height):
            raise ResourceChanged('Resolution changed during resource work.')
        self._size = frame.width, frame.height
        return frame

    def _tap(self, center, frame):
        self._check()
        self.client.tap(*map(int, center), width=frame.width, height=frame.height)
        self._wait(.22)
        return self._capture()

    def _observe(self, frame):
        result = self.vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        return result

    def _record(self, key, event, **details):
        key = self._resolve_key(key)
        entry = self._state['items'].setdefault(key, {})
        entry.update(details)
        entry['last_event'] = event
        entry['updated_at'] = datetime.now(UTC).isoformat()
        self._state['events'].append({'item': key, 'event': event, **details,
                                      'at': entry['updated_at']})
        self._state['events'] = self._state['events'][-200:]
        _save_json(self.state_path, self._state)

    def _resolve_key(self, key):
        seen = set()
        while key not in seen:
            seen.add(key)
            entry = self._state['items'].get(key, {})
            target = entry.get('alias_of')
            if not target:
                return key
            key = target
        raise ResourceChanged('Saved resource identities contain a loop.')

    @staticmethod
    def _normalized_icon(icon):
        image = cv2.imdecode(np.frombuffer(icon, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ResourceChanged('The selected item image could not be read.')
        if image.shape[2] == 4:
            mask = image[:, :, 3] >= 200
        else:
            cream = ResourceVision._cream(image).astype(np.uint8)
            _, labels = cv2.connectedComponents(cream)
            border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            mask = ~np.isin(labels, border[border != 0])
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            raise ResourceChanged('The selected item image has too little visible artwork.')
        image, mask = image[ys.min():ys.max()+1, xs.min():xs.max()+1, :3], mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        # Constant matte and size make identity independent of paper color,
        # screenshot PNG metadata, camera scale and transparent RGB values.
        matte = np.full_like(image, 128)
        matte[mask] = image[mask]
        return cv2.resize(matte, (80, 80), interpolation=cv2.INTER_AREA)

    def _known_icon(self, icon):
        normalized = self._normalized_icon(icon)
        interiors = []
        for key, entry in self._state['items'].items():
            # Aliases can retain a distinct board/recipe crop. Recognize that
            # artwork, but always route its stock and work to the canonical item.
            names = entry.get('icon_references', [key + '.png'])
            for name in names:
                if not isinstance(name, str) or Path(name).name != name:
                    continue
                path = self.references / name
                if not path.is_file():
                    continue
                reference = self._normalized_icon(path.read_bytes())
                difference = np.abs(normalized.astype(float)-reference.astype(float)).mean()
                if difference < 8:
                    return self._resolve_key(key)
                # Ready checks shift the board's item-body crop by a few
                # pixels. Retain a large native interior rather than lowering
                # the identity threshold for the entire resized silhouette.
                if isinstance(self.vision, ResourceVision):
                    reference_image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
                    height, width = reference_image.shape[:2]
                    if min(width, height) >= 35:
                        patch = reference_image[round(height*.15):round(height*.85), round(width*.15):round(width*.85)]
                        success, encoded = cv2.imencode('.png', patch)
                        if not hasattr(self, '_identity_matcher'):
                            self._identity_matcher = ResourceVision()
                        if success and self._identity_matcher.find_item(icon, encoded.tobytes(), min_scale=.85,
                                max_scale=1.2, threshold=.97, cancel=self.cancel_event.is_set):
                            interiors.append(self._resolve_key(key))
        distinct = set(interiors)
        if len(distinct) == 1:
            return distinct.pop()
        return None

    def _learn(self, icon, *, source: Screenshot, box=None, parent=None):
        key = self._known_icon(icon)
        if key:
            return key
        key = hashlib.sha256(self._normalized_icon(icon).tobytes()).hexdigest()[:20]
        self.references.mkdir(parents=True, exist_ok=True)
        target = self.references / (key + '.png')
        if not target.exists():
            target.write_bytes(icon)
            _save_json(target.with_suffix('.json'), {
                'source': 'Direct BlueStacks ADB screenshot', 'captured_at': source.captured_at,
                'source_sha256': hashlib.sha256(source.png).hexdigest(),
                'source_size': [source.width, source.height], 'crop_xywh': box,
                'parent_item': parent, 'transformation': 'Original cropped pixels; runtime excludes boundary-connected cream for matching',
            })
        self._record(key, 'icon_observed', icon_references=[target.name])
        return key

    def _bind_title(self, key, title):
        """Bind different icon crops to one durable identity using item-name artwork."""
        key = self._resolve_key(key)
        matches = []
        for other, entry in self._state['items'].items():
            name = entry.get('title_reference')
            if entry.get('alias_of') or not isinstance(name, str) or Path(name).name != name:
                continue
            path = self.references / name
            if path.is_file() and self.vision.titles_match(title, path.read_bytes()):
                matches.append(other)
        if matches:
            # A pre-existing pending batch has priority over a newly observed crop.
            canonical = next((candidate for candidate in matches if self._state['items'][candidate].get('pending_batch')), matches[0])
            if canonical != key:
                old, current = self._state['items'].get(key, {}), self._state['items'][canonical]
                references = sorted(set(current.get('icon_references', [canonical+'.png']))
                                    | set(old.get('icon_references', [key+'.png'])))
                if old.get('pending_batch'):
                    current.update({name: value for name, value in old.items()
                                    if name.startswith(('pending_', 'empty_', 'collection_', 'awaiting_'))})
                current['icon_references'] = references
                self._state['items'][key] = {'alias_of': canonical}
                key = canonical
        name = key + '.title.png'
        self.references.mkdir(parents=True, exist_ok=True)
        (self.references / name).write_bytes(title)
        self._record(key, 'identity_confirmed', title_reference=name)
        return key

    def _inventory_confirmed(self, key, available=None):
        key = self._resolve_key(key)
        entry = self._state['items'].get(key, {})
        if entry.get('pending_batch') and entry.get('pending_stage') == 'awaiting_collection':
            self._record(key, 'inventory_confirmed', pending_batch=False,
                         pending_stage=None, collection_probed=False, inventory_confirmation=None)

    def observe_inventory(self, frame, items):
        """Retire a finished batch only when an observed requirement is fulfilled."""
        self._check()
        for item in items:
            available, required = self._quantity(frame, item.quantity_bounds, item)
            if item.status != 'fulfilled' and type(available) is not int:
                continue
            key = self._known_icon(OrderReader.item_icon(frame.png, item))
            if key:
                self._acknowledge_inventory(key, item.status, available, required, frame.captured_at)

    def _acknowledge_inventory(self, key, status, available, required, observation_id=None):
        key = self._resolve_key(key)
        if self.fruits is not None:
            self.fruits.observe_inventory(key, status, available, required, observation_id)
        if self.orchards is not None:
            self.orchards.observe_inventory(key, status, available, required, observation_id)
        before = self._state['items'].get(key, {}).get('inventory_before', {})
        previous = before.get('available')
        same_requirement = type(required) is int and required > 0 and before.get('required') == required
        increased = same_requirement and type(available) is int and type(previous) is int and available > previous
        # Another order can require fewer of the same item. Its green check is
        # not evidence that stock increased unless the requirement also agrees.
        same_requirement_fulfilled = (
            status == 'fulfilled' and before.get('status') == 'missing'
            and same_requirement
        )
        entry = self._state['items'].get(key, {})
        if entry.get('work_retry_at') and (increased or same_requirement_fulfilled):
            self._record(key, 'work_recheck_due', work_retry_at=None)
        if not entry.get('pending_batch') or entry.get('pending_stage') != 'awaiting_collection':
            return
        if not (increased or same_requirement_fulfilled):
            if entry.get('inventory_confirmation'):
                self._record(key, 'inventory_unconfirmed', inventory_confirmation=None)
            return
        if observation_id is None:
            return
        evidence = [available, required, status]
        confirmation = entry.get('inventory_confirmation') or {}
        if confirmation.get('observation_id') == observation_id:
            return
        count = confirmation.get('count', 0)+1 if confirmation.get('evidence') == evidence else 1
        self._record(key, 'inventory_confirmation', inventory_confirmation={
            'evidence': evidence, 'observation_id': observation_id, 'count': count,
        })
        if count >= 2:
            self._inventory_confirmed(key, available)

    @staticmethod
    def _quantity(frame, box, item=None):
        available, required = getattr(item, 'available', None), getattr(item, 'required', None)
        if type(available) is int:
            return available, required
        try:
            from hayday.quantities import read_quantity
        except ImportError:
            return None, required if type(required) is int else None
        quantity = read_quantity(frame.png, box)
        return (quantity.available, quantity.required) if quantity is not None else (
            None, required if type(required) is int else None
        )

    def _baseline(self, key, frame, item):
        if self._state['items'].get(key, {}).get('pending_batch'):
            return
        name = key + '.quantity-before.png'
        x, y, w, h = item.quantity_bounds
        with Image.open(io.BytesIO(frame.png)) as image:
            image.crop((x, y, x+w, y+h)).save(self.references / name)
        available, required = self._quantity(frame, item.quantity_bounds, item)
        self._record(key, 'inventory_baseline', inventory_before={
            'quantity_reference': name, 'quantity_bounds': list(item.quantity_bounds),
            'status': item.status, 'available': available,
            'required': required, 'captured_at': frame.captured_at,
        })

    def pending_reconciliation(self, frame, item, *, item_key=None):
        """Read pending fruit stock at the order board before any location tap.

        FruitWorker owns the immutable harvest intent and its positive two-frame
        confirmation. This query never changes that intent or authorizes a retry.
        ``item_key`` carries a previous result's identity through reconciliation.
        """
        self._check()
        state = getattr(self.fruits, 'state', None)
        if not isinstance(state, dict) or not isinstance(state.get('items'), dict):
            return None
        selected_key = self._known_icon(OrderReader.item_icon(frame.png, item))
        selected_key = self._resolve_key(selected_key) if selected_key else None
        key = self._resolve_key(item_key) if item_key else selected_key
        entry = state['items'].get(key)
        if not isinstance(entry, dict):
            return None
        if entry.get('stage') == 'confirmed':
            return (ResourceResult('collected', 'The pending fruit harvest has two positive stock confirmations.',
                                   {'item': key, 'pending_harvest': False, 'inventory_confirmed': True})
                    if item_key else None)
        if entry.get('stage') != 'attempted':
            return None
        details = {'item': key, 'operation': entry.get('operation'), 'pending_harvest': True,
                   'inventory_only': True, 'wait_seconds': 3, 'observation_id': frame.captured_at}
        if selected_key != key:
            # Reopen a known parent recipe to read its ingredient stock. The
            # recipe path below confirms the pending harvest in two frames
            # before production or another ingredient visit can proceed.
            ancestor, seen = selected_key, set()
            while ancestor and ancestor not in seen and len(seen) <= self.max_depth:
                seen.add(ancestor)
                child = self._state['items'].get(ancestor, {}).get('dependency')
                ancestor = self._resolve_key(child) if child else None
                if ancestor == key:
                    return None
            return ResourceResult('unsupported',
                'An earlier fruit harvest is still unconfirmed for a recipe ingredient. Its intent is preserved; '
                'inspect that ingredient’s stock before another resource visit.', details)
        available, required = self._quantity(frame, item.quantity_bounds, item)
        before = entry.get('inventory_before', {})
        details.update(available=available, required=required, stock_status=item.status,
            unchanged_stock=(item.status == 'missing' and type(available) is int and type(required) is int
                             and type(before.get('available')) is int and before.get('required') == required
                             and available == before['available']))
        return ResourceResult('waiting',
            'The fruit harvest is unconfirmed. Checking its order stock without revisiting the tree or repeating the drag.',
            details)

    @staticmethod
    def _patch(frame, box):
        x, y, w, h = box
        with Image.open(io.BytesIO(frame.png)) as image:
            return np.asarray(image.convert('RGB').crop((x, y, x+w, y+h))).copy()

    def _unchanged_patch(self, old, new, box):
        a, b = self._patch(old, box), self._patch(new, box)
        return a.shape == b.shape and float(np.abs(a.astype(float)-b.astype(float)).mean()) < 12

    def _dismiss(self, frame):
        """Tap observed open ground; Android Back opens Hay Day's exit dialog."""
        ground = CameraNavigator._grass_start(frame, 0, 0)
        if ground is None:
            raise ResourceChanged('No clear farm ground is visible to close the resource menu.')
        return self._tap(ground[:2], frame)

    def _finish(self, status, message, frame, **details):
        # Every return leaves a visible farm when possible so board recovery can
        # inspect ground, instead of trying to pan a production recipe panel.
        if frame is not None:
            scene = self._observe(frame)
            if scene.popups or scene.empty_slots or self._menu_open:
                frame = self._dismiss(frame)
                scene = self._observe(frame)
                if scene.popups or scene.empty_slots:
                    self._dismiss(frame)
        self._menu_open = False
        return ResourceResult(status, message, details)

    def _wait_remaining(self, key):
        if not key:
            return 0.
        value = self._state['items'].get(self._resolve_key(key), {}).get('work_retry_at')
        if type(value) not in (int, float) or not np.isfinite(value):
            return 0.
        return min(600., max(0., value-utc_timestamp()))

    def _remember_wait(self, key, result):
        delay = retry_delay(result)
        if delay is not None:
            self._record(key, 'work_deferred', work_retry_at=utc_timestamp()+delay)
        return result

    def _follow_location(self, popup):
        """Follow a still-visible item prompt before doing expensive bookkeeping."""
        fresh = self._capture()
        observed_at = time.monotonic()
        scene = self._observe(fresh)
        prompts = [p for p in scene.popups if p.navigation and not p.rows]
        if (len(prompts) != 1 or len(prompts[0].navigation) != 1
                or not self.vision.titles_match(popup.title_png, prompts[0].title_png)
                or time.monotonic()-observed_at > 2.0):
            raise ResourceChanged('The item location prompt changed or expired before navigation; it will be read again.')
        self._tap(prompts[0].navigation[0].center, fresh)
        self._wait(.5)
        return self._capture()

    def work(self, frame: Screenshot, item) -> ResourceResult:
        try:
            self._check()
            self._menu_open = False
            self._size = frame.width, frame.height
            pending = self.pending_reconciliation(frame, item)
            if pending is not None:
                return pending
            icon = OrderReader.item_icon(frame.png, item)
            key = self._learn(icon, source=frame, box=item.icon_bounds)
            remaining = self._wait_remaining(key)
            if remaining:
                return ResourceResult('waiting', 'This item already has work in progress; checking other requirements.',
                    {'item': key, 'defer_item': True, 'retry_after_seconds': remaining})
            fresh = self._capture()
            if not self._unchanged_patch(frame, fresh, item.bounds):
                return ResourceResult('changed', 'The requested item changed before resource navigation.')
            popup_frame = self._tap(item.center, fresh)
            scene = self._observe(popup_frame)
            prompts = [popup for popup in scene.popups if popup.navigation and not popup.rows]
            if len(prompts) != 1:
                return ResourceResult('unsupported', 'The item location prompt was not recognized. Its screenshot is saved.')
            popup = prompts[0]
            self.progress('Following the selected item’s own resource-location link.')
            located = self._follow_location(popup)
            # Title matching, OCR and durable writes can outlast the short-lived
            # popover. Navigate first, then save the original board inventory
            # before any harvesting or production can occur at the source.
            key = self._bind_title(key, popup.title_png)
            self._baseline(key, frame, item)
            self._record(key, 'location_observed', reference=str(self.references / (key + '.png')))
            return self._remember_wait(key, self._at_source(located, icon, key, popup.title_png, (), 0))
        except ResourceChanged as exc:
            return ResourceResult('changed', str(exc))

    def _at_source(self, frame, icon, key, expected_title, ancestors, depth):
        self._check()
        if depth > self.max_depth or key in ancestors:
            return self._finish('unsupported', 'A resource dependency loop or depth limit was reached.', frame)
        scene = self._observe(frame)
        # A nested ingredient may first show its own location prompt.
        prompts = [p for p in scene.popups if p.navigation and not p.rows]
        if len(prompts) == 1:
            p = prompts[0]
            frame = self._follow_location(p)
            expected_title = p.title_png
            scene = self._observe(frame)
        if expected_title:
            key = self._bind_title(key, expected_title)
            if key in ancestors:
                return self._finish('unsupported', 'A resource dependency identity returned to an ancestor.', frame)

        # Field and animal gestures are recognized separately from production
        # queues, using actual tool+target references from the observed game.
        # An animal's timer describes that animal, not its entire pen. Handle
        # sheep before the generic growth bar so ready neighbors can be
        # collected and hungry neighbors fed during the same resource visit.
        self._ensure_fruit_worker()
        fruit_result = self.fruits.work_if_recognized(
            frame, icon, key, baseline=self._state['items'].get(key, {}).get('inventory_before')
        )
        if fruit_result is not None:
            if fruit_result.details.get('storage_blocked'):
                return fruit_result
            self._menu_open = True
            return self._finish(fruit_result.status, fruit_result.message, self._capture(),
                                **fruit_result.details)

        if self.fields is None:
            from hayday.farming import FarmingWorker
            self.fields = FarmingWorker(self.client, self._capture, self.cancel_event,
                                        self.progress, self.state_path.with_name('farming.json'))
        farm_result = self.fields.work_if_recognized(frame, icon, key)
        if farm_result is not None:
            return farm_result

        # The requested item's own location link can open Trees & Bushes when
        # no harvestable plant exists.  Only the orchard worker may interpret
        # that explicit catalog state and authorize one guarded placement.
        self._ensure_orchard_worker()
        orchard_result = self.orchards.work_if_recognized(
            frame, icon, key, baseline=self._state['items'].get(key, {}).get('inventory_before')
        )
        if orchard_result is not None:
            canonical = orchard_result.details.get('item')
            if canonical in self._state['items'] and self._resolve_key(key) != canonical:
                if self._state['items'].get(key, {}).get('pending_batch'):
                    raise ResourceChanged('Orchard identity conflicts with saved production; its intent is preserved.')
                self._record(key, 'orchard_identity_confirmed', alias_of=canonical)
            if orchard_result.details.get('placement_mode'):
                # Tapping generic grass while a purchase ghost is active could
                # spend coins. OrchardWorker must resolve or cancel that state.
                return orchard_result
            self._menu_open = bool(orchard_result.details.get('catalog_open'))
            return self._finish(orchard_result.status, orchard_result.message, self._capture(),
                                **orchard_result.details)

        state = self._state['items'].get(key, {})
        if (state.get('pending_batch') and state.get('pending_stage') == 'awaiting_collection'
                and isinstance(self.vision, ResourceVision)):
            from hayday.production_ready import ReadyProductVision, collect_ready
            if not hasattr(self, '_ready_products'):
                self._ready_products = ReadyProductVision()
            if self._ready_products.supports(icon, expected_title, self.cancel_event.is_set):
                result = collect_ready(self, frame, icon, expected_title, key)
                if result is not None:
                    return result
                return self._finish('waiting',
                    'The finished product and its machine could not both be verified; checking other items before retrying.',
                    self._capture(), item=key, defer_item=True, retry_after_seconds=30)
        reference = self.vision.production_reference(icon, expected_title, frame.height)
        minimum_score = .84
        if isinstance(reference, bytes):
            icon, minimum_score = reference, .90
        # Prefer an observed producer menu over the generic world-item probe.
        # Product artwork can also be part of a machine's decoration (the Loom
        # is a real example), so closing a usable producer menu and matching
        # that decoration could tap an unrelated object instead of crafting.
        menu_items = self.vision.find_item(
            frame.png, icon, min_scale=.8, max_scale=3.4,
            region=(0, 0, int(frame.width*.64), int(frame.height*.86)),
            cancel=self.cancel_event.is_set, threshold=minimum_score,
        )
        visible_slots = tuple(
            target for target in scene.empty_slots if self._outside_popups(target, scene)
        )
        # The camera and floating product artwork can still be settling after
        # a location jump. Visible EMPTY slots identify a producer view: give
        # its requested product two fresh observations before considering any
        # world collection candidate or dismissing the menu.
        for attempt in range(2):
            if menu_items or not visible_slots:
                break
            if attempt == 0:
                self.progress('Waiting for the requested product to settle in the production menu.')
            self._wait(.25)
            frame = self._capture()
            scene = self._observe(frame)
            menu_items = self.vision.find_item(
                frame.png, icon, min_scale=.8, max_scale=3.4,
                region=(0, 0, int(frame.width*.64), int(frame.height*.86)),
                cancel=self.cancel_event.is_set, threshold=minimum_score,
            )
            visible_slots = tuple(
                target for target in scene.empty_slots if self._outside_popups(target, scene)
            )
        if visible_slots and not menu_items:
            return self._finish('unsupported',
                'The production menu opened, but the requested product could not be verified in three observations.',
                frame, item=key)
        producer_menu = bool(menu_items and visible_slots)

        # A positively tracked finished batch is the one case where the world
        # output must be collected before another batch is considered. Close
        # the producer menu first, then make the generic search in a fresh farm
        # frame where there is no credible menu candidate to confuse with it.
        awaiting_collection = (
            state.get('pending_batch')
            and state.get('pending_stage') == 'awaiting_collection'
            and not state.get('collection_probed')
        )
        if producer_menu and awaiting_collection:
            frame = self._dismiss(frame)
            scene = self._observe(frame)
            menu_items = self.vision.find_item(
                frame.png, icon, min_scale=.8, max_scale=3.4,
                region=(0, 0, int(frame.width*.64), int(frame.height*.86)),
                cancel=self.cancel_event.is_set, threshold=minimum_score,
            )
            visible_slots = tuple(
                target for target in scene.empty_slots if self._outside_popups(target, scene)
            )
            producer_menu = bool(menu_items and visible_slots)

        # Collect only once per visit-cycle before considering another batch.
        # A small matched world product is still merely a candidate; return to
        # the order board to establish whether inventory actually increased.
        if not state.get('collection_probed') and not producer_menu:
            candidates = self.vision.find_item(frame.png, icon, min_scale=.22, max_scale=.72,
                                               cancel=self.cancel_event.is_set)
            candidates = [c for c in candidates if frame.width*.14 < c.center[0] < frame.width*.86
                          and frame.height*.2 < c.center[1] < frame.height*.84]
            if candidates:
                frame = self._dismiss(frame)
                candidates = self.vision.find_item(frame.png, icon, min_scale=.22, max_scale=.72,
                                                   cancel=self.cancel_event.is_set)
                candidates = [c for c in candidates if frame.width*.14 < c.center[0] < frame.width*.86
                              and frame.height*.2 < c.center[1] < frame.height*.84]
                if candidates:
                    target = max(candidates, key=lambda c: c.score)
                    self._record(key, 'collection_attempted', collection_probed=True)
                    frame = self._tap(target.center, frame)
                    return self._finish('waiting', 'Collection attempted; returning to verify order inventory.', frame,
                                        wait_seconds=1, item=key)
                self._record(key, 'collection_checked', collection_probed=True)
                return self._finish('waiting', 'Collection view changed; returning to reopen the resource location.', frame,
                                    wait_seconds=1, item=key)
            self._record(key, 'collection_checked', collection_probed=True)

        # A production menu has large product icons and explicit EMPTY labels.
        # Never treat the plus/diamond slot as an available queue slot.
        if not menu_items:
            return self._finish('unsupported', 'The required product or a supported harvest tool was not found at its source.', frame,
                                item=key)
        menu = max(menu_items, key=lambda c: c.score)
        self._menu_open = True
        if state.get('pending_batch'):
            return self._pending_at_source(frame, scene, menu, key)
        frame = self._tap(menu.center, frame)
        scene = self._observe(frame)
        recipes = [p for p in scene.popups if p.rows
                   and (not expected_title or self.vision.titles_match(expected_title, p.title_png))]
        if len(recipes) != 1:
            return self._finish('unsupported', 'The selected product’s full ingredient recipe was not recognized.', frame, item=key)
        recipe = recipes[0]
        key = self._bind_title(key, recipe.title_png)
        if self._state['items'][key].get('pending_batch'):
            return self._pending_at_source(frame, scene, menu, key)
        if not recipe.recipe_complete or any(row.missing is None for row in recipe.rows):
            return self._finish('unsupported', 'One or more ingredient amounts could not be verified.', frame, item=key)
        for row in recipe.rows:
            if row.missing is not None:
                ingredient = self._known_icon(row.icon_png)
                if ingredient:
                    available, required = self._quantity(frame, row.quantity_box)
                    self._acknowledge_inventory(ingredient, 'missing' if row.missing else 'fulfilled',
                                                available, required, frame.captured_at)
        fruit_items = getattr(self.fruits, 'state', {}).get('items', {})
        pending = [self._known_icon(row.icon_png) for row in recipe.rows]
        pending = [ingredient for ingredient in pending if isinstance(fruit_items, dict)
                   and fruit_items.get(ingredient, {}).get('stage') == 'attempted']
        if pending:
            self.progress('Checking harvested ingredient stock in two fresh recipe observations.')
            fresh = self._capture()
            fresh_scene = self._observe(fresh)
            checked = [p for p in fresh_scene.popups if p.recipe_complete
                       and len(p.rows) == len(recipe.rows)
                       and self.vision.titles_match(recipe.title_png, p.title_png)]
            if len(checked) != 1:
                return self._finish('changed', 'The ingredient stock view changed during harvest confirmation.', fresh, item=key)
            for row in checked[0].rows:
                ingredient = self._known_icon(row.icon_png)
                if ingredient in pending and row.missing is not None:
                    available, required = self._quantity(fresh, row.quantity_box)
                    self._acknowledge_inventory(ingredient, 'missing' if row.missing else 'fulfilled',
                                                available, required, fresh.captured_at)
            unresolved = [ingredient for ingredient in pending
                          if fruit_items[ingredient].get('stage') == 'attempted']
            if unresolved:
                return self._finish('waiting',
                    'The ingredient harvest is still unconfirmed; checking other items before rereading its recipe stock.',
                    fresh, item=unresolved[0], pending_harvest=True, defer_item=True, retry_after_seconds=60)
            frame, scene, recipe = fresh, fresh_scene, checked[0]
        missing = [row for row in recipe.rows if row.missing]
        if missing:
            eligible = [row for row in missing if not self._wait_remaining(self._known_icon(row.icon_png))]
            if not eligible:
                delay = min(self._wait_remaining(self._known_icon(row.icon_png)) for row in missing)
                return self._finish('waiting', 'All missing recipe ingredients have work in progress; checking other items.',
                    frame, item=key, defer_item=True, retry_after_seconds=delay)
            row = eligible[0]
            child = self._learn(row.icon_png, source=frame, box=row.item_box, parent=key)
            self._record(key, 'ingredient_needed', dependency=child)
            self.progress(f'Following a missing ingredient (dependency level {depth+1}).')
            from types import SimpleNamespace
            self._baseline(child, frame, SimpleNamespace(quantity_bounds=row.quantity_box, status='missing'))
            self._tap(row.navigation.center, frame)
            self._wait(.5)
            result = self._remember_wait(child, self._at_source(
                self._capture(), row.icon_png, child, None, (*ancestors, key), depth+1))
            if retry_delay(result) is not None and len(eligible) > 1:
                # Reopen the parent from its location link on the next pass;
                # sibling row coordinates became stale when the camera moved.
                details = {name: value for name, value in result.details.items()
                           if name not in {'defer_item', 'defer_session', 'orchard_pending', 'retry_after_seconds'}}
                return ResourceResult(result.status, result.message+' Another recipe ingredient remains to check.',
                    {**details, 'more_ingredients': True, 'wait_seconds': 1})
            return result

        # Keep the recipe visible. The product and one explicit EMPTY label must
        # both remain outside every popup. This permits all three observations
        # (ingredients, product, destination) to come from the same fresh frame.
        fresh = self._capture()
        verified_at = time.monotonic()
        fresh_scene = self._observe(fresh)
        checked = [popup for popup in fresh_scene.popups if popup.rows and (
            self.vision.titles_match(recipe.title_png, popup.title_png)
        )]
        if len(checked) != 1 or not checked[0].recipe_complete or (
            len(checked[0].rows) != len(recipe.rows)
        ) or any(row.missing is not False for row in checked[0].rows):
            return self._finish('changed', 'The full ingredient recipe changed before queuing.', fresh, item=key)
        slots = sorted((target for target in fresh_scene.empty_slots if self._outside_popups(target, fresh_scene)),
                       key=lambda target: target.center[0])
        products = self.vision.find_item(fresh.png, icon, min_scale=.8, max_scale=3.4,
                                        region=(0, 0, int(fresh.width*.64), int(fresh.height*.86)),
                                        cancel=self.cancel_event.is_set, threshold=minimum_score)
        products = [target for target in products if self._outside_popups(target, fresh_scene)
                    # The observed product floats vertically when its recipe
                    # opens (24 px for the 119 px ice-cream reference). Its
                    # freshly matched artwork and title still identify it.
                    and self._near(target.center, menu.center, max(8, menu.width*.30))]
        if not slots:
            return self._finish('waiting', 'No EMPTY queue target remains visible outside the recipe; no production gesture was sent.', fresh,
                                wait_seconds=30, item=key, defer_item=True, retry_after_seconds=30)
        if len(products) != 1 or time.monotonic()-verified_at > 2.0:
            return self._finish('changed', 'The product could not be revalidated while its ingredient evidence was fresh.', fresh, item=key)
        menu, slot = products[0], slots[0]
        # Persist uncertain intent before input; a command error must not cause
        # a second batch to be started automatically on restart.
        self._record(key, 'queue_attempted', pending_batch=True, pending_stage='uncertain',
                     empty_before=len(slots), collection_probed=False, inventory_confirmation=None, pending_slot={
                         'dx': slot.center[0]-menu.center[0], 'dy': slot.center[1]-menu.center[1],
                         'menu_width': menu.width, 'label_width': slot.width,
                     })
        self._check()
        self.client.swipe(*menu.center, *slot.center, width=fresh.width,
                          height=fresh.height, duration_ms=650)
        self._wait(.65)
        after = self._capture()
        changed = self._queue_changed(after, icon, menu, slot, minimum_score)
        if changed:
            self._wait(.45)
            after = self._capture()
            changed = self._queue_changed(after, icon, menu, slot, minimum_score)
        if not changed:
            return self._finish('unsupported', 'Production input was attempted, but its queue change is unconfirmed; no repeat batch will be sent.', after,
                                item=key, pending_batch=True)
        self._record(key, 'batch_queued', pending_batch=True, pending_stage='queued')
        return self._finish('queued', 'One batch was queued after verifying every ingredient. Returning to check the order.', after,
                            item=key, wait_seconds=1, defer_item=True, retry_after_seconds=60)

    @staticmethod
    def _near(first, second, distance):
        return abs(first[0]-second[0]) <= distance and abs(first[1]-second[1]) <= distance

    @staticmethod
    def _outside_popups(target, scene):
        cx, cy = target.center
        return not any(x-10 <= cx <= x+w+10 and y-10 <= cy <= y+h+10
                       for x, y, w, h in (popup.box for popup in scene.popups))

    def _queue_changed(self, frame, icon, previous_menu, previous_slot, minimum_score=.84):
        scene = self._observe(frame)
        if scene.popups:
            return False
        products = self.vision.find_item(frame.png, icon, min_scale=.8, max_scale=3.4,
                                        region=(0, 0, int(frame.width*.64), int(frame.height*.86)),
                                        cancel=self.cancel_event.is_set, threshold=minimum_score)
        menus = [product for product in products if self._near(
            product.center, previous_menu.center, max(8, previous_menu.width*.30)
        )]
        if len(menus) != 1:
            return False
        center = (previous_slot.center[0]+menus[0].center[0]-previous_menu.center[0],
                  previous_slot.center[1]+menus[0].center[1]-previous_menu.center[1])
        return not any(self._near(slot.center, center, max(12, previous_slot.width*.4))
                       for slot in scene.empty_slots)

    def _pending_at_source(self, frame, scene, menu, key):
        state = self._state['items'][key]
        marker = state.get('pending_slot')
        if state.get('pending_stage') in {'queued', 'awaiting_collection'} and isinstance(marker, dict):
            scale = menu.width / max(1, marker.get('menu_width', menu.width))
            center = (menu.center[0]+marker['dx']*scale, menu.center[1]+marker['dy']*scale)
            if any(self._near(slot.center, center, max(12, marker['label_width']*scale*.4))
                   for slot in scene.empty_slots):
                if state.get('pending_stage') == 'queued':
                    self._record(key, 'queue_vacated', pending_batch=True,
                                 pending_stage='awaiting_collection', collection_probed=False)
                    return self._finish('waiting', 'The queued batch left its slot; collection and inventory confirmation are still required.', frame,
                                        wait_seconds=1, item=key)
                if state.get('collection_probed'):
                    # A probe can miss a briefly covered world icon. A later,
                    # positively empty source slot permits another fresh search,
                    # while preserving the original batch and inventory baseline.
                    self._record(key, 'collection_recheck_due', collection_probed=False)
                    return self._finish('waiting', 'The source slot remains empty; collection will be checked again on the next visit.', frame,
                                        wait_seconds=30, item=key, defer_item=True, retry_after_seconds=30)
        return self._finish('waiting', 'This item has pending production; no duplicate batch will be sent before inventory confirmation.', frame,
                            wait_seconds=30, item=key, pending_stage=state.get('pending_stage'),
                            defer_item=state.get('pending_stage') == 'queued', retry_after_seconds=60)

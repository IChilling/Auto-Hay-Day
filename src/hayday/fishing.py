"""Replenish red lures and perform bounded casts with durable stock confirmation."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from hayday.fishing_vision import FishingVision


class FishingWorker:
    def __init__(self, client, capture, cancel_event, progress, state_path, *, vision=None,
                 feedback_capture=None):
        self.client, self.capture = client, capture
        self.feedback_capture = feedback_capture or getattr(client, 'capture_fast', capture)
        self.cancel_event, self.progress = cancel_event, progress
        self.serial = client.serial
        self.state_path = Path(state_path)
        self.vision = vision or FishingVision()
        self.state = {'version': 1, 'serial': self.serial, 'items': {}}
        self._size = None
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text('utf-8'))
            if (not isinstance(self.state, dict) or self.state.get('version') != 1
                    or self.state.get('serial') != self.serial
                    or not isinstance(self.state.get('items'), dict)
                    or any(not isinstance(entry, dict) or entry.get('stage') not in {'attempted', 'confirmed', 'no_effect'}
                           for entry in self.state['items'].values())
                    or ('red_lure' in self.state and (
                        not isinstance(self.state['red_lure'], dict)
                        or self.state['red_lure'].get('stage') not in {'attempted', 'queued', 'stocked'}))):
                raise ValueError('Saved fishing intent is invalid; no cast can be sent.')

    def _check(self):
        from hayday.resources import ResourceChanged
        if self.cancel_event.is_set() or not self.serial or self.client.serial != self.serial:
            raise ResourceChanged('Fishing cancelled or device selection changed.')

    def _frame(self, *, feedback=False):
        from hayday.resources import ResourceChanged
        self._check()
        frame = self.feedback_capture() if feedback else self.capture()
        self._check()
        if self._size and self._size != (frame.width, frame.height):
            raise ResourceChanged('Resolution changed during fishing.')
        self._size = frame.width, frame.height
        return frame

    def _record(self, key, **values):
        from hayday.resources import _save_json
        self.state['items'].setdefault(key, {}).update(values)
        _save_json(self.state_path, self.state)

    def observe_inventory(self, key, status, available, required, observation_id):
        # The shared harvest contract requires two distinct positive readings.
        from hayday.harvest_inventory import observe_harvest_inventory
        observe_harvest_inventory(self, key, status, available, required, observation_id)
        entry = self.state['items'].get(key, {})
        if entry.get('stage') != 'attempted' or not entry.get('after_capture'):
            return
        before = entry.get('inventory_before', {})
        unchanged = (status in {'missing', 'fulfilled'} and type(available) is int
                     and type(required) is int and required > 0
                     and available == before.get('available') and required == before.get('required'))
        if not unchanged or not observation_id:
            if entry.get('inventory_unchanged'):
                self._record(key, inventory_unchanged=None)
            return
        confirmation = entry.get('inventory_unchanged') or {}
        if confirmation.get('observation_id') == observation_id:
            return
        self._record(key, inventory_unchanged={
            'count': min(2, confirmation.get('count', 0)+1), 'observation_id': observation_id})

    def can_check_returned_lure(self, key):
        entry = self.state['items'].get(key, {})
        return (entry.get('stage') == 'attempted' and bool(entry.get('after_capture'))
                and (entry.get('inventory_unchanged') or {}).get('count', 0) >= 2)

    def _record_lure(self, **values):
        from hayday.resources import _save_json
        if 'operation' in values and values['operation'] != self.state.get('red_lure', {}).get('operation'):
            self.state['red_lure'] = {}
        self.state.setdefault('red_lure', {}).update(values)
        _save_json(self.state_path, self.state)

    def _lure_wait(self, key, message, *, queued=False, stocked=False):
        from hayday.resources import ResourceResult
        self._return_home()
        return ResourceResult('queued' if queued else 'waiting', message,
            {'item': key, 'defer_item': True, 'retry_after_seconds': 1 if stocked else 600,
             'needs_red_lure': not stocked})

    def _replenish_lure(self, frame, menu, key):
        """Use the empty red lure's own location prompt to reach its producer."""
        from hayday.resources import ResourceResult
        self._check()
        self.client.tap(*menu.tool, width=frame.width, height=frame.height)
        self.cancel_event.wait(.22)
        # Location bubbles expire quickly; use checked captures without generic
        # dialog recovery while verifying this known, short-lived control.
        popup_frame = self._frame(feedback=True)
        popup = self.vision.lure_location(popup_frame.png, self.cancel_event.is_set)
        fresh = self._frame(feedback=True)
        observed_at = time.monotonic()
        checked = self.vision.lure_location(fresh.png, self.cancel_event.is_set)
        if (popup is None or checked is None or time.monotonic()-observed_at > 2
                or np.linalg.norm(np.subtract(popup.navigation[0].center,
                                              checked.navigation[0].center)) > 8):
            return self._lure_wait(key, 'The red-lure workbench link could not be verified; checking other requirements.')
        self._check()
        self.client.tap(*checked.navigation[0].center, width=fresh.width, height=fresh.height)
        self.cancel_event.wait(.7)
        for _ in range(3):
            frame = self._frame()
            bench = self.vision.workbench(frame.png, self.cancel_event.is_set)
            if bench:
                return self._queue_lure(frame, bench, key)
            self.cancel_event.wait(.4)
        return ResourceResult('unsupported', 'The red-lure workbench could not be verified; no production was started.')

    def _queue_lure(self, frame, bench, key):
        from hayday.resources import ResourceResult
        pending = self.state.get('red_lure', {})
        if bench.stock is not None and bench.stock > 0:
            return self._confirm_lure_stock(bench, key)
        if self._confirm_lure_queue(frame, bench):
            return self._lure_wait(key, 'One free red lure is producing; checking other order requirements.', queued=True)
        if pending.get('stage') == 'attempted':
            return ResourceResult('unsupported', 'A red-lure queue attempt remains unconfirmed; it will not be repeated.')
        if pending.get('stage') == 'queued':
            if self.vision.first_slot_empty(bench):
                return self._collect_lure(frame, bench, key)
            return self._lure_wait(key, 'Red-lure production is pending; checking other order requirements.')
        if bench.stock != 0 or not self.vision.first_slot_empty(bench):
            return self._lure_wait(key, 'Red-lure stock or an empty workbench slot is unavailable; checking other requirements.')
        fresh = self._frame()
        checked = self.vision.workbench(fresh.png, self.cancel_event.is_set)
        if (checked is None or checked.stock != 0 or not self.vision.first_slot_empty(checked)
                or np.linalg.norm(np.subtract(checked.tool, bench.tool)) > 12*bench.scale):
            return ResourceResult('changed', 'The lure workbench changed before inspecting its recipe.')
        self._check()
        self.client.tap(*checked.tool, width=fresh.width, height=fresh.height)
        self.cancel_event.wait(.25)
        recipe = self._frame(feedback=True)
        first = self.vision.workbench(recipe.png, self.cancel_event.is_set)
        if first is None or not self.vision.free_red_recipe(recipe.png, first, self.cancel_event.is_set):
            return ResourceResult('unsupported', 'The free red-lure recipe could not be verified; no production was started.')
        fresh = self._frame(feedback=True)
        observed_at = time.monotonic()
        checked = self.vision.workbench(fresh.png, self.cancel_event.is_set)
        if (checked is None or checked.stock != 0 or not self.vision.first_slot_empty(checked)
                or not self.vision.free_red_recipe(fresh.png, checked, self.cancel_event.is_set)
                or not first.empty_slots
                or np.linalg.norm(np.subtract(checked.empty_slots[0].center, first.empty_slots[0].center)) > 8
                or time.monotonic()-observed_at > 2):
            return ResourceResult('changed', 'The free lure recipe or EMPTY queue slot changed before production.')
        operation = uuid.uuid4().hex
        evidence = self.state_path.parent/'fishing_evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence/f'{operation}_lure_before.png').write_bytes(fresh.png)
        self._record_lure(stage='attempted', operation=operation, stock_before=0,
                          before_capture=f'{operation}_lure_before.png')
        self.progress('Queuing one free red lure in the verified EMPTY workbench slot.')
        self._check()
        slot = checked.empty_slots[0].center
        self.client.drag_path((checked.tool, checked.tool, slot, slot), width=fresh.width,
                              height=fresh.height, duration_ms=1000, min_waypoint_ms=120, max_step_px=12)
        self.cancel_event.wait(.7)
        after = self._frame()
        (evidence/f'{operation}_lure_after.png').write_bytes(after.png)
        self._record_lure(after_capture=f'{operation}_lure_after.png')
        bench = self.vision.workbench(after.png, self.cancel_event.is_set)
        if bench and self._confirm_lure_queue(after, bench):
            return self._lure_wait(key, 'One free red lure is producing; checking other order requirements.', queued=True)
        return ResourceResult('unsupported', 'Red-lure production was attempted, but its queue is unconfirmed; no repeat will be sent.')

    def _confirm_lure_queue(self, frame, bench):
        # Recognize the actual red lure in the first queue position twice.
        # An absent EMPTY label alone cannot prove successful input.
        if not self.vision.queued_red_lure(frame.png, bench, self.cancel_event.is_set):
            return False
        self.cancel_event.wait(.4)
        fresh = self._frame()
        checked = self.vision.workbench(fresh.png, self.cancel_event.is_set)
        if (checked is None or np.linalg.norm(np.subtract(bench.title.center, checked.title.center)) > 8
                or not self.vision.queued_red_lure(fresh.png, checked, self.cancel_event.is_set)):
            return False
        self._record_lure(stage='queued')
        return True

    def _confirm_lure_stock(self, bench, key):
        from hayday.resources import ResourceResult
        fresh = self._frame()
        checked = self.vision.workbench(fresh.png, self.cancel_event.is_set)
        if (checked is None or checked.stock != bench.stock
                or checked.stock is None or checked.stock <= 0):
            return ResourceResult('changed', 'Red-lure stock changed before confirmation; no further input was sent.')
        self._record_lure(stage='stocked', stock_after=checked.stock)
        return self._lure_wait(key, 'Red-lure stock is available; returning to recheck the fishing requirement.', stocked=True)

    def _collect_lure(self, frame, bench, key):
        """Reopen the verified workbench once, then read stock, never requeue."""
        from hayday.fishing_navigation import FishingNavigator, water_points
        from hayday.resources import ResourceResult
        # Lure artwork sways independently of the queue. Allow that short
        # animation to settle, without sending any input or weakening identity.
        checked = None
        for _ in range(6):
            fresh = self._frame()
            checked = self.vision.workbench(fresh.png, self.cancel_event.is_set)
            if checked is not None:
                break
            self.cancel_event.wait(.25)
        if (checked is None or not self.vision.first_slot_empty(checked)
                or np.linalg.norm(np.subtract(checked.title.center, bench.title.center)) > 8):
            return ResourceResult('changed', 'The lure queue changed before collection.')
        # The translucent lure palette contains green pixels over its tools.
        # Inspect open water instead of mistaking those pixels for tappable grass.
        clearance = 50*fresh.height/1080
        points = water_points(fresh.png, left=.12, minimum=clearance)
        water = FishingNavigator(self)._fresh_water(
            points[:1], left=.12, minimum=clearance) if points else None
        if water is None:
            return self._lure_wait(key, 'The workbench collection view is obscured; checking other requirements.')
        self._check()
        self.client.tap(*points[0], width=water.width, height=water.height)
        self.cancel_event.wait(.3)
        frame = self._frame()
        anchor = self.vision.workbench_anchor(frame.png, self.cancel_event.is_set)
        fresh = self._frame()
        other = self.vision.workbench_anchor(fresh.png, self.cancel_event.is_set)
        if (anchor is None or other is None or np.linalg.norm(np.subtract(anchor.center, other.center)) > 8
                or self.vision.workbench(fresh.png, self.cancel_event.is_set) is not None):
            return self._lure_wait(key, 'The lure workbench could not be revalidated for collection; checking other requirements.')
        self._record_lure(collection_probed=True)
        self._check()
        self.client.tap(*other.center, width=fresh.width, height=fresh.height)
        self.cancel_event.wait(.5)
        after = self._frame()
        checked = self.vision.workbench(after.png, self.cancel_event.is_set)
        if checked and checked.stock is not None and checked.stock > 0:
            return self._confirm_lure_stock(checked, key)
        return self._lure_wait(key, 'The lure workbench was checked; stock is not confirmed yet, so no replacement batch will be sent.')

    def _return_home(self):
        from hayday.resources import ResourceChanged
        attempts = 0
        for _ in range(12):
            frame = self._frame()
            home = self.vision.home(frame.png, self.cancel_event.is_set)
            if home is not None:
                fresh = self._frame()
                checked = self.vision.home(fresh.png, self.cancel_event.is_set)
                if checked and np.linalg.norm(np.subtract(home.center, checked.center)) < 8:
                    self._check()
                    self.client.tap(*checked.center, width=fresh.width, height=fresh.height)
                    attempts += 1
                    for _ in range(8):
                        self.cancel_event.wait(.5)
                        after = self._frame()
                        if self.vision.home(after.png, self.cancel_event.is_set) is None:
                            return
                    # The first touch can dismiss a catch photo. Only retry
                    # after recognizing the home control again in fresh frames.
                    if attempts >= 2:
                        raise ResourceChanged('The fishing-area home control did not return to the farm.')
            self.cancel_event.wait(.5)
        raise ResourceChanged('The fishing-area home control could not be verified.')

    def work_if_recognized(self, frame, icon, key, *, baseline=None):
        from hayday.resources import ResourceResult
        self._check()
        if not self.vision.supports(icon, self.cancel_event.is_set):
            return None
        pending = self.state['items'].get(key, {})
        if pending.get('stage') == 'attempted' and not self.can_check_returned_lure(key):
            return ResourceResult('waiting', 'A fishing attempt is awaiting order-stock confirmation.',
                                  {'item': key, 'pending_harvest': True, 'wait_seconds': 1})
        self._size = frame.width, frame.height
        observed = self.vision.menu(frame.png, self.cancel_event.is_set)
        if observed is None:
            if self.vision.cooldown(frame.png, self.cancel_event.is_set):
                fresh = self._frame()
                if not self.vision.cooldown(fresh.png, self.cancel_event.is_set):
                    return ResourceResult('changed', 'The fishing-area cooldown changed before returning home.')
                from hayday.fishing_navigation import FishingNavigator
                alternate = FishingNavigator(self).find_spot(fresh)
                if alternate is None:
                    self._return_home()
                    return ResourceResult('waiting',
                        'No available fishing picker was verified during the bounded water search; checking other requirements.',
                        {'item': key, 'defer_item': True, 'retry_after_seconds': 600, 'fishing_cooldown': True})
                frame = alternate
                observed = self.vision.menu(frame.png,self.cancel_event.is_set)
            if observed is None:
                return None
        fresh = self._frame()
        checked = self.vision.menu(fresh.png, self.cancel_event.is_set)
        if (checked is None or np.linalg.norm(np.subtract(observed.tool, checked.tool)) > 10*observed.scale
                or self.vision.home(fresh.png, self.cancel_event.is_set) is None):
            return ResourceResult('changed', 'The fishing controls changed before casting; no lure was used.')
        if checked.stock is None or observed.stock != checked.stock:
            return ResourceResult('unsupported', 'Red-lure stock could not be verified; no cast was sent.')
        if pending.get('stage') == 'attempted':
            # A completed failed cast returns its lure. Only unchanged fish
            # stock in two board readings AND unchanged positive lure stock in
            # these two picker readings establish no effect. Interrupted casts
            # and spent/unknown lures keep the original intent.
            returned = (checked.stock > 0 and checked.stock == pending.get('lure_stock_before'))
            if returned:
                self._record(key, stage='no_effect', lure_returned=True,
                             lure_stock_after=checked.stock)
            self._return_home()
            return ResourceResult('waiting',
                'Fish stock is unchanged and the lure was returned; the completed cast had no effect.'
                if returned else 'The previous catch and lure return remain unconfirmed; checking other requirements.',
                {'item': key, 'defer_item': True, 'retry_after_seconds': 1 if returned else 600,
                 'lure_returned': returned, 'pending_harvest': not returned})
        if checked.stock == 0:
            return self._replenish_lure(fresh, checked, key)
        if self.state.get('red_lure', {}).get('stage') in {'attempted', 'queued'}:
            self._record_lure(stage='stocked', stock_after=checked.stock)
        operation = uuid.uuid4().hex
        evidence = self.state_path.parent/'fishing_evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence/f'{operation}_before.png').write_bytes(fresh.png)
        self._record(key, stage='attempted', operation=operation,
                     inventory_before=dict(baseline or {}), inventory_confirmation=None,
                     inventory_unchanged=None, after_capture=None, lure_returned=False,
                     lure_stock_after=None, end_reason=None, observations=0,
                     lure_stock_before=checked.stock, before_capture=f'{operation}_before.png')
        self.progress('Casting one red lure and following the fish with fresh screen feedback.')
        scale = checked.scale
        center = None
        last_float = None
        missing = 0
        missing_ring = 0
        lost_frames = 0
        observations = 0
        tracking = []
        end_reason = 'time_limit'

        def update(point, elapsed):
            nonlocal center, last_float, missing, missing_ring, lost_frames, observations, end_reason
            current = self._frame(feedback=True)
            observations += 1
            end = self.vision.line_end(current.png, point)
            if end is None:
                missing += 1
            else:
                missing = 0
                if center is None and self.vision.bobber(current.png, end):
                    last_float = (end[0], round(end[1]+22*scale))
                elif center is None and last_float is not None:
                    center = self.vision.ring_center(current.png, last_float)
                    if center is not None:
                        (evidence/f'{operation}_hooked.png').write_bytes(current.png)
            if center is not None:
                found = self.vision.ring_center(current.png, center)
                missing_ring = 0 if found is not None else missing_ring+1
                if found and np.linalg.norm(np.subtract(found, center)) < 35*scale:
                    center = found
                if end is None and lost_frames < 3:
                    lost_frames += 1
                    (evidence/f'{operation}_line_lost_{lost_frames}.png').write_bytes(current.png)
            tracking.append({'elapsed': round(elapsed, 3), 'contact': tuple(map(int, point)),
                             'line_end': end, 'center': center, 'missing': missing,
                             'missing_ring': missing_ring})
            # Once the line disappears, lifting cannot replay a cast. The stock
            # check, never this animation, determines whether a fish was caught.
            if elapsed >= 40 or (center is not None and missing >= 3 and missing_ring >= 3) or (last_float is None and elapsed > 8):
                end_reason = 'line_and_ring_disappeared' if missing >= 3 and missing_ring >= 3 else 'observation_limit'
                return None
            if center is not None:
                if end is None:
                    return point
                delta = (np.array(center)-end)*.9
                target = np.array(point)+np.clip(delta, -100*scale, 100*scale)
            else:
                # Offsets describe a drag from the freshly recognized lure,
                # not a stored pond position. Small moves let the cast settle.
                dx, dy = ((80,195), (130,295), (180,395), (130,495))[min(int(elapsed/4),3)]
                goal = np.array(checked.tool)+np.array([dx,dy])*scale
                target = np.array(point)+np.clip(goal-point, -50*scale, 50*scale)
            return (int(np.clip(target[0], current.width*.20, current.width*.78)),
                    int(np.clip(target[1], current.height*.24, current.height*.89)))

        self.client.drag_feedback(checked.tool, update, width=fresh.width, height=fresh.height,
                                  max_seconds=46, cancel_event=self.cancel_event)
        self.cancel_event.wait(.6)
        after = self._frame()
        (evidence/f'{operation}_after.png').write_bytes(after.png)
        self._record(key, after_capture=f'{operation}_after.png', observations=observations,
                     end_reason=end_reason)
        (evidence/f'{operation}_tracking.json').write_text(json.dumps(tracking, indent=2), encoding='utf-8')
        self._return_home()
        return ResourceResult('waiting', 'Fishing attempted; returning to verify the order inventory.',
                              {'item': key, 'pending_harvest': True, 'wait_seconds': 1})

"""Work the connected field beside the roadside shop."""
from __future__ import annotations

import math
import time
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np

from hayday.adb import Screenshot
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker
from hayday.resource_vision import VisualTarget, _decode
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision


class WheatFields:
    def __init__(self, runner):
        self.run = runner
        state_path = runner.device_root/'fields.json'
        try:
            self.worker = WheatCropWorker(runner.client, runner.capture, runner.cancel_event,
                runner.publish, state_path,
                fast_capture=lambda: runner.capture(fast=True),
                vision=WheatFarmingVision(cancel=runner.cancel_event.is_set))
        except (OSError, ValueError) as exc:
            quarantine = getattr(runner, 'quarantine_file', None)
            if not callable(quarantine):
                raise
            quarantine(state_path, f'Saved wheat field state could not be read: {exc}')
            self.worker = WheatCropWorker(runner.client, runner.capture, runner.cancel_event,
                runner.publish, state_path,
                fast_capture=lambda: runner.capture(fast=True),
                vision=WheatFarmingVision(cancel=runner.cancel_event.is_set))
        from hayday.wheating_growth import WheatGrowthTimer
        if getattr(runner, '_wheat_growth', None) is None:
            runner._wheat_growth = WheatGrowthTimer(runner)
        self.worker.growth_timer = runner._wheat_growth
        # The operator starts each session with the entire field mature. A
        # saved deadline must not send a newly prepared field back to the shop.
        self.next_harvest = 0.
        self.has_fields = False
        self._visited = []
        self._reference = None
        self._occluded = 0
        self._prepared = False
        self.last_stock = None
        self._known_before = None
        self._known_points = []
        self._retry_harvest_key = None
        self._load_layout()
        from hayday.wheating_field_group import WheatFieldGroup
        self._group = WheatFieldGroup(self)

    def _load_layout(self):
        """Reuse planting evidence; live translation and wheat still gate input."""
        layout = getattr(self.run, 'state', {}).get('field_layout')
        if not isinstance(layout, dict) or layout.get('version') != 2:
            return
        proof, points = layout.get('before'), layout.get('points')
        if not isinstance(proof, dict) or not isinstance(points, list) or not 1 <= len(points) <= 98:
            return
        width, height = proof.get('width'), proof.get('height')
        if any(type(n) is not int or not 2 <= n <= 16384 for n in (width, height)):
            return
        if (any(not isinstance(p, list) or len(p) != 2 or any(type(n) is not int for n in p)
                or not 0 <= p[0] < width or not 0 <= p[1] < height for p in points)
                or len({tuple(p) for p in points}) != len(points)):
            return
        # _saved_frame verifies a local evidence filename, dimensions and SHA.
        # It does not establish this session's device resolution.
        size = self.worker._size
        self.worker._size = (width, height)
        try:
            before = self.worker._saved_frame(proof)
        finally:
            self.worker._size = size
        if before is not None:
            self._known_before = before
            self._known_points = [tuple(p) for p in points]

    def _remember_layout(self, entry):
        before = self.worker.plant_before
        points = self.worker.planted_points
        if not isinstance(before, Screenshot) or not points or not entry.get('replanted'):
            return
        # Cache only coordinates actually verified by this planting. Projecting
        # older sections into another picker can accumulate phantom plot rows.
        # Independent live wheat/soil searches discover anything outside this
        # cache, so an incomplete cache never defines the field boundary.
        # Record only a completed planting. Existing images can be shared with
        # its diagnostic proof, so remembering a field adds no new screenshot.
        proof = entry.get('plant_before')
        if (not isinstance(proof, dict)
                or self.worker._saved_frame(proof) != before):
            proof = self.worker._evidence(before, uuid.uuid4().hex, 'field_layout')
        self.run.state['field_layout'] = {'version': 2, 'before': proof,
            'points': [list(map(int, point)) for point in points]}
        self._known_before, self._known_points = before, list(points)
        if group := getattr(self, '_group', None):
            group.planted()

    def _relocate(self, frame):
        if self._reference is not None and self._visited:
            origin = (self._reference.width//2, self._reference.height//2)
            moved = self.worker._translated_plot(self._reference, frame, origin, require_visible=False)
            if moved is None:
                self.run.block('Previously visited wheat fields could not be aligned after camera movement.')
            dx, dy = np.subtract(moved, origin)
            self._visited = [((x+dx, y+dy), radius) for (x, y), radius in self._visited]
        self._reference = frame

    def _clear(self):
        frame = self.run.capture()
        if not self.run.vision.farm(frame):
            self.run.block('The farm is obscured; close its dialog before starting Wheating.')
        return self.worker._close(frame)

    def ready_to_harvest(self):
        frame = self.run.capture()
        if not self.run.vision.farm(frame):
            return False
        # This schedules a field pass only. Fresh selected crop controls still
        # have to confirm the crop before any harvest gesture is sent.
        ready = bool(self._known_before is not None and self._known_points and self._tracked_wheat(frame))
        # A growing saved section must not hide ripe wheat or bare plots found
        # elsewhere. Crop selection still verifies every resulting action.
        ready = ready or bool(self.run.vision.plots(frame, 'ripe') or self.run.vision.plots(frame, 'empty'))
        group = getattr(self, '_group', None)
        if group and (ready or group.active) and group.plan(frame) == 'wait':
            return False
        return ready

    def has_live_work(self, frame):
        return (self.run.vision.farm(frame) and not CameraNavigator._modal_visible(frame)
                and bool(self.run.vision.plots(frame, 'ripe') or self.run.vision.plots(frame, 'empty')))

    def _mature_field_confirmed(self, frame):
        """Return whether the operator's fresh-start field invariant is visible.

        A persisted crop gesture can outlive the game process.  When the farm is
        visibly full of mature wheat, that live observation is safer than trying
        to replay yesterday's touch.  Known planting layouts get a stronger
        all-points check; a new layout still needs a dense, bounded wheat field.
        """
        if not self.run.vision.farm(frame) or CameraNavigator._modal_visible(frame):
            return False
        if self._known_before is not None and self._known_points:
            tracked = self._tracked_wheat(frame)
            required = max(3, math.ceil(len(self._known_points)*.80))
            if len(tracked) >= required:
                return True
        ripe = tuple(self.run.vision.plots(frame, 'ripe'))
        if not ripe:
            return False
        bounds = getattr(self.run.vision, 'field_bounds', lambda _frame: None)(frame)
        if bounds is not None:
            area = bounds.width * bounds.height
            if area >= frame.width * frame.height * .025:
                return True
        # A connected crop candidate is only accepted when it occupies a
        # substantial part of the farm viewport; isolated yellow decorations do
        # not clear durable crop work.
        return any(target.width * target.height >= frame.width * frame.height * .015
                   for target in ripe)

    def reconcile_stale_state(self, frame):
        """Quarantine stale crop work when a fresh mature field is confirmed.

        The archive is diagnostic and reversible.  Live state is cleared only
        after the current farm has passed the mature-field invariant, so an
        uncertain touch is never silently replayed or counted twice.
        """
        pending = {
            key: entry for key, entry in self.worker.state['items'].items()
            if entry.get('stage') in self.worker._PENDING
        }
        recovery = self.run.state.get('silo_recovery')
        shop_pending = self.run.state.get('pending')
        stale_shop_pending = (shop_pending if isinstance(shop_pending, dict)
                              and shop_pending.get('operation') in {'list', 'advertise'} else None)
        if not pending and recovery is None and stale_shop_pending is None:
            return False
        if recovery is not None:
            # Remaining ripe wheat after a partial harvest must not erase the
            # storage-release checkpoint or the layout of already cleared soil.
            return False
        if not self._mature_field_confirmed(frame):
            return False

        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')
        archive = {
            'version': 1,
            'reason': 'Fresh mature field superseded unresolved crop work.',
            'captured_at': frame.captured_at,
            'crop_items': pending,
            'silo_recovery': recovery,
            'shop_intent': stale_shop_pending,
        }
        self.run.save_json(self.run.device_root/'recovery_archive'/f'{stamp}_stale_crop.json', archive)

        items = {
            key: entry for key, entry in self.worker.state['items'].items()
            if entry.get('stage') not in self.worker._PENDING
        }
        self.worker.state = {**self.worker.state, 'items': items}
        self.run.save_json(self.worker.state_path, self.worker.state)
        self.run.state.pop('silo_recovery', None)
        if stale_shop_pending is not None:
            self.run.state.pop('pending', None)
        self.run.state['field_ready_at'] = 0.
        self.run.state['wheat_empty'] = False
        self.run.state['seed_reserve'] = max(1, int(self.run.state.get('seed_reserve', 1)))
        # Keep a verified full layout when farm landmarks still align. A partial
        # mature patch does not establish a smaller field after interruption.
        keep_layout = (self._known_before is not None and self._known_points
            and self.worker._translated_plot(self._known_before, frame,
                self._known_points[0], require_visible=False) is not None)
        if not keep_layout:
            self.run.state.pop('field_layout', None)
        self.run.persist()
        if not keep_layout:
            self._known_before, self._known_points = None, []
        self._retry_harvest_key = None
        self.next_harvest = 0.
        self.has_fields = False
        self.run.publish('Wheating: fresh mature wheat confirmed. Archived stale crop work and resumed a new cycle.')
        return True

    def work_view(self, frame, *, soil_only=False):
        planted = 0
        frame = self._center_field(frame)
        shop = self.run.vision.shop_building(frame)
        if shop:
            self.run._shop_anchor = frame, shop
        # Each iteration rediscovers remaining soil/wheat after the last gesture.
        # Growth happens while subsequent plots and shop sales are serviced.
        for _ in range(200):
            self.run.check()
            self._relocate(frame)
            group = getattr(self, '_group', None)
            decision = group.plan(frame) if group else None
            if decision == 'wait':
                return planted, frame
            if decision == 'repair':
                soil_only = True
            # Harvest ripe wheat first to supply seeds for every remaining plot.
            # A single weak/swaying tile should not discard the complete
            # planting grid.  When at least 80% of the saved points still show
            # wheat, use the full translated grid for the gesture while using
            # only confirmed points to select the crop controls.
            known = self._known_wheat_points(frame, minimum=.80) if not soil_only else None
            if group and group.full_route:
                known = None
            if known and getattr(self.run.vision, 'wheat_outside', lambda *a: False)(frame, known[0]):
                known = None
            tracked = known[0] if known else []
            observed = known[1] if known else []
            candidates = [('ripe', VisualTarget(x-15, y-15, 30, 30, 1.))
                          for x, y in observed[:1]]
            if not candidates and not soil_only:
                candidates = [('ripe', t) for t in self.run.vision.plots(frame, 'ripe')]
            candidates = [(kind, target) for kind, target in candidates
                          if not any(np.linalg.norm(np.subtract(target.center, point)) < radius
                                     for point, radius in self._visited)]
            if not candidates:
                if self._retry_harvest_key is not None:
                    return planted, frame
                candidates = [('empty', t) for t in self.run.vision.plots(frame, 'empty')
                              if not any(np.linalg.norm(np.subtract(t.center, point)) < radius
                                         for point, radius in self._visited)]
            if not candidates:
                return planted, frame
            kind, target = candidates[0]
            self.has_fields = True
            fresh = self.run.capture(fast=True) if kind == 'ripe' else self.run.capture()
            self._relocate(fresh)
            # Swaying stalks can join/split foliage components between frames.
            # Any freshly recognized, unvisited wheat patch is valid work; its
            # selected tool and outline still gate the actual harvest gesture.
            known = self._known_wheat_points(fresh, minimum=.80) if kind == 'ripe' else None
            if group and group.full_route:
                known = None
            if known and getattr(self.run.vision, 'wheat_outside', lambda *a: False)(fresh, known[0]):
                known = None
            tracked = known[0] if known else []
            observed = known[1] if known else []
            targets = ([VisualTarget(observed[0][0]-15, observed[0][1]-15, 30, 30, 1.)] if observed
                       else self.run.vision.plots(fresh, kind))
            checked = next((t for t in sorted(targets, key=lambda t: -t.y)
                            if not any(np.linalg.norm(np.subtract(t.center, point)) < radius
                                       for point, radius in self._visited)), None)
            if checked is None:
                frame = fresh
                continue
            self.worker.harvest_plan = None
            self.worker.harvest_plot_centers = False
            self.worker.harvest_route_mode = None
            self.worker.harvest_observed_points = 0
            self.worker.harvest_expected_points = 0
            self.worker.harvest_sample_spacing = None
            self.worker.harvest_waypoint_hold = None
            self.worker.soil_frame = fresh
            if kind == 'ripe':
                sweep = tracked or self.run.vision.harvest_sweep(fresh)
                if not sweep:
                    self.run.block('The whole wheat field could not be traced for a single harvest sweep.')
                self.worker.harvest_plan = fresh, sweep
                self.worker.harvest_plot_centers = bool(tracked)
                self.worker.harvest_route_mode = (
                    'known_grid' if known and len(observed) == len(tracked)
                    else 'known_grid_relaxed' if known else 'foliage')
                self.worker.harvest_observed_points = len(observed)
                self.worker.harvest_expected_points = len(tracked)
                self.worker.harvest_sample_spacing = None if tracked else max(
                    1, round(16*fresh.height/1080))
                self.worker.harvest_waypoint_hold = None if tracked else 16
            selected, expected, control, ready = self._select(fresh, checked, kind)
            if not ready:
                if self.run.diagnostics:
                    (self.run.diagnostics/'obscured_selection.png').write_bytes(selected.png)
                    self.run.save_json(self.run.diagnostics/'obscured_selection.json',
                        {'kind': kind, 'candidate': list(checked.box),
                         'expected': list(expected) if expected is not None else None,
                         'control': list(control.box) if control is not None else None})
                # A neighboring wheat sprite can hide an empty tile's outline.
                # No crop gesture has been sent, so inspect other candidates and
                # revisit this location next pass instead of stopping the shop.
                after = self._clear()
                self._relocate(after)
                point = self.worker._translated_plot(fresh, after, checked.center)
                if point is None:
                    self.run.block('The obscured field could not be relocated after selection.')
                self._visited.append((point, max(30, checked.width*.5)))
                self._occluded += 1
                frame = after
                continue
            self.worker._size = selected.width, selected.height
            if self._retry_harvest_key is not None:
                from hayday.wheating_harvest_resume import retire_reselected
                retire_reselected(self, self._retry_harvest_key, selected)
            key = 'wheat_'+uuid.uuid4().hex
            result = self.worker.work_if_recognized(selected, self.run.vision.wheat_icon, key)
            entry = self.worker.state['items'].get(key, {})
            if result and result.status == 'changed' and entry.get('stage') not in self.worker._PENDING:
                frame = self._clear()
                continue
            if result and result.status == 'waiting' and not entry.get('replanted') and entry.get('stage') not in self.worker._PENDING:
                self.worker.state['items'].pop(key, None)
                self.run.save_json(self.worker.state_path, self.worker.state)
                after = self._clear()
                self._relocate(after)
                point = self.worker._translated_plot(fresh, after, checked.center)
                if point:
                    self._visited.append((point, max(35, checked.width*.65)))
                frame = after
                continue
            if not result or not entry.get('replanted') or entry.get('stage') != 'growing':
                self.run.block(result.message if result else 'The selected wheat controls are unsupported.')
            gain = max(1, len(self.worker.planted_points))
            planted += gain
            self.run.planted += gain
            self.last_stock = self.worker.remaining_seed_stock
            self.next_harvest = min(self.next_harvest, self.worker.growth_ready_at or time.monotonic()+self.worker.growth_duration)
            self.run.state['field_ready_at'] = time.time()+max(0., self.next_harvest-time.monotonic())
            self.run.state['wheat_empty'] = self.last_stock == 0
            self._remember_layout(entry)
            self.run.persist()
            # Only completed work is retired; uncertain gestures survive restart.
            self.worker.state['items'].pop(key)
            self.run.save_json(self.worker.state_path, self.worker.state)
            self.run.publish(f'Wheating: planted {gain}/{self.worker.field_size} plots in one sweep. Checking wheat sales while it grows.')
            frame = self._clear()
            self._relocate(frame)
            if not self.run.vision.plots(frame, 'empty'):
                # No remaining soil needs a second section. Avoid aligning the
                # dismissed picker just to create an unused visited-point list.
                return planted, frame
            if self.worker.plant_before and self.worker.planted_points:
                origin = self.worker.planted_points[0]
                moved = self.worker._translated_plot(self.worker.plant_before, frame, origin)
                if moved is None:
                    from hayday.wheating_restart_camera import farm_transform
                    matrix = farm_transform(self.worker.plant_before, frame)
                    if matrix is not None and abs(np.hypot(matrix[0, 0], matrix[1, 0])-1) < .025:
                        moved = tuple(int(round(n)) for n in matrix @ np.array([*origin, 1.]))
                if moved is None:
                    self.run.block('The planted wheat sweep could not be relocated after closing the seed menu.')
                shift = np.subtract(moved, origin)
                # A foliage bounding box can cover untouched neighboring soil.
                # Mark only the individual tiles that actually confirmed growth.
                self._visited.extend((tuple(np.add(point, shift)), 20*frame.height/1080)
                                     for point in self.worker.planted_points)
            # Finish disconnected bare sections left by an interrupted harvest.
            # A fully planted field exits above using this same captured frame.
            soil_only = True
            continue
        self.run.block('Field pass exceeded its observation limit; coverage is incomplete.')

    def _select(self, fresh, checked, kind):
        if kind == 'ripe':
            from hayday.wheating_selection import select_wheat
            return select_wheat(self.run, self.worker, fresh, checked)
        # Soil selection retains its existing timing and outline requirements.
        self.run.tap(checked.center, fresh)
        prior = None
        for _ in range(6):
            self.run.wait(.25)
            selected = self.run.capture()
            expected = self.worker._translated_plot(fresh, selected, checked.center)
            control = self.worker.vision.empty_plot(selected.png)
            associated = (expected is not None and control is not None
                          and np.linalg.norm(np.subtract(expected, control.center)) < max(80, control.width, control.height))
            if associated and FarmingWorker._same_target(prior, control, selected):
                return selected, expected, control, True
            prior = control if associated else None
        return selected, expected, control, False

    def _known_wheat_points(self, frame, *, minimum=1.0):
        if self._known_before is None or not self._known_points:
            return None
        if (self._known_before.width, self._known_before.height) != (frame.width, frame.height):
            return None
        origin = self._known_points[0]
        moved = self.worker._translated_plot(self._known_before, frame, origin, require_visible=False)
        if moved is None:
            return None
        shift = np.subtract(moved, origin)
        points = [tuple(map(int, np.add(point, shift)-(0, round(16*frame.height/1080))))
                  for point in self._known_points]
        hsv = cv2.cvtColor(_decode(frame.png), cv2.COLOR_BGR2HSV)
        observed = []
        for x, y in points:
            if not (frame.width*.12 < x < frame.width*.88 and frame.height*.20 < y < frame.height*.83):
                return None
            patch = hsv[y-8:y+9, x-12:x+13]
            wheat = cv2.inRange(patch, (18, 140, 160), (31, 255, 255))
            if (wheat > 0).mean() >= .4:
                observed.append((x, y))
        required = max(3, math.ceil(len(points)*float(minimum)))
        if len(observed) < required:
            return None
        return points, observed

    def _tracked_wheat(self, frame):
        known = self._known_wheat_points(frame)
        return known[0] if known else []

    def _center_field(self, frame):
        for _ in range(3):
            bounds = self.run.vision.field_bounds(frame)
            if bounds is None and self._known_before is not None and self._known_points:
                from hayday.wheating_restart_camera import farm_transform
                matrix = farm_transform(self._known_before, frame)
                if matrix is not None:
                    points = np.c_[np.asarray(self._known_points), np.ones(len(self._known_points))] @ matrix.T
                    left, top = points.min(axis=0)-35*frame.height/1080
                    right, bottom = points.max(axis=0)+35*frame.height/1080
                    bounds = VisualTarget(int(left), int(top), int(right-left), int(bottom-top), 1.)
            if bounds is None or (bounds.x > frame.width*.10 and bounds.x+bounds.width < frame.width*.89
                                  and bounds.y > frame.height*.18 and bounds.y+bounds.height < frame.height*.83):
                return frame
            dx = float(np.clip(.56-bounds.center[0]/frame.width, -.25, .25))
            dy = float(np.clip(.50-bounds.center[1]/frame.height, -.28, .28))
            drag = CameraNavigator._grass_start(frame, dx, dy)
            if drag is None:
                return frame
            self.run.check()
            self.run.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=300)
            prior = None
            for _ in range(6):
                self.run.wait(.25)
                frame = self.run.capture()
                if not self.run.vision.farm(frame):
                    self.run.block('The wheat field disappeared during camera positioning.')
                current = CameraNavigator._view(frame)
                if prior is not None and CameraNavigator._same_view(prior, current):
                    break
                prior = current
        return frame

    def pass_all(self):
        self.last_stock = None
        pending = [key for key, entry in self.worker.state['items'].items() if entry.get('stage') in self.worker._PENDING]
        if len(pending) > 1:
            self.run.block('Multiple earlier wheat gestures are unresolved; inspect their saved evidence before resuming.')
        if pending:
            from hayday.wheating_harvest_resume import can_reselect
            current = self.run.capture()
            if can_reselect(self, pending[0], current):
                self._retry_harvest_key = pending[0]
                # A manually restored farm may have a different field layout.
                # Trace it afresh; retirement waits for normal crop/tool checks.
                if not (self._known_before is not None and len(self._known_points) >= 3
                        and self.worker._translated_plot(self._known_before, current,
                            self._known_points[0], require_visible=False) is not None):
                    self._known_before, self._known_points = None, []
                pending = []
        for key in pending:
            frame = self.run.capture()
            if self.worker.state['items'][key].get('stage') == 'harvest_attempted':
                frame = self._center_field(frame)
            result = self.worker.work_if_recognized(frame, self.run.vision.wheat_icon, key)
            if not result or not self.worker.state['items'][key].get('replanted'):
                detail = result.message if result else 'No matching field controls were found.'
                self.run.block(f'An earlier wheat gesture is unresolved. {detail}')
            gain = 0 if self.worker.state['items'][key].get('externally_replanted') else max(1, len(self.worker.planted_points))
            self.run.planted += gain
            self.has_fields = True
            self.last_stock = self.worker.remaining_seed_stock
            self.next_harvest = self.worker.growth_ready_at or time.monotonic()+self.worker.growth_duration
            self.run.state['field_ready_at'] = time.time()+max(0., self.next_harvest-time.monotonic())
            self.run.state['wheat_empty'] = self.last_stock == 0
            self._remember_layout(self.worker.state['items'][key])
            self.run.persist()
            self.worker.state['items'].pop(key)
            self.run.save_json(self.worker.state_path, self.worker.state)
            if self.worker.soil_frame is not None:
                shop = self.run.vision.shop_building(self.worker.soil_frame)
                if shop:
                    self.run._shop_anchor = self.worker.soil_frame, shop
            self._prepared = True
            cleared = self._clear()
            if self.worker.resumed_mature:
                self.run.publish('Wheating: verified the interrupted planting. Its wheat is ripe; starting the next harvest without repeating planting.')
                return self.pass_all()
            self.run.publish(f'Wheating: planted {gain}/{self.worker.field_size} plots in one sweep. Checking wheat sales while it grows.')
            # A resumed planting can cover only one section of the interrupted
            # harvest. Check the remaining soil before accepting cycle coverage.
            if not getattr(self.run.vision, 'plots', lambda *a: ())(cleared, 'empty'):
                return gain
            self._visited, self._reference = [], cleared
            additional, _ = self.work_view(cleared, soil_only=True)
            return gain+additional
        frame = self._clear()
        self._prepared = True
        self._visited, self._reference = [], frame
        self._occluded = 0
        self.next_harvest = math.inf
        count, frame = self.work_view(frame)
        if count or (getattr(self, '_group', None) and self._group.waiting):
            return count
        # There is one attached field beside the shop. A swaying crop, growing
        # field, or obscured selection must not launch a farm-wide camera scan.
        for _ in range(2):
            self.run.wait(.3)
            frame = self._clear()
            gain, frame = self.work_view(frame)
            count += gain
            if gain or (getattr(self, '_group', None) and self._group.waiting):
                return count
        shop = self.run.vision.shop_building(frame)
        if not shop:
            self.run.block('Keep the connected wheat field and roadside shop visible before starting Wheating.')
        self.run._shop_anchor = frame, shop
        self.next_harvest = time.monotonic()+15
        self.run.publish('Wheating: waiting for the field beside the shop to be ready. Checking wheat sales.')
        return count

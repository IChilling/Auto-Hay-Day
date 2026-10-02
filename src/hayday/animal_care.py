"""Durable feeding passes and Feed Mill replenishment between truck orders."""
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from hayday.adb import Screenshot
from hayday.animals import SheepVision
from hayday.camera import CameraNavigator, FarmScan
from hayday.cows import CowVision
from hayday.farming import FarmingWorker
from hayday.herd_vision import ANIMALS, HerdVision
from hayday.panel import PanelVerifier
from hayday.resource_vision import _decode, _png
from hayday.resources import ResourceChanged, ResourceResult, _save_json
from hayday.work_scheduling import utc_timestamp


def _load(path, serial):
    state = json.loads(path.read_text('utf-8')) if path.exists() else {
        'version': 2, 'serial': serial, 'jobs': {}, 'pending': None}
    if state.get('serial') != serial or state.get('version') not in {1, 2}:
        raise ResourceChanged('Saved animal care belongs to a different device or has an invalid version.')
    if state['version'] == 1:
        # Retire the old farm-wide requests, preserving any uncertain gesture
        # and production history. Only an observed pen can create a new job.
        state = {'version': 2, 'serial': serial, 'jobs': {},
                 'pending': state.get('pending'), 'legacy': state}
    if not isinstance(state.get('jobs'), dict) or any(
            not isinstance(job, dict) or job.get('animal') not in ANIMALS
            or job.get('stage') not in {'awaiting_harvest', 'needed', 'waiting_feed', 'complete'}
            for job in state['jobs'].values()):
        raise ResourceChanged('Saved animal-feeding tasks are invalid.')
    return state


def request_feeding(path, serial, *, animal, frame, polygon, item_key,
                    harvest_operation=None, pen_index=0):
    """Request care for one observed pen, never for all pens of a species."""
    if animal not in ANIMALS or len(polygon) < 4:
        raise ResourceChanged('A feeding request requires an identified animal pen.')
    path = Path(path)
    state = _load(path, serial)
    operation = harvest_operation or uuid.uuid4().hex
    if pen_index:
        operation += '_pen_'+str(pen_index)
    if operation in state['jobs']:
        return operation
    center = tuple(map(float, np.mean(polygon, axis=0)))
    evidence = path.parent/'animal_care_evidence'
    evidence.mkdir(exist_ok=True)
    # Repeated ingredient checks must not create duplicate feeding jobs.
    for key, job in state['jobs'].items():
        if job['animal'] != animal or job['stage'] == 'complete':
            continue
        if Path(job['file']).name != job['file']:
            raise ResourceChanged('Invalid animal-care evidence path.')
        png = (evidence/job['file']).read_bytes()
        if hashlib.sha256(png).hexdigest() != job['sha256']:
            raise ResourceChanged('Saved animal-care evidence changed.')
        before = Screenshot(png, job['width'], job['height'], 'saved-pen')
        projected = FarmingWorker._translated_plot(before, frame, job['center'], require_visible=False, allow_scale=True)
        if projected is not None and np.linalg.norm(np.subtract(projected, center)) < 35:
            if job['stage'] == 'awaiting_harvest' and harvest_operation:
                # A stale observation may have cancelled the preceding input
                # after its care intent was saved. Follow the new harvest's
                # receipt instead of leaving this pen waiting on that old ID.
                job['harvest_operation'] = harvest_operation
                _save_json(path, state)
            return key
    name = uuid.uuid4().hex+'.png'
    (evidence/name).write_bytes(frame.png)
    state['jobs'][operation] = {
        'animal': animal, 'item_key': item_key, 'harvest_operation': harvest_operation,
        'stage': 'awaiting_harvest' if harvest_operation else 'needed', 'next_due': 0,
        'file': name, 'sha256': hashlib.sha256(frame.png).hexdigest(),
        'width': frame.width, 'height': frame.height, 'center': center,
        'polygon': polygon}
    _save_json(path, state)
    return operation


class AnimalCareWorker:
    def __init__(self, owner):
        self.owner = owner
        self.path = owner.state_path.with_name('animal_care.json')
        self.vision = None
        self.state = None
        self.home_recovery = None
        self._evidence = self.path.parent/'animal_care_evidence'

    def _save(self):
        _save_json(self.path, self.state)

    def _capture(self):
        return self.owner._capture()

    def _saved_pen(self, job):
        name = job['file']
        if Path(name).name != name:
            raise ResourceChanged('Invalid animal-care evidence path.')
        png = (self._evidence/name).read_bytes()
        if hashlib.sha256(png).hexdigest() != job['sha256']:
            raise ResourceChanged('Saved animal-care evidence changed.')
        return Screenshot(png, job['width'], job['height'], 'saved-pen')

    def _project(self, job, frame):
        projected = FarmingWorker._translated_plot(self._saved_pen(job), frame,
            job['center'], require_visible=False, allow_scale=True)
        if projected is not None or not job.get('navigation_view'):
            return projected
        view = job['navigation_view']
        return FarmingWorker._translated_plot(self._saved_pen(view), frame,
            view['center'], require_visible=False, allow_scale=True)

    def remember_view(self, frame):
        if self.path.exists():
            # Resource collection may have just created another feeding job.
            self.state = _load(self.path, self.owner.serial)
            self._remember_view(frame)

    def _remember_view(self, frame):
        """Bridge distant views through a frame matched to the original pen.

        Each bridge starts from the original evidence, never another bridge,
        so repeated camera searches cannot accumulate positional drift. This
        only guides navigation; feeding still needs a freshly verified pen.
        """
        if CameraNavigator._modal_visible(frame):
            return
        projections = []
        for job in self.state['jobs'].values():
            if job['stage'] == 'complete':
                continue
            projected = FarmingWorker._translated_plot(self._saved_pen(job), frame,
                job['center'], require_visible=False, allow_scale=True)
            if projected is not None:
                projections.append((job, projected))
        if not projections:
            return
        digest = hashlib.sha256(frame.png).hexdigest()
        name = 'navigation_'+digest+'.png'
        self._evidence.mkdir(exist_ok=True)
        path = self._evidence/name
        if not path.exists():
            path.write_bytes(frame.png)
        previous = {job.get('navigation_view', {}).get('file') for job, _ in projections}
        for job, center in projections:
            job['navigation_view'] = {'file': name, 'sha256': digest,
                'width': frame.width, 'height': frame.height, 'center': list(center),
                'captured_at': frame.captured_at}
        self._save()
        retained = {job.get('navigation_view', {}).get('file') for job in self.state['jobs'].values()}
        # Only replace expendable navigation views; original pen and action
        # evidence remain untouched, including jobs with uncertain gestures.
        for old in previous-retained-{None}:
            if Path(old).name == old and old.startswith('navigation_'):
                (self._evidence/old).unlink(missing_ok=True)

    def _target_pens(self, job, frame):
        projected = self._project(job, frame)
        if projected is None:
            return ()
        return tuple(pen for pen in self.vision.pens(frame.png, job['animal'], self.owner.cancel_event.is_set)
                     if np.linalg.norm(np.subtract(projected, np.mean(pen[2], axis=0))) < 35)

    def _close_board(self, frame):
        verifier = PanelVerifier()
        first = verifier.verify(frame.png, cancel=self.owner.cancel_event.is_set)
        if not first.verified:
            return frame
        fresh = self._capture()
        second = verifier.verify(fresh.png, cancel=self.owner.cancel_event.is_set)
        controls = [feature for feature in second.features if feature.name == 'close_button']
        if not second.verified or len(controls) != 1 or fresh.captured_at == frame.captured_at:
            raise ResourceChanged('The order panel changed before animal care could start.')
        x, y, w, h = controls[0].bounds
        frame = self.owner._tap((x+w//2, y+h//2), fresh)
        clear, previous = 0, None
        for _ in range(6):
            self.owner._check()
            visible = verifier.verify(frame.png, cancel=self.owner.cancel_event.is_set).verified
            if not visible and not CameraNavigator._modal_visible(frame) and frame.captured_at != previous:
                clear += 1
                if clear >= 2:
                    return frame
            else:
                clear = 0
            previous = frame.captured_at
            self.owner._wait(.2)
            frame = self._capture()
        raise ResourceChanged('The order panel did not finish closing before animal care; no pen input was sent.')

    def _dismiss(self, frame, menu=None):
        if menu is not None:
            left = max(0, menu.tool.x-round(400*frame.height/1080))
            ground = CameraNavigator._grass_start(frame, 0, 0,
                exclude_regions=((left, 0, frame.width-left, frame.height),))
        else:
            ground = CameraNavigator._menu_ground(frame, self.owner._observe(frame))
        if ground is None:
            raise ResourceChanged('No clear farm ground was verified outside the animal controls.')
        frame = self.owner._tap(ground[:2], frame)
        if menu is not None and self.vision.menu(frame.png, menu.animal, self.owner.cancel_event.is_set):
            raise ResourceChanged('The animal menu remained open; no camera motion was sent.')
        return frame

    def _stock_receipts(self, job, first, second, menu):
        key = job.get('feed_key') or self.owner._known_icon(menu.icon)
        if key:
            key = self.owner._resolve_key(key)
            job['feed_key'] = key
            self._save()
            for frame in (first, second):
                self.owner._acknowledge_inventory(key, 'fulfilled' if menu.available else 'missing',
                                                  menu.available, 1, frame.captured_at)

    def _supply(self, job, frame, menu):
        owner = self.owner
        animal = job['animal']
        entry = job
        # Store the currently observed original feed artwork, not a farm point.
        x, y, w, h = menu.tool.box
        icon = _png(_decode(frame.png)[y:y+h, x:x+w])
        key = owner._learn(icon, source=frame, box=menu.tool.box)
        fresh = self._capture()
        checked = self.vision.menu(fresh.png, animal, owner.cancel_event.is_set)
        if (checked is None or not checked.enabled or checked.available != menu.available
                or fresh.captured_at == frame.captured_at
                or not owner.fruits._same(menu.tool, checked.tool, fresh)):
            raise ResourceChanged('The feed bag changed before its production link could be opened.')
        popup_frame = owner._tap(checked.tool.center, fresh)
        scene = owner._observe(popup_frame)
        prompts = [popup for popup in scene.popups if not popup.rows and len(popup.navigation) == 1]
        if len(prompts) != 1:
            raise ResourceChanged('The feed bag did not expose a verified Feed Mill link.')
        popup = prompts[0]
        located = owner._follow_location(popup)
        key = owner._bind_title(key, popup.title_png)
        entry['feed_key'] = key
        self._save()
        if not owner._state['items'][key].get('pending_batch'):
            owner._record(key, 'inventory_baseline', inventory_before={
                'status': 'missing', 'available': menu.available, 'required': 1,
                'captured_at': frame.captured_at})
        owner.progress(f'Checking {animal} feed at its verified Feed Mill source.')
        result = owner._at_source(located, menu.icon, key, popup.title_png, (), 0)
        collected = menu.available and not owner._state['items'][key].get('pending_batch')
        entry.update(stage='needed' if collected else 'waiting_feed',
                     next_due=0 if collected else utc_timestamp()+60)
        self._save()
        return ResourceResult(result.status, result.message, {**result.details, 'animal_care': animal})

    def _receipt(self, job, frame, current):
        pending = self.state['pending']
        before = pending['stock_before']
        after = current.available
        if after is None or after > before:
            return None
        outcome = 'fed' if after < before else 'already_fed' if not current.enabled else 'no_effect'
        operation = pending['operation']
        if Path(operation).name != operation:
            raise ResourceChanged('Invalid animal-feed receipt path.')
        (self._evidence/(operation+'_after.png')).write_bytes(frame.png)
        job['last_feed'] = {'before': before, 'after': after,
                            'operation': operation, 'outcome': outcome}
        # A partial stock decrease does not mean every animal in this pen ate.
        job.update(stage='needed' if current.enabled else 'complete',
                   next_due=utc_timestamp()+60 if current.enabled else 0)
        self.state['pending'] = None
        self._save()
        self._dismiss(frame, current)
        return ResourceResult('waiting',
            f"{job['animal'].capitalize()} feed stock confirmed: {before} to {after}. "
            + ('This pen still needs feeding.' if current.enabled else 'This pen is fed.'),
            {'animal_care': job['animal'], 'feed_confirmed': outcome == 'fed'})

    def _feed(self, job_id, job, before, pen, frame, menu):
        owner = self.owner
        animal = job['animal']
        fresh = self._capture()
        started = time.monotonic()
        checked = self.vision.menu(fresh.png, animal, owner.cancel_event.is_set)
        polygon = self.vision.confirm_pen(before, fresh, animal, pen, owner.cancel_event.is_set)
        if (checked is None or checked.available != menu.available or not checked.enabled
                or not polygon or not owner.fruits._same(menu.tool, checked.tool, fresh)
                or fresh.captured_at == frame.captured_at or time.monotonic()-started > 4):
            raise ResourceChanged('The feed tool, stock or fenced pen changed before feeding.')
        self._stock_receipts(job, frame, fresh, checked)
        operation = uuid.uuid4().hex
        self._evidence.mkdir(exist_ok=True)
        (self._evidence/(operation+'_before.png')).write_bytes(fresh.png)
        self.state['pending'] = {'job': job_id, 'animal': animal, 'stock_before': checked.available,
                                 'operation': operation, 'polygon': polygon}
        self._save()
        owner._check()
        if time.monotonic()-started > 4:
            self.state['pending'] = None
            self._save()
            raise ResourceChanged('Feeding evidence expired before input; no gesture was sent.')
        owner.progress(f'Feeding this {animal} pen and checking the feed-stock decrease.')
        owner.client.drag_path((checked.tool.center, *SheepVision.sweep(polygon, pen[0].width*.55)),
            width=fresh.width, height=fresh.height, duration_ms=4500, cancel_event=owner.cancel_event)
        previous, confirmations, previous_stamp = None, 0, fresh.captured_at
        for _ in range(5):
            owner._wait(.4)
            frame = self._capture()
            current = self.vision.menu(frame.png, animal, owner.cancel_event.is_set)
            if current is None or current.available is None or frame.captured_at == previous_stamp:
                continue
            previous_stamp = frame.captured_at
            observation = (current.available, current.enabled)
            confirmations = confirmations+1 if observation == previous else 1
            previous = observation
            if confirmations >= 2:
                result = self._receipt(job, frame, current)
                if result is not None:
                    return result
        raise ResourceChanged('The feed gesture remains unconfirmed; its saved intent prevents another sweep.')

    def step(self, frame):
        if not self.path.exists():
            return None
        self.state = _load(self.path, self.owner.serial)
        self._save()
        pending = self.state.get('pending')
        jobs = self.state['jobs']
        if pending and pending.get('job') not in jobs:
            return ResourceResult('unsupported', 'An animal-feeding gesture remains unconfirmed; its saved intent is preserved.')
        fruit_items = getattr(self.owner.fruits, 'state', {}).get('items', {})
        if not pending and any(item.get('stage') == 'attempted' for item in fruit_items.values()):
            # Inventory confirmation belongs to the preceding collection.
            # Unrelated old feeding jobs must not send the camera away first.
            return None
        for job in jobs.values():
            if job['stage'] == 'awaiting_harvest':
                harvest = fruit_items.get(job['item_key'], {})
                if (harvest.get('operation') == job['harvest_operation']
                        and harvest.get('stage') == 'confirmed'):
                    job['stage'] = 'needed'
        due = [(key, job) for key, job in jobs.items()
               if job['stage'] in {'needed', 'waiting_feed'} and job.get('next_due', 0) <= utc_timestamp()]
        if pending:
            due = [(pending['job'], jobs[pending['job']])]
        if not due:
            self._save()
            return None
        self.vision = self.vision or HerdVision()
        job_id, job = min(due, key=lambda pair: pair[1].get('last_visit', 0))
        animal = job['animal']
        job.update(last_visit=utc_timestamp(), next_due=utc_timestamp()+60)
        self._save()
        self.owner.progress(f'Returning to the {animal} pen that needs feeding.')
        frame = self._close_board(frame)
        scan = FarmScan()
        deadline = time.monotonic()+180
        navigator = CameraNavigator(self.owner.client, capture=self._capture,
                                    cancel_event=self.owner.cancel_event)
        navigator._deadline = deadline
        menu_dismissals = 0
        for _ in range(64):
            self.owner._check()
            if time.monotonic() >= deadline:
                break
            if self.home_recovery is None:
                from hayday.farm_home import FarmHomeRecovery
                self.home_recovery = FarmHomeRecovery(self.owner.client, capture=self._capture,
                    cancel_event=self.owner.cancel_event, check=self.owner._check,
                    wait=self.owner._wait, progress=self.owner.progress)
            returned = self.home_recovery.process(frame)
            if returned is not frame:
                frame, scan = returned, FarmScan()
                continue
            scene = self.owner._observe(frame)
            if scene.popups or scene.empty_slots:
                # A camera release can select a producer. Its translucent
                # picker contains apparent grass, but dragging there could
                # start production instead of moving the camera.
                if menu_dismissals >= 3 or CameraNavigator._modal_visible(frame):
                    break
                frame = self._dismiss(frame)
                menu_dismissals += 1
                continue
            menu_dismissals = 0
            self._remember_view(frame)
            pens = self._target_pens(job, frame)
            if len(pens) == 1:
                pen = pens[0]
                before = self._capture()
                verified_at = time.monotonic()
                candidates = [candidate for candidate in self._target_pens(job, before)
                              if CowVision.same_pen(candidate[2], pen[2])]
                if len(candidates) != 1 or time.monotonic()-verified_at > 4:
                    # Camera inertia can outlast one capture after centering.
                    # Rediscover in the fresh frame; never tap the older pen.
                    frame = before
                    continue
                pen = candidates[0]
                # Select the trough/coop itself. An animal walking over clear
                # floor can otherwise open its large timer over the fence.
                opened = self.owner._tap(pen[0].center, before)
                menu = self.vision.menu(opened.png, animal, self.owner.cancel_event.is_set)
                if menu is None:
                    self._dismiss(opened)
                    break
                fresh = self._capture()
                checked = self.vision.menu(fresh.png, animal, self.owner.cancel_event.is_set)
                if (checked is None or checked.available != menu.available or checked.enabled != menu.enabled
                        or fresh.captured_at == opened.captured_at):
                    raise ResourceChanged('Animal feed stock changed before confirmation.')
                from hayday.fruit import FruitWorker
                if (not pending and not checked.enabled
                        and FruitWorker._same(menu.tool, checked.tool, fresh)):
                    # The pen was matched twice before opening its trough.
                    # A growth timer can now cover the fence, but two stable
                    # disabled feed controls already establish that no feeding
                    # is available. No sweep needs that occluded outline.
                    self._stock_receipts(job, opened, fresh, checked)
                    job.update(stage='complete', next_due=0)
                    self._save()
                    self._dismiss(fresh, checked)
                    return ResourceResult('waiting', f'The requested {animal} pen is already fed.')
                if not self.vision.confirm_pen(before, fresh, animal, pen, self.owner.cancel_event.is_set):
                    frame = self._dismiss(fresh, checked)
                    continue
                if pending:
                    result = self._receipt(job, fresh, checked)
                    if result is not None:
                        return result
                    raise ResourceChanged('Saved feed intent cannot yet be reconciled; no repeat sweep was sent.')
                if not checked.enabled:
                    self._stock_receipts(job, opened, fresh, checked)
                    job.update(stage='complete', next_due=0)
                    self._save()
                    self._dismiss(fresh, checked)
                    return ResourceResult('waiting', f'The requested {animal} pen is already fed.')
                if checked.available is None:
                    raise ResourceChanged('Animal feed stock is unreadable; no feeding or production input was sent.')
                self._stock_receipts(job, opened, fresh, checked)
                production = self.owner._state['items'].get(job.get('feed_key'), {})
                if checked.available == 0 or production.get('pending_batch'):
                    return self._supply(job, fresh, checked)
                return self._feed(job_id, job, before, pen, fresh, checked)
            if time.monotonic() >= deadline or CameraNavigator._modal_visible(frame):
                break
            menu = self.vision.menu(frame.png, animal, self.owner.cancel_event.is_set)
            if menu is not None:
                frame = self._dismiss(frame, menu)
                continue
            projected = self._project(job, frame)
            if projected is not None:
                dx = float(np.clip(.60-projected[0]/frame.width, -.28, .28))
                dy = float(np.clip(.56-projected[1]/frame.height, -.24, .24))
                if abs(dx)+abs(dy) < .04:
                    break
                direction = (dx, dy)
                message = f'Moving the requested {animal} pen into view.'
            else:
                if scan.phase == 'done':
                    break
                direction = scan.direction
                message = f'{scan.description} to find the requested {animal} pen.'
            drag = CameraNavigator._search_drag(frame, *direction)
            if drag is None:
                break
            previous = frame
            self.owner.progress(message)
            self.owner.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=650)
            try:
                frame = navigator._settled_observe()
            except TimeoutError:
                break
            scan.observe(moved=not CameraNavigator._same_farm_position(previous, frame))
        job['next_due'] = utc_timestamp()+600
        self._save()
        return ResourceResult('waiting', f'The requested {animal} pen was not verified; its feeding task is saved.')

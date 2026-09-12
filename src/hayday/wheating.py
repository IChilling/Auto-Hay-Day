"""Independent, cancellable wheat/roadside-shop sessions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hayday.adb import AdbClient, Screenshot
from hayday.camera import CameraNavigator
from hayday.farming import FarmingWorker
from hayday.resources import _save_json


class WheatingBlocked(RuntimeError):
    pass


class WheatingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class WheatingResult:
    status: str
    message: str
    planted: int = 0
    listed: int = 0
    collected: int = 0
    advertisements: int = 0
    frame: Screenshot | None = None
    diagnostics: Path | None = None

    @property
    def success(self):
        return self.status in {'cancelled', 'limit'}


class WheatingRunner:
    def __init__(self, client, diagnostics_root, *, cancel_event=None, progress=None,
                 max_seconds=None, vision=None, fields_factory=None, shop_factory=None,
                 reset_restart_cooldown=False):
        if max_seconds is not None and (isinstance(max_seconds, bool)
                or not isinstance(max_seconds, (float, int)) or not math.isfinite(max_seconds)
                or max_seconds <= 0):
            raise ValueError('Wheating duration must be positive or unlimited.')
        self.client, self.serial = client, client.serial
        self.root = Path(diagnostics_root)
        device = 'device_'+hashlib.sha256(self.serial.encode()).hexdigest()[:16]
        self.device_root = self.root/device
        # Reuse the orders lock without reading or rewriting order intent.
        self.lock_path = self.root.parent/'orders'/device/'session.lock'
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress = progress
        self.max_seconds = max_seconds
        self.vision = vision
        self.fields_factory, self.shop_factory = fields_factory, shop_factory
        self.frame = None
        self.diagnostics = None
        self.planted = self.listed = self.collected = self.advertisements = 0
        self.state_path = self.device_root/'state.json'
        self.state = {'version': 1, 'serial': self.serial, 'ad_after': 0., 'seed_reserve': 1, 'pending': None}
        self._size = None
        self._deadline = math.inf
        self._lock_file = None
        self._captured = 0.
        self._shop_anchor = None
        self._recovery = None
        self._failsafe = None
        self._reset_restart_cooldown = reset_restart_cooldown
        # Set by popup recovery; consumed by _work before pending crop replay.
        self._level_up_dismissed = False

    save_json = staticmethod(_save_json)

    @staticmethod
    def block(message):
        raise WheatingBlocked(message)

    def cancel(self):
        self.cancel_event.set()
        self.client.close()

    def check(self):
        if self.cancel_event.is_set():
            raise WheatingCancelled('Wheating stopped.')
        if not self.serial or self.client.serial != self.serial:
            self.block('The selected device changed during Wheating.')
        if time.monotonic() >= self._deadline:
            raise TimeoutError('Wheating session time limit reached.')
        if self._failsafe:
            self._failsafe.check()

    def wait(self, seconds):
        self.check()
        self.cancel_event.wait(min(max(0., seconds), max(0., self._deadline-time.monotonic())))
        self.check()

    def publish(self, message):
        if self.progress:
            self.progress({'message': message, 'planted': self.planted, 'listed': self.listed,
                           'collected': self.collected, 'advertisements': self.advertisements,
                           'frame': self.frame})

    def _capture_raw(self, *, fast=False):
        self.check()
        if popup := getattr(self._recovery, 'host_popup', None):
            popup.process()
        capture = getattr(self.client, 'capture_fast', None) if fast else None
        frame = capture() if capture is not None else self.client.capture()
        self.check()
        if self._size is not None and self._size != (frame.width, frame.height):
            self.block('Device resolution changed during Wheating.')
        self._size = frame.width, frame.height
        self.frame, self._captured = frame, time.monotonic()
        return frame

    def capture(self, *, fast=False):
        frame = self._capture_raw(fast=fast)
        if self._recovery:
            frame = self._recovery.process(frame)
        return frame

    def tap(self, point, frame, *, settle=.20):
        self.check()
        if self.frame is not frame or time.monotonic()-self._captured > 5:
            self.block('The observed control became stale before input.')
        self.client.tap(*map(int, point), width=frame.width, height=frame.height)
        self.wait(settle)

    def persist(self):
        self.save_json(self.state_path, self.state)

    def quarantine_file(self, path, reason):
        """Move malformed durable input to a diagnostic archive before reset."""
        path = Path(path)
        if not path.exists():
            return None
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')
        archive = self.device_root/'recovery_archive'/f'{stamp}_{path.name}'
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.replace(archive)
        self.save_json(archive.with_suffix(archive.suffix+'.json'),
                       {'reason': str(reason), 'source': path.name, 'archived_at': time.time()})
        return archive

    def intent(self, operation, **details):
        self.check()
        if self.state.get('pending'):
            self.block('An earlier wheat shop action still needs confirmation.')
        if self.frame:
            (self.device_root/'pending.png').write_bytes(self.frame.png)
        self.state['pending'] = {'operation': operation, 'at': time.time(), **details}
        self.persist()

    def confirmed(self):
        self.state['pending'] = None
        self.persist()

    def pan(self, frame, scan):
        self.check()
        if not self.vision.farm(frame) or CameraNavigator._modal_visible(frame):
            self.block('The farm is obscured; camera scanning paused.')
        dx, dy = scan.direction
        drag = next((p for fraction in (1., .5, .25)
                     if (p := CameraNavigator._grass_start(frame, dx*fraction, dy*fraction))), None)
        if drag is None:
            self.block('No clear grass is available for the next farm scan row.')
        before = CameraNavigator._view(frame)
        self.check()
        self.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=250)
        self.wait(.3)
        previous = None
        for _ in range(5):
            after = self.capture()
            if not self.vision.farm(after):
                self.block('The farm disappeared during camera movement.')
            view = CameraNavigator._view(after)
            if previous is not None and CameraNavigator._same_view(previous, view):
                scan.observe(not CameraNavigator._same_view(before, view))
                return after
            previous = view
            self.wait(.2)
        self.block('The camera did not settle after its movement.')

    def open_shop(self):
        frame = self.frame if self.frame is not None and time.monotonic()-self._captured < 2 else self.capture()
        moved = False
        crop_controls_cleared = False
        for observation in range(12):
            if not self.vision.farm(frame):
                if self.vision.shop(frame).kind == 'overview':
                    return
                self.block('Leave the farm visible so Wheating can find the roadside shop.')
            target = self.vision.shop_building(frame)
            if target:
                fresh = self.capture()
                checked = self.vision.shop_building(fresh)
                if self.vision.farm(fresh) and FarmingWorker._same_target(target, checked, fresh):
                    self._open_verified_shop(fresh, checked)
                    return
                # A crop-menu camera animation can still be settling. All input
                # waits for the same counter and platform in consecutive captures.
                frame = fresh
                self.wait(.15)
                continue
            if not crop_controls_cleared:
                # A sickle/seed overlay can obscure the shop while leaving
                # the farm HUD visible. Clear that verified overlay once, then
                # recognize the actual shop again after the camera settles.
                from hayday.wheating_shop_navigation import dismiss_crop_controls
                cleared = dismiss_crop_controls(self, frame)
                if cleared is not None:
                    crop_controls_cleared = True
                    frame = cleared
                    continue
            if observation >= 2 and not moved and self._shop_anchor is not None:
                before, prior = self._shop_anchor
                point = FarmingWorker._translated_plot(before, frame, prior.center, require_visible=False)
                # Only restore a shop proven to have moved beyond a clear input
                # area. Failed recognition of an already visible shop never pans.
                if point is not None and not (frame.width*.13 < point[0] < frame.width*.87
                                               and frame.height*.18 < point[1] < frame.height*.80):
                    dx = max(-.28, min(.28, .50-point[0]/frame.width))
                    dy = max(-.30, min(.30, .50-point[1]/frame.height))
                    drag = CameraNavigator._grass_start(frame, dx, dy)
                    if drag is not None:
                        self.check()
                        self.client.swipe(*drag, width=frame.width, height=frame.height, duration_ms=300)
                        moved = True
                        self.wait(.3)
            self.wait(.15)
            frame = self.capture()
        self.block('The roadside shop counter and platform could not be recognized. Keep the shop beside the field visible.')

    def _open_verified_shop(self, frame, target):
        is_open = getattr(self.vision, 'shop_is_open', lambda shot: self.vision.shop(shot).kind == 'overview')
        for attempt in range(2):
            anchor = frame, target
            self.tap(target.center, frame)
            previous = False
            for _ in range(6):
                frame = self.capture()
                found = is_open(frame)
                settled = getattr(self.vision, 'shop_layout_current', lambda shot: False)
                if found and (previous or settled(frame)):
                    self._shop_anchor = anchor
                    return
                previous = found
                self.wait(.15)
            if attempt:
                break
            # The first tap can dismiss a lingering crop picker. Retry only the
            # same building, positively recognized again in two fresh farm frames.
            first = self.vision.shop_building(frame) if self.vision.farm(frame) else None
            fresh = self.capture()
            checked = self.vision.shop_building(fresh) if self.vision.farm(fresh) else None
            if not (FarmingWorker._same_target(target, first, frame)
                    and FarmingWorker._same_target(first, checked, fresh)):
                break
            frame, target = fresh, checked
        self.block('The verified roadside shop did not open; no other building was selected.')

    def _acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.lock_path.open('a+b')
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            self.block('Another game session owns this device. Stop it before starting Wheating.')
        self._lock_file = stream
        self.device_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text('utf-8'))
                ready = state.get('field_ready_at', 0.) if isinstance(state, dict) else None
                valid = (isinstance(state, dict) and state.get('version') == 1
                    and state.get('serial') == self.serial
                    and type(state.get('seed_reserve')) is int
                    and 0 <= state['seed_reserve'] <= 9999
                    and type(state.get('ad_after')) in (float, int)
                    and math.isfinite(state['ad_after']) and state['ad_after'] >= 0
                    and type(ready) in (float, int) and math.isfinite(ready) and ready >= 0
                    and type(state.get('wheat_empty', False)) is bool)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                valid, state = False, None
                reason = f'Saved Wheating state could not be read: {exc}'
            else:
                reason = ('Saved Wheating state failed validation.' if not valid else '')
            if not valid:
                self.quarantine_file(self.state_path, reason)
                self.state = {'version': 1, 'serial': self.serial, 'ad_after': 0.,
                              'seed_reserve': 1, 'pending': None}
                self.persist()
            else:
                self.state = state
                pending = self.state.get('pending')
                if pending is not None and not isinstance(pending, dict):
                    self.state['pending'] = None
                    self.persist()
                elif pending is not None and pending.get('operation') != 'collect':
                    # A live AdbClient can reconcile list/advertise intents
                    # through the restart supervisor after one fresh frame.
                    # Keep the strict legacy guard for lightweight callers that
                    # cannot provide that verification contract.
                    if not isinstance(self.client, AdbClient):
                        self.block('An earlier wheat shop action is unconfirmed. Inspect pending.png and state.json before resuming.')

    def _wait_in_field(self, fields):
        """Watch the crop after filling the shop; buyers never gate harvesting."""
        shop_check_at = time.monotonic()+10
        while time.monotonic() < fields.next_harvest:
            self.check()
            if fields.ready_to_harvest():
                fields.next_harvest = time.monotonic()
                return
            now = time.monotonic()
            if now >= fields.next_harvest:
                return
            # Refill newly freed slots while crops are still growing, but once
            # all surplus is listed there is no reason to wait on receipts.
            has_surplus = not self.state.get('wheat_empty', False)
            if has_surplus and now >= shop_check_at:
                self.open_shop()
                return
            remaining = max(0, math.ceil(fields.next_harvest-now))
            self.publish(f'Wheating: wheat is growing; checking the field for harvest ({remaining}s remaining).')
            until = min(fields.next_harvest, shop_check_at if has_surplus else math.inf)
            self.wait(min(5., max(0., until-now)))

    def _work(self, fields, shop):
        from hayday.wheating_silo import SiloFullDetected

        def recover_silo(detected=None):
            """Resume a saved Silo Full transaction without a terminal timeout."""
            result = self._recovery.silo_full.recover(fields, shop, detected)
            # Real WheatSiloFull returns False when a shop pass made no progress
            # before its slice expired.  Re-enter the shop after a short wait;
            # a test double or legacy recovery returning None remains a single
            # pass so it cannot accidentally create an infinite loop.
            while self.state.get('silo_recovery') is not None and result is not None:
                if result is True:
                    return
                self.publish('Wheating: silo recovery paused. Retrying the roadside shop shortly.')
                self.wait(5.)
                try:
                    frame = self.capture()
                except SiloFullDetected as exc:
                    result = self._recovery.silo_full.recover(fields, shop, exc)
                    continue
                if getattr(fields, 'reconcile_stale_state', lambda _frame: False)(frame):
                    return
                result = self._recovery.silo_full.recover(fields, shop)

        # Inspect the live farm before replaying durable crop work.  The farm's
        # documented initial state is fully mature wheat, which supersedes an
        # interrupted gesture or partial Silo Full recovery when it is visible.
        try:
            frame = self.capture()
        except SiloFullDetected as exc:
            recover_silo(exc)
            frame = self.capture()
        # A level-up dismissal can leave the farm animation/camera settling.
        # Take one short, fresh observation before reconciling durable crop work.
        if self._level_up_dismissed:
            self.publish('Wheating: stabilizing the farm after level-up dismissal.')
            self.wait(.35)
            frame = self.capture()
            self._level_up_dismissed = False
        if getattr(fields, 'reconcile_stale_state', lambda _frame: False)(frame):
            frame = self.capture()
        if self.state.get('pending'):
            pending = self.state['pending']
            if pending.get('operation') == 'collect':
                shop.reconcile_collection()
            elif self._failsafe:
                self._failsafe.resume_pending(shop)
            else:
                self.block('An earlier wheat shop action is unconfirmed.')
        if self.state.get('silo_recovery') is not None:
            recover_silo()
            frame = self.capture()
        self.publish('Wheating started: one harvest sweep, one planting sweep, then wheat sales while the field grows.')
        initial = self.vision.shop(frame)
        if initial.kind in {'overview', 'composer', 'edit'}:
            shop.close_to_farm()
        if fields.next_harvest > time.monotonic():
            self.open_shop()
        while True:
            self.check()
            if time.monotonic() >= fields.next_harvest:
                shop.close_to_farm()
                while True:
                    try:
                        fields.pass_all()
                        break
                    except SiloFullDetected as exc:
                        recover_silo(exc)
                if fields.has_fields and self.planted:
                    group = getattr(fields, '_group', None)
                    self.state['seed_reserve'] = group.seed_reserve if group else 0
                    self.persist()
                self.open_shop()
                stock = getattr(fields, 'last_stock', None)
                if stock is not None:
                    shop._stock_empty_until = math.inf if stock == 0 else 0.
            idle = shop.service(deadline=fields.next_harvest)
            # Finish verification of the current transaction, then leave
            # immediately when harvesting is due. Do not extend the crop
            # timer or add a buyer/ad wait after the final listing.
            if time.monotonic() >= fields.next_harvest:
                self.publish('Wheating: wheat is ready. Returning to harvest and replant.')
                continue
            if idle:
                self.publish('Wheating: shop stocked. Returning to check the wheat field.')
                shop.close_to_farm()
                self._wait_in_field(fields)
                continue
            self.wait(min(.5, max(0., fields.next_harvest-time.monotonic())))

    def run(self):
        from hayday.wheating_fields import WheatFields
        from hayday.wheating_shop import WheatShop
        from hayday.wheating_vision import WheatingVision

        status, message = 'error', 'Wheating did not start.'
        try:
            self.check()
            self._acquire()
            self._deadline = time.monotonic()+(self.max_seconds or math.inf)
            self.diagnostics = self.root/(datetime.now(UTC).strftime('%Y%m%dT%H%M%S')+'_'+os.urandom(3).hex())
            self.diagnostics.mkdir(parents=True)
            self.vision = self.vision or WheatingVision(cancel=self.cancel_event.is_set)
            if isinstance(self.vision, WheatingVision):
                from hayday.wheating_recovery import WheatRecovery
                self._recovery = WheatRecovery(self)
                if isinstance(self.client, AdbClient):
                    from hayday.wheating_restart import WheatRestart
                    self._failsafe = WheatRestart(self)
                    if self._reset_restart_cooldown:
                        self._failsafe.reset_cooldown()
            if self._failsafe:
                self._failsafe.monitoring = True
                self._failsafe.reset_progress()
            restarting = False
            while True:
                fields = None
                try:
                    self.check()
                    fields = (self.fields_factory or WheatFields)(self)
                    shop = (self.shop_factory or WheatShop)(self)
                    if restarting:
                        # Preserve the ongoing cycle's growth deadline, while
                        # rebuilding all observations from the new game process.
                        fields.next_harvest = time.monotonic()+max(0., self.state.get('field_ready_at', 0)-time.time())
                        worker = getattr(fields, 'worker', None)
                        if worker and any(item.get('stage') in worker._PENDING
                                          for item in worker.state['items'].values()):
                            fields.next_harvest = 0.
                        self._failsafe.resume_pending(shop)
                    self._work(fields, shop)
                    break
                except (WheatingCancelled, TimeoutError):
                    raise
                except Exception as exc:
                    if self._failsafe is None:
                        raise
                    self._failsafe.recover(exc, fields)
                    restarting = True
        except WheatingCancelled as exc:
            status, message = 'cancelled', str(exc)
        except TimeoutError as exc:
            status, message = 'limit', str(exc)
        except Exception as exc:
            status = 'cancelled' if self.cancel_event.is_set() else 'blocked' if isinstance(exc, WheatingBlocked) else 'error'
            message = 'Wheating stopped.' if self.cancel_event.is_set() else f'Wheating stopped: {exc}'
        finally:
            if self._failsafe:
                self._failsafe.monitoring = False
            if popup := getattr(self._recovery, 'host_popup', None):
                popup.close()
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
        result = WheatingResult(status, message, self.planted, self.listed, self.collected,
                                self.advertisements, self.frame, self.diagnostics)
        if self.diagnostics:
            if self.frame:
                (self.diagnostics/'last.png').write_bytes(self.frame.png)
            self.save_json(self.diagnostics/'result.json', {**result.__dict__, 'frame': None,
                           'diagnostics': str(self.diagnostics)})
        return result

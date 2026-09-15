"""Last-resort, device-scoped game restart after local recovery is exhausted."""
from __future__ import annotations

import json
import math
import time

from hayday.adb import AdbError
from hayday.wheating import WheatingBlocked, WheatingPrerequisite


class WheatStalled(WheatingBlocked):
    pass


class WheatRestart:
    STALL_SECONDS = 180.
    MIN_INTERVAL = 120.
    INTERVAL_STEP = 60.
    MAX_INTERVAL = 600.
    HISTORY_LIMIT = 1+int((MAX_INTERVAL-MIN_INTERVAL)/INTERVAL_STEP)

    def __init__(self, run, *, clock=time.monotonic, wall=time.time):
        self.run, self.clock, self.wall = run, clock, wall
        self.path = run.device_root/'restart.json'
        self.active = False
        self.monitoring = False
        self.last_progress = clock()
        self.signature = self._progress()
        self.history = []
        self.stage = 'idle'
        self._launch_reserved = False
        self._local_signature = None
        self.reason = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text('utf-8'))
                history = data.get('attempts') if isinstance(data, dict) else None
                valid = (isinstance(data, dict) and data.get('version') == 1
                    and data.get('serial') == run.serial and isinstance(history, list)
                    and len(history) <= 100
                    and all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in history))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                valid, history = False, []
                reason = f'Saved game restart history could not be read: {exc}'
            else:
                reason = 'Saved game restart history failed validation.'
            if not valid:
                run.quarantine_file(self.path, reason)
                history = []
            self.history = history[-self.HISTORY_LIMIT:]
            self.reason = data.get('reason') if valid else None
        # Persist wall timestamps, but wait on a monotonic clock in this session.
        # A clock rollback requires at most one full interval, never an early retry.
        remaining = (min(self.interval, max(0., self.interval-(wall()-self.history[-1])))
                     if self.history else 0.)
        self._retry_at = clock()+remaining

    @property
    def interval(self):
        return min(self.MAX_INTERVAL, self.MIN_INTERVAL+max(0, len(self.history)-1)*self.INTERVAL_STEP)

    def reset_cooldown(self):
        """A manual Start begins a fresh budget after acquiring this device."""
        self.run.check()
        if not self.run._lock_file:
            self.run.block('Resetting the restart cooldown requires the active device session lock.')
        self.history = []
        self._retry_at = self.clock()
        self._launch_reserved = False
        self._local_signature = None
        self.reason = None
        self._record('manual_start')

    def wait_for_retry(self):
        self.run.check()
        if not self.run._lock_file:
            self.run.block('Game restart requires the active device session lock.')
        while (remaining := self._retry_at-self.clock()) > 0:
            minutes, seconds = divmod(math.ceil(remaining), 60)
            self.run.publish(f'Wheating: restart cooldown {minutes}:{seconds:02d} remaining '
                             f'({self.interval/60:g}-minute interval).')
            self.run.wait(min(10., remaining))
            self.run.check()

    def _attempt(self, stage, reason=None):
        self.history = [*self.history, self.wall()][-self.HISTORY_LIMIT:]
        self._retry_at = self.clock()+self.interval
        self._record(stage, reason)

    def before_launch(self):
        # The first relaunch completes the restart already counted at force-stop.
        if not self._launch_reserved:
            self.wait_for_retry()

    def launch_attempted(self):
        if self._launch_reserved:
            self._launch_reserved = False
        else:
            self._attempt('launch_requested')

    def _progress(self):
        return self.run.planted, self.run.listed, self.run.collected, self.run.advertisements

    def reset_progress(self):
        self.signature, self.last_progress = self._progress(), self.clock()

    def check(self):
        if not self.monitoring or self.active:
            return
        signature = self._progress()
        if signature != self.signature:
            self.reset_progress()
        if self.clock()-self.last_progress >= self.STALL_SECONDS:
            raise WheatStalled('Wheating made no confirmed progress for three minutes.')

    def _record(self, stage, reason=None):
        self.stage = stage
        if reason is not None:
            self.reason = str(reason)
        data = {'version': 1, 'serial': self.run.serial, 'attempts': self.history,
                'stage': stage, 'updated_at': self.wall(), 'interval_seconds': self.interval}
        if self.reason:
            data['reason'] = self.reason
        self.run.save_json(self.path, data)
        if self.run.diagnostics:
            self.run.save_json(self.run.diagnostics/'restart.json', data)

    def _capture(self, label=None):
        # Bypass an exhausted game/host-dialog helper. The fixed-package stop
        # doesn't need to click through the screen that caused the failure.
        self.run.check()
        frame = self.run.client.capture()
        self.run.check()
        if self.run._size is not None and self.run._size != (frame.width, frame.height):
            self.run.block('Device resolution changed; game restart was stopped.')
        self.run._size = frame.width, frame.height
        self.run.frame, self.run._captured = frame, time.monotonic()
        if label and self.run.diagnostics:
            (self.run.diagnostics/(label+'.png')).write_bytes(frame.png)
        return frame

    @staticmethod
    def _retryable_restart_error(error):
        message = str(error).casefold()
        return not any(term in message for term in (
            'resolution', 'selected device', 'device changed', 'session lock',
            'cancelled', 'deadline',
        ))

    def recover(self, error, fields):
        if isinstance(error, WheatingPrerequisite):
            raise error
        self.active = True
        self._record('recovery_requested', error)
        try:
            if (self.run.state.get('pending') or {}).get('operation') == 'advertise':
                self.run.check()
                if not self.run._lock_file:
                    self.run.block('Local recovery requires the active device session lock.')
                if not self._retryable_restart_error(error):
                    raise error
                from hayday.wheating_advertising import reconcile, save_trace, trace
                from hayday.wheating_shop import WheatShop
                shop = WheatShop(self.run)
                frame = self._capture('recovery_before')
                view = self.run.vision.shop(frame)
                trace(shop, 'recovery', frame, view, error=str(error))
                try:
                    resolved = reconcile(shop, (frame, view))
                finally:
                    save_trace(shop)
                if resolved:
                    self._record('advertisement_deferred_locally', error)
                    self.reset_progress()
                    return
            # Recognition failures on a visible farm are not evidence that the
            # game crashed. Rebuild observations and resume durable crop intent
            # locally once; restarting cannot repair a persistently bad match.
            if fields is not None and self.run.vision is not None and not isinstance(error, WheatStalled):
                self.run.check()
                if not self.run._lock_file:
                    self.run.block('Local recovery requires the active device session lock.')
                frame = self._capture('recovery_before')
                from hayday.camera import CameraNavigator
                if self.run.vision.farm(frame) and not CameraNavigator._modal_visible(frame):
                    if getattr(fields, 'has_live_work', lambda _frame: False)(frame):
                        if self._local_signature == self._progress():
                            self._record('recognition_blocked', error)
                            self.run.block(f'Field recovery made no progress: {error}. '
                                           'Hay Day was left open and pending work was preserved.')
                        self._local_signature = self._progress()
                        # Restarting a visible, actionable field cannot repair
                        # stale geometry. Let the crop worker rebuild from its
                        # fresh soil/crop observations, preserving pending input.
                        self._record('live_field_retry', error)
                        self.run.publish('Wheating: visible field found. Retrying its crop controls without restarting Hay Day.')
                        try:
                            fields.worker._close(frame)
                        except (WheatingBlocked, AdbError) as exc:
                            self._record('live_field_retry_failed', exc)
                        else:
                            self.run.wait(2.)
                            self.reset_progress()
                            return
                    exhausted = False
                    if self._local_signature == self._progress():
                        # A real WheatFields instance can reconcile a fresh
                        # mature field and quarantine stale crop work.  Keep a
                        # compatibility guard for lightweight callers that do
                        # not provide that recovery contract.
                        if not callable(getattr(fields, 'reconcile_stale_state', None)):
                            self._record('recognition_blocked', error)
                            self.run.block(
                                f'Farm recognition could not recover: {error}. '
                                'Hay Day was left open and the pending work was preserved.'
                            )
                        self._record('local_retry_exhausted', error)
                        self.run.publish(
                            'Wheating: the same farm view persisted after a local retry. '
                            'Restarting Hay Day to rebuild live observations.'
                        )
                        exhausted = True
                    self._local_signature = self._progress()
                    if not exhausted:
                        self._record('local_retry', error)
                        self.run.publish('Wheating: refreshing the field controls and resuming pending work without restarting Hay Day.')
                        try:
                            fields.worker._close(frame)
                            # Refreshing controls alone cannot repair a restart
                            # or reconnect that changed the farm's zoom/position.
                            if getattr(fields, '_known_before', None) is not None:
                                from hayday.wheating_restart_camera import restore_workspace
                                restore_workspace(self.run, fields)
                        except (WheatingBlocked, AdbError) as exc:
                            # A failed overlay clear is still recoverable; let
                            # the background restart path handle it instead of
                            # turning the local cleanup error into a hard stop.
                            self._record('local_retry_failed', exc)
                        else:
                            self.reset_progress()
                            return
            try:
                self.wait_for_retry()
                self.run.publish('Wheating: local recovery failed. Restarting Hay Day in the background.')
                self._capture('restart_before')  # require a live, same-size device before stopping
                self._attempt('stop_requested', error)
                self.run.check()
                self.run.client.force_stop_hay_day()
                for _ in range(10):
                    self.run.check()
                    if not self.run.client.hay_day_running():
                        break
                    self.run.wait(.2)
                else:
                    self.run.block('Hay Day did not stop; no repeated stop command was sent.')
                self._record('stopped', error)
                self.run.client.home()
                self.run.wait(.3)
                frame = self._capture('restart_launcher')
                from hayday.wheating_recovery import WheatRecovery
                from hayday.wheating_vision import WheatingVision

                previous = self.run._recovery
                # Reset observations for the stopped process. The persistent retry
                # schedule belongs to this failsafe, shared with the launcher helper.
                previous.host_popup.close()
                self.run.vision = WheatingVision(cancel=self.run.cancel_event.is_set)
                self.run._shop_anchor = None
                self.run._workspace_zoom_attempted = False
                self.run._workspace_needs_restore = True
                self.run._recovery = WheatRecovery(self.run)
                self._record('launch_requested', error)
                self._launch_reserved = True
                recovered = self.run._recovery.launch.process(frame)
                if recovered is frame or self.run.client.foreground_package() != 'com.supercell.hayday':
                    self.run.block('Hay Day could not be verified after the background restart.')
                self._record('reopened', error)
                self.run.publish('Wheating: Hay Day restarted. Restoring the wheat workspace.')
                from hayday.wheating_restart_camera import restore_workspace
                try:
                    restore_workspace(self.run, fields)
                except WheatingBlocked as exc:
                    # A stale crop record or a changed camera view must not turn a
                    # successful app restart into a terminal Wheating stop.  The
                    # next runner iteration will inspect the live farm and can
                    # quarantine stale work when the mature-field invariant holds.
                    self._record('workspace_restore_retry', exc)
                    self.run.publish('Wheating: workspace restore needs another live farm frame; retrying automatically.')
                    self._local_signature = None
                    self.reset_progress()
                    return
                self._record('ready')
                self._local_signature = None
                self.reset_progress()
            except WheatingBlocked as exc:
                # A real field runner can safely retry failed stop/launch
                # verification after the current backoff.  Lightweight test or
                # integration callers without that contract retain the strict
                # error so unsafe inputs are never hidden.
                if not callable(getattr(fields, 'reconcile_stale_state', None)) or not self._retryable_restart_error(exc):
                    raise
                self._record('restart_retry', exc)
                self.run.publish('Wheating: restart verification did not settle; waiting before retrying automatically.')
                self._local_signature = None
                self.wait_for_retry()
                return
        finally:
            self._launch_reserved = False
            self.active = False

    def resume_pending(self, shop):
        """Confirm supported old outcomes; a restart never implies success."""
        pending = self.run.state.get('pending')
        if not pending:
            return
        self.active = True
        try:
            if pending.get('operation') == 'advertise':
                from hayday.wheating_advertising import reconcile, save_trace
                try:
                    if reconcile(shop):
                        return
                    self.run.open_shop()
                    if reconcile(shop):
                        return
                    self.run.block('The shop is obscured while inspecting an earlier advertisement.')
                finally:
                    save_trace(shop)
            self.run.open_shop()
            if pending.get('operation') == 'collect':
                shop.reconcile_collection()
                return
            from hayday.resource_vision import VisualTarget
            from hayday.wheating_vision import SaleSlot
            box = pending.get('slot')
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(n) is not int for n in box) or min(box[:2]) < 0 or min(box[2:]) <= 0):
                self.run.block('Saved shop intent has invalid slot bounds.')
            slot = SaleSlot('wheat', VisualTarget(*box, 1.))
            for _ in range(2):
                frame, view = shop.observe()
                found = shop._slot(view, slot, {'wheat', 'sold'}) if view.kind == 'overview' else None
                if found is None:
                    self.run.block('Hay Day restarted, but the earlier shop action is still unconfirmed; its intent was preserved.')
                self.run.wait(.2)
            if pending.get('operation') == 'list':
                quantity = pending.get('quantity')
                if type(quantity) is not int or not 1 <= quantity <= 10:
                    self.run.block('Saved wheat sale quantity is invalid.')
                self.run.listed += quantity
                if self.run.state.get('silo_recovery') is not None:
                    self.run.state['silo_recovery']['released'] += quantity
                    stock = pending.get('stock')
                    if type(stock) is int and stock >= quantity:
                        empty = stock-quantity <= self.run.state['seed_reserve']
                        self.run.state['silo_recovery']['surplus_empty'] = empty
                        self.run.state['wheat_empty'] = empty
            else:
                self.run.block('The earlier shop operation cannot be reconciled automatically.')
            self.run.confirmed()
            shop._ready_view = frame, view
            self.run.publish('Wheating: verified the earlier shop action after restarting.')
        finally:
            self.active = False

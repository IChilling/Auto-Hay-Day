"""Optional advertisements must never hold the crop/sales transaction queue."""
from __future__ import annotations

import json
import math
import time

from hayday.adb import AdbError
from hayday.camera import CameraNavigator
from hayday.resource_vision import VisualTarget
from hayday.wheating import WheatingBlocked
from hayday.wheating_vision import SaleSlot

COOLDOWN = 301.


def submission_after(run):
    """Keep actual submission attempts protected independently of ad scheduling."""
    last = run.state.get('last_advertisement')
    if last is None:
        return 0.
    if isinstance(last, dict) and isinstance(last.get('intent'), dict):
        attempted = last['intent'].get('at')
        if type(attempted) not in (int, float) or not math.isfinite(attempted):
            attempted = last.get('at')
        if type(attempted) in (int, float) and math.isfinite(attempted):
            return attempted+COOLDOWN
    # Incomplete legacy evidence cannot authorize an early repeat.
    return run.state['ad_after']


def trace(shop, stage, frame=None, view=None, **details):
    events = getattr(shop, '_ad_events', None)
    if events is None:
        shop._ad_events = events = []
    event = {'stage': stage, 'at': time.time(), **details}
    if frame is not None:
        event['captured_at'] = frame.captured_at
        if shop.run.frame is frame:
            event['frame_age_ms'] = round(1000*max(0., time.monotonic()-shop.run._captured), 1)
        shop._ad_last_frame = frame
    if view is not None:
        event.update(view=view.kind, free=view.ad_free, checked=view.ad_checked,
                     cooldown=view.cooldown, submit=list(view.submit.box) if view.submit else None)
    events.append(event)
    del events[:-16]


def save_trace(shop):
    """Bounded per-device evidence; no disk writes on the critical tap path."""
    events = getattr(shop, '_ad_events', [])
    if not events:
        return
    path = shop.run.device_root/'advertisements.json'
    try:
        try:
            records = json.loads(path.read_text('utf-8'))
        except (OSError, ValueError):
            records = []
        if not isinstance(records, list):
            records = []
        shop.run.save_json(path, [*records[-7:], list(events)])
        if any(e['stage'] in {'uncertain', 'error'} for e in events):
            frame = getattr(shop, '_ad_last_frame', None)
            if frame is not None:
                (shop.run.device_root/'advertisement_after.png').write_bytes(frame.png)
            before = getattr(shop, '_ad_before', None)
            if before is not None:
                (shop.run.device_root/'advertisement_before.png').write_bytes(before.png)
    except OSError:
        pass  # Optional diagnostics cannot block farming.
    shop._ad_events = []


def _healthy(shop, frame, view):
    return (view.kind in {'overview', 'edit', 'composer'}
            or (shop.run.vision.farm(frame) and not CameraNavigator._modal_visible(frame)))


def _retryable(error):
    from hayday.wheating_restart import WheatRestart
    return WheatRestart._retryable_restart_error(error)


def _finish(shop, frame, view, outcome):
    run = shop.run
    run.check()
    pending = run.state.get('pending')
    if not pending or pending.get('operation') != 'advertise':
        return False
    attempted = pending.get('at')
    if type(attempted) not in (int, float) or not math.isfinite(attempted):
        attempted = time.time()
    # No repeated submit after ambiguous delivery, including across restarts.
    run.state['ad_after'] = max(run.state.get('ad_after', 0), attempted+COOLDOWN,
                               time.time()+30 if outcome == 'uncertain' else 0)
    if view.cooldown is not None:
        run.state['ad_after'] = max(run.state['ad_after'], time.time()+view.cooldown+1)
    run.state['last_advertisement'] = {'intent': dict(pending), 'outcome': outcome, 'at': time.time()}
    if outcome == 'confirmed':
        run.advertisements += 1
    # SOLD/empty is terminal for this listing, but does not prove an ad ran.
    trace(shop, outcome, frame, view)
    run.confirmed()
    if view.kind == 'edit':
        shop.close(frame, view)
    elif view.kind == 'overview':
        shop._ready_view = frame, view
    run.publish('Wheating: advertisement checked. Continuing wheat work.' if outcome == 'confirmed'
                else 'Wheating: advertisement deferred; cooldown retained. Continuing wheat work.')
    return True


def reconcile(shop, initial=None, *, probe=True):
    """Observe a pending ad, optionally inspect its edit panel, never resubmit."""
    run = shop.run
    pending = run.state.get('pending')
    if not pending or pending.get('operation') != 'advertise':
        return False
    box = pending.get('slot')
    valid = (isinstance(box, list) and len(box) == 4 and all(type(n) is int for n in box)
             and min(box[:2]) >= 0 and min(box[2:]) > 0)
    slot = SaleSlot('wheat', VisualTarget(*box, 1.)) if valid else None
    previous = None
    for attempt in range(2):
        frame, view = initial if attempt == 0 and initial else shop.observe()
        trace(shop, 'verify', frame, view)
        if not _healthy(shop, frame, view):
            return False
        found = shop._slot(view, slot, {'wheat', 'sold', 'empty'}) if slot and view.kind == 'overview' else None
        outcome = ('confirmed' if (found and found.advertised) or (
            view.kind == 'edit' and view.wheat is not None and not view.ad_free
            and (view.cooldown is not None or view.ad_checked)) else
            'listing_gone' if found and found.kind in {'sold', 'empty'} else 'uncertain')
        if previous == outcome or view.layout_verified:
            if outcome != 'uncertain':
                return _finish(shop, frame, view, outcome)
        previous = outcome
        if attempt == 0:
            run.wait(.08)
    # Inspect only an observed wheat listing, and don't delay a due harvest.
    if (probe and found and found.kind == 'wheat' and view.kind == 'overview'
            and time.time() < run.state.get('field_ready_at', float('inf'))):
        try:
            opened = shop._open_slot(frame, view, found, 'edit')
        except WheatingBlocked as error:
            if not _retryable(error):
                raise
            trace(shop, 'probe_failed', error=str(error))
            opened = None
        if opened:
            return reconcile(shop, opened, probe=False)
        frame, view = shop.observe()
        if not _healthy(shop, frame, view):
            return False
    return _finish(shop, frame, view, 'uncertain')


def advertise(shop, frame, view, slot, *, check_ready=False):
    if shop.run.state.get('pending'):
        return
    shop._ad_events = []
    shop._ad_before = frame
    trace(shop, 'begin', frame, view, slot=list(slot.target.box))
    next_check = None
    try:
        next_check = shop._advertise_once(frame, view, slot, check_ready=check_ready)
    except (WheatingBlocked, AdbError) as error:
        shop.run.check()  # Cancellation/device/deadline guards still win.
        trace(shop, 'error', error=str(error))
        if not _retryable(error):
            raise
        current, observed = shop.observe()
        if not _healthy(shop, current, observed):
            raise
        if (shop.run.state.get('pending') or {}).get('operation') == 'advertise':
            if not reconcile(shop, (current, observed)):
                raise
        elif shop.run.state.get('pending'):
            raise  # Never retire a sale or collection through optional recovery.
        else:
            retained = submission_after(shop.run) if check_ready else shop.run.state['ad_after']
            shop.run.state['ad_after'] = max(retained, time.time()+30)
            shop.run.persist()
            trace(shop, 'uncertain', current, observed)
            if observed.kind == 'edit':
                shop.close(current, observed)
            elif observed.kind == 'overview':
                shop._ready_view = current, observed
    finally:
        save_trace(shop)
        recovery = shop.run.state.get('silo_recovery')
        if check_ready and recovery is not None:
            recovery['ad_check_after'] = max(time.time()+30,
                shop.run.state['ad_after'] if next_check is None else next_check)
            shop.run.persist()

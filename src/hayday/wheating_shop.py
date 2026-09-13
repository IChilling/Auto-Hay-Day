"""Fill empty shop slots with 10 wheat at 36 coins and use free advertisements."""
from __future__ import annotations

import time

import numpy as np

from hayday.farming import FarmingWorker
from hayday.resource_vision import VisualTarget
from hayday.wheating_vision import SaleSlot

STACK = 10
MAX_PRICE = 36
AD_COOLDOWN = 300


def saleable_stacks(stock, reserve):
    if type(stock) is not int or type(reserve) is not int or reserve < 0:
        return 0
    return max(0, stock-reserve)//STACK


def sale_quantity(stock, reserve):
    if type(stock) is not int or type(reserve) is not int or reserve < 0:
        return 0
    return min(STACK, max(0, stock-reserve))


class WheatShop:
    def __init__(self, runner):
        self.run = runner
        self._stock_empty_until = float('inf') if runner.state.get('wheat_empty') else 0.
        self._ready_view = None

    def observe(self):
        frame = self.run.capture()
        view = self.run.vision.shop(frame)
        self.run.check()
        return frame, view

    def _tap(self, point, frame, *, settle=.06):
        """Static shop controls need only a short delay before observing again."""
        self.run.check()
        if self.run.frame is not frame or time.monotonic()-self.run._captured > 2:
            self.run.block('The observed shop control became stale before input.')
        self.run.client.tap(*map(int, point), width=frame.width, height=frame.height)
        self.run.wait(settle)

    def reconcile_collection(self):
        pending = self.run.state.get('pending')
        boxes = pending.get('slots', [pending.get('slot')]) if isinstance(pending, dict) else None
        if (not pending or pending.get('operation') != 'collect' or not isinstance(boxes, list)
                or not 1 <= len(boxes) <= 24
                or any(not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box)
                       or min(box[:2]) < 0 or min(box[2:]) <= 0 for box in boxes)):
            self.run.block('The saved wheat receipt collection is invalid.')
        slots = [SaleSlot('sold', VisualTarget(*box, 1.)) for box in boxes]
        for _ in range(2):
            frame, view = self.observe()
            if (view.kind != 'overview'
                    or any(slot.target.x+slot.target.width > frame.width or slot.target.y+slot.target.height > frame.height
                           or self._slot(view, slot, {'empty'}) is None for slot in slots)):
                self.run.block('An earlier wheat receipt collection is unconfirmed. No collection was replayed.')
            self.run.wait(.2)
        self._ready_view = frame, view
        self.run.collected += len(slots)
        self.run.confirmed()
        self.run.publish('Wheating: verified the earlier receipt collection. Resuming wheat work.')

    def current(self, frame, view):
        if self.run.frame is frame and time.monotonic()-self.run._captured < 2:
            self.run.check()
            return frame, view
        return self.observe()

    @staticmethod
    def _slot(view, previous, kind):
        return next((slot for slot in view.slots if slot.kind in kind
                     and np.linalg.norm(np.subtract(slot.target.center, previous.target.center))
                     < min(slot.target.width, previous.target.width)*.38), None)

    def close(self, frame, view):
        if view.close is None:
            self.run.block('The shop close control is not recognized.')
        fresh, checked = self.current(frame, view)
        if checked.kind != view.kind or not FarmingWorker._same_target(view.close, checked.close, fresh):
            self.run.block('The shop changed before closing its panel.')
        self._tap(checked.close.center, fresh)
        self._ready_view = None

    def close_to_farm(self):
        closes = 0
        ready, self._ready_view = self._ready_view, None
        for _ in range(10):
            frame, view = self.current(*ready) if ready else self.observe()
            ready = None
            if view.kind == 'unknown' and self.run.vision.farm(frame):
                return
            if view.kind == 'unknown' and closes < 3 and self._dismiss_dialog(frame):
                closes += 1
                continue
            if view.kind == 'unknown' or view.close is None:
                self.run.wait(.2)
                continue
            if closes >= 3:
                break
            self.close(frame, view)
            closes += 1
        self.run.block('The farm did not return after closing the shop.')

    def _dismiss_dialog(self, frame):
        detect = getattr(self.run.vision, 'dialog_close', None)
        target = detect(frame) if detect else None
        if target is None:
            return False
        fresh, view = self.observe()
        checked = detect(fresh)
        if view.kind != 'unknown' or not FarmingWorker._same_target(target, checked, fresh):
            return False
        self.run.tap(checked.center, fresh)
        self.run.publish('Wheating: closed an interrupting game dialog. Resuming wheat work.')
        return True

    def _await_slot(self, slot, kinds):
        return self._await_slots([slot], kinds)

    def _await_slots(self, slots, kinds):
        prior = False
        deadline = time.monotonic()+6
        for _ in range(24):
            frame, view = self.observe()
            if view.kind == 'unknown' and self._dismiss_dialog(frame):
                prior = False
                continue
            found = view.kind == 'overview' and all(self._slot(view, slot, kinds) for slot in slots)
            if found and (prior or view.layout_verified):
                self._ready_view = frame, view
                return frame, view
            prior = found
            if time.monotonic() >= deadline:
                break
            self.run.wait(.08)
        self.run.block('The shop action was sent once, but its slot did not confirm. Its saved intent remains pending.')

    def _open_slot(self, frame, view, slot, expected):
        fresh, checked = self.current(frame, view)
        target = self._slot(checked, slot, {slot.kind})
        if checked.kind != 'overview' or target is None:
            return None
        # Capturing halfway through the dialog's opening animation costs a
        # whole extra screenshot. Let that transition finish before reading.
        self._tap(target.target.center, fresh, settle=.30)
        prior = None
        for _ in range(5):
            after, panel = self.observe()
            if (expected == 'edit' and panel.kind == 'overview'
                    and self._slot(panel, slot, {'sold', 'empty'})):
                # A buyer can purchase between the observed wheat and our tap.
                # Resume ordinary receipt/refill work instead of treating that
                # successful sale as a missing advertisement dialog.
                self._ready_view = after, panel
                return None
            ready = panel.kind == expected and panel.close is not None
            if expected == 'composer':
                ready = ready and (panel.wheat is not None or panel.silo_tab is not None)
            if ready and (panel.layout_verified or prior and FarmingWorker._same_target(prior.close, panel.close, after)):
                return after, panel
            prior = panel if ready else None
            self.run.wait(.08)
        if expected == 'composer' and panel.kind == 'overview':
            # No sale was submitted. Refresh a stale crate or missed opening
            # tap locally; no transaction or app restart is needed here.
            self._ready_view = after, panel
            self.run.publish('Wheating: sale panel did not open; refreshing shop slots.')
            return None
        self.run.block(f'The wheat shop did not show its {expected} controls.')

    def _collect(self, frame, view, slot):
        fresh, checked = self.current(frame, view)
        target = self._slot(checked, slot, {'sold'})
        if target is None:
            return
        self.run.intent('collect', slot=list(target.target.box))
        self._tap(target.target.center, fresh)
        self._await_slot(slot, {'empty'})
        self.run.collected += 1
        self.run.confirmed()

    def _collect_many(self, frame, view, slots):
        fresh, checked = self.current(frame, view)
        targets = [self._slot(checked, slot, {'sold'}) for slot in slots]
        targets = [slot for slot in targets if slot is not None]
        if checked.kind != 'overview' or not targets:
            return
        self.run.intent('collect', slots=[list(slot.target.box) for slot in targets])
        for slot in targets:
            self.run.check()
            if time.monotonic()-self.run._captured > 4:
                self.run.block('Sold-slot controls became stale during collection; its intent remains pending.')
            self.run.client.tap(*slot.target.center, width=fresh.width, height=fresh.height)
            self.run.wait(.035)
        self._await_slots(targets, {'empty'})
        self.run.collected += len(targets)
        self.run.confirmed()

    def _select_wheat(self, frame, view, *, maximize_price=False):
        if view.silo_tab:
            fresh, checked = self.current(frame, view)
            if checked.kind != 'composer' or not FarmingWorker._same_target(view.silo_tab, checked.silo_tab, fresh):
                self.run.block('The silo inventory tab changed before selection.')
            self._tap(checked.silo_tab.center, fresh)
            frame, view = self.observe()
        for _ in range(4):
            if view.wheat is not None or view.inventory is None:
                break
            # Inventory sorts by stock: the final few wheat can move below the
            # viewport. Scroll only inside the observed inventory panel.
            fresh, checked = self.current(frame, view)
            if checked.kind != 'composer' or not FarmingWorker._same_target(view.inventory, checked.inventory, fresh):
                self.run.block('The wheat inventory panel changed before scrolling.')
            box = checked.inventory
            x = round(box.x+box.width*.82)
            self.run.client.swipe(x, round(box.y+box.height*.80), x, round(box.y+box.height*.34),
                width=fresh.width, height=fresh.height, duration_ms=300)
            self.run.wait(.35)
            frame, view = self.observe()
        if view.wheat is None:
            self.run.block('Wheat is not recognized in the shop inventory. Open the Silo tab and leave wheat visible.')
        if view.quantity is not None:
            return self._stable_composer()
        fresh, checked = self.current(frame, view)
        if checked.kind != 'composer' or not FarmingWorker._same_target(view.wheat, checked.wheat, fresh):
            self.run.block('The wheat inventory icon moved before selection.')
        self._tap(checked.wheat.center, fresh)
        if maximize_price and checked.layout_verified and checked.max_price is not None:
            # Both positions were verified in this settled dialog. Select wheat
            # and press max as one bounded pair, then read all resulting values.
            self._tap(checked.max_price.center, fresh)
        return self._stable_composer()

    def _stable_composer(self, *, quantity=None, price=None):
        for _ in range(7):
            frame, view = self.observe()
            values = (view.quantity, view.price, view.stock)
            if (view.kind == 'composer' and view.wheat is not None
                    and all(value is not None for value in values)
                    and 1 <= view.quantity <= STACK and 1 <= view.price <= MAX_PRICE
                    and view.stock >= view.quantity
                    and (quantity is None or view.quantity == quantity)
                    and (price is None or view.price == price)):
                # This post-input capture supplies the fresh values for commit.
                return frame, view
            self.run.wait(.1)
        self.run.block('The wheat sale quantity, price, or stock did not become readable and stable.')

    def _increment(self, frame, view, name, count):
        if not 0 < count <= MAX_PRICE:
            self.run.block('Invalid wheat sale adjustment.')
        target = getattr(view, name)
        fresh, checked = self.current(frame, view)
        if (checked.kind != 'composer' or checked.wheat is None or checked.quantity != view.quantity
                or checked.price != view.price or checked.stock != view.stock
                or not FarmingWorker._same_target(target, getattr(checked, name), fresh)):
            self.run.block('The wheat sale changed before adjusting it.')
        # These are bounded taps on a verified quantity/price control. The entire
        # amount is read again before a listing can be committed.
        for _ in range(count):
            self.run.check()
            if time.monotonic()-self.run._captured > 4:
                self.run.block('The wheat price controls became stale while adjusting them.')
            self.run.client.tap(*getattr(checked, name).center, width=fresh.width, height=fresh.height)
            self.run.wait(.035)
        if (name in {'plus_quantity', 'minus_quantity'} and checked.max_price is not None
                and FarmingWorker._same_target(view.max_price, checked.max_price, fresh)):
            # Amount and price share fixed controls. Set both before the next
            # capture; a missed max tap still falls back to the ordinary check.
            self._tap(checked.max_price.center, fresh)
        self.run.wait(.06)
        return self._stable_composer(
            quantity=(view.quantity+count if name == 'plus_quantity' else
                      view.quantity-count if name == 'minus_quantity' else view.quantity),
            price=view.quantity*MAX_PRICE//STACK if name == 'max_price'
                  else view.price+count if name == 'plus_price' else None)

    def _list(self, frame, view, slot):
        opened = self._open_slot(frame, view, slot, 'composer')
        if opened is None:
            return False
        frame, view = self._select_wheat(*opened, maximize_price=True)
        reserve = self.run.state['seed_reserve']
        if view.stock is None:
            self.run.block('Wheat stock is unreadable; a listing cannot preserve the reseeding reserve.')
        quantity = sale_quantity(view.stock, reserve)
        if view.quantity is None or not 1 <= view.quantity <= max(1, quantity):
            # Animated counts can briefly pass number recognition while the
            # composer settles. Nothing has been submitted; observe again.
            self.run.wait(.12)
            frame, view = self._stable_composer()
            quantity = sale_quantity(view.stock, reserve)
        price = quantity*MAX_PRICE//STACK
        if not quantity:
            self.close(frame, view)
            self._stock_empty_until = time.monotonic()+20
            recovery = self.run.state.get('silo_recovery')
            if recovery is not None:
                recovery['surplus_empty'] = True
                self.run.state['wheat_empty'] = True
                self.run.persist()
            return False
        if not 1 <= view.quantity <= quantity:
            if view.minus_quantity is None:
                self.close(frame, view)
                self._stock_empty_until = time.monotonic()+20
                self.run.publish('Wheating: keeping the reseeding reserve; rechecking wheat stock shortly.')
                return False
            frame, view = self._increment(frame, view, 'minus_quantity', view.quantity-quantity)
        if view.quantity < quantity:
            frame, view = self._increment(frame, view, 'plus_quantity', quantity-view.quantity)
        if view.quantity != quantity or view.price is None or not 1 <= view.price <= price:
            self.run.block('The wheat quantity or price could not be read after adjustment.')
        if view.price < price:
            if view.max_price:
                frame, view = self._increment(frame, view, 'max_price', 1)
            else:
                frame, view = self._increment(frame, view, 'plus_price', price-view.price)
        # Handle ads separately through Edit Sale so the five-minute cooldown
        # never turns into a paid skip hidden in the listing composer.
        if view.ad_checked:
            fresh, checked = self.current(frame, view)
            if not checked.ad_checked or not FarmingWorker._same_target(view.newspaper, checked.newspaper, fresh):
                self.run.block('The listing advertisement checkbox could not be rechecked.')
            self._tap(checked.newspaper.center, fresh)
            frame, view = self.observe()
        fresh, checked = self.current(frame, view)
        if (view.kind != 'composer' or checked.kind != 'composer' or view.wheat is None or checked.wheat is None
                or view.quantity != quantity or checked.quantity != quantity
                or view.price != price or checked.price != price
                or sale_quantity(checked.stock, reserve) != quantity or checked.stock != view.stock
                or view.ad_checked or checked.ad_checked or checked.submit is None
                or not FarmingWorker._same_target(view.submit, checked.submit, fresh)):
            self.run.block('The maximum wheat quantity and price were not confirmed in a fresh listing.')
        self.run.intent('list', slot=list(slot.target.box), quantity=quantity, price=price, stock=checked.stock)
        self._tap(checked.submit.center, fresh)
        self._await_slot(slot, {'wheat', 'sold'})
        self.run.listed += quantity
        recovery = self.run.state.get('silo_recovery')
        if recovery is not None:
            recovery['released'] += quantity
            recovery['surplus_empty'] = checked.stock-quantity <= reserve
        self.run.state['wheat_empty'] = checked.stock-quantity <= reserve
        self.run.confirmed()
        if checked.stock-quantity <= reserve:
            # A timer expiring cannot replenish inventory. The next verified
            # harvest/replant updates this when it supplies fresh wheat.
            self._stock_empty_until = float('inf')
        self.run.publish(f'Wheating: listed {quantity} wheat for {price} coins ({self.run.listed} total).')
        return True

    def _advertise(self, frame, view, slot, *, check_ready=False):
        from hayday.wheating_advertising import advertise
        return advertise(self, frame, view, slot, check_ready=check_ready)

    def _advertise_once(self, frame, view, slot, *, check_ready=False):
        from hayday.wheating_advertising import reconcile, submission_after, trace
        opened = self._open_slot(frame, view, slot, 'edit')
        if opened is None:
            trace(self, 'listing_changed')
            return time.time()+30
        frame, view = opened
        trace(self, 'edit_opened', frame, view)
        if view.wheat is None:
            self.run.block('The Edit Sale panel does not confirm wheat.')
        if view.cooldown is not None:
            trace(self, 'cooldown_active', frame, view)
            self.run.state['ad_after'] = max(time.time()+max(2, view.cooldown)+1,
                                            submission_after(self.run))
            self.run.persist()
            self.close(frame, view)
            return
        if not view.ad_free:
            trace(self, 'advertisement_unavailable', frame, view)
            # An existing live advertisement or unrecognized cooldown needs no tap.
            next_check = time.time()+30
            self.run.state['ad_after'] = max(next_check, submission_after(self.run))
            self.run.persist()
            self.close(frame, view)
            return next_check
        if check_ready:
            # A live free control supersedes a cached estimate, but never the
            # protection for an actual (possibly unconfirmed) submission.
            self.run.state['ad_after'] = submission_after(self.run)
            self.run.persist()
            if time.time() < self.run.state['ad_after']:
                trace(self, 'submission_retained', frame, view)
                self.close(frame, view)
                return
        if not view.ad_checked:
            fresh, checked = self.current(frame, view)
            if (checked.wheat is None or not checked.ad_free or checked.cooldown is not None
                    or not FarmingWorker._same_target(view.newspaper, checked.newspaper, fresh)):
                self.run.block('The free advertisement control changed before selection.')
            self._tap(checked.newspaper.center, fresh)
            frame, view = self.observe()
        fresh, checked = self.current(frame, view)
        if (view.kind != 'edit' or checked.kind != 'edit' or checked.wheat is None
                or not checked.ad_free or not checked.ad_checked or checked.cooldown is not None
                or time.time() < self.run.state['ad_after']
                or not FarmingWorker._same_target(view.submit, checked.submit, fresh)):
            self.run.block('A free wheat advertisement was not confirmed; no advertisement was purchased.')
        self.run.state['ad_after'] = time.time()+AD_COOLDOWN+1
        trace(self, 'before_intent', fresh, checked)
        self._ad_before = fresh
        self.run.intent('advertise', slot=list(slot.target.box))
        trace(self, 'before_tap', fresh, checked)
        self._tap(checked.submit.center, fresh)
        trace(self, 'tap_returned')
        if not reconcile(self):
            self.run.block('The shop became obscured after the advertisement attempt.')

    def service(self, deadline=float('inf'), *, listing_limit=None):
        """Service available work; return True when the shop can be left idle."""
        for _ in range(40):
            self.run.check()
            if (self.run.state.get('silo_recovery') or {}).get('surplus_empty', False):
                return True
            if listing_limit is not None and self.run.listed >= listing_limit:
                return True
            if time.monotonic() >= deadline:
                return False
            if self._ready_view is not None:
                frame, view = self.current(*self._ready_view)
                self._ready_view = None
            else:
                frame, view = self.observe()
            if time.monotonic() >= deadline:
                return False
            if view.kind != 'overview':
                if view.kind == 'unknown' and self._dismiss_dialog(frame):
                    continue
                self.run.block('The roadside shop overview is not recognized.')
            # Use already empty crates first. Fast buyers can then be collected
            # together, instead of adding a receipt round trip to every sale.
            empty = next((s for s in view.slots if s.kind == 'empty'), None)
            if empty and time.monotonic() >= self._stock_empty_until:
                if self._list(frame, view, empty):
                    continue
                # A no-surplus check closes its composer. The old overview
                # predates that input and must not supply receipt/ad targets.
                continue
            sold = [s for s in view.slots if s.kind == 'sold']
            if sold:
                if len(sold) > 1:
                    self._collect_many(frame, view, sold)
                else:
                    self._collect(frame, view, sold[0])
                continue
            if time.monotonic() >= deadline:
                return False
            # Listing never waits for an advertisement. Fill the available
            # crates first, then advertise one of the existing wheat listings.
            wheat = next((s for s in view.slots if s.kind == 'wheat' and not s.advertised), None)
            recovery = self.run.state.get('silo_recovery')
            # Before waiting on a full shop, inspect the live ad control even
            # when our cached timer says to wait. Pace further inspections.
            check_ready = (recovery is not None and empty is None
                           and time.time() >= recovery.get('ad_check_after', 0))
            if (wheat and not self.run.state.get('pending')
                    and (check_ready or time.time() >= self.run.state['ad_after'])):
                self._advertise(frame, view, wheat, check_ready=check_ready)
                continue
            self._ready_view = frame, view
            if self.run.state.get('silo_recovery') is not None:
                # A freshly observed shop waiting for buyers is healthy. Do
                # not extend this exemption to unverified input or navigation.
                failsafe = getattr(self.run, '_failsafe', None)
                if failsafe:
                    failsafe.reset_progress()
            return True
        return False

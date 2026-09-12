"""Bounded truck-order sessions with durable intent and visual confirmation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hayday.adb import AdbClient, Screenshot
from hayday.camera import CameraNavigator
from hayday.game_scene import farm_scene_vision
from hayday.order_vision import TICKET_FINGERPRINT_SCHEME, OrderReader, fingerprints_match
from hayday.reconnect import MaintenanceRecovery, ReconnectBlocked, ReconnectRecovery
from hayday.tutorials import TutorialBlocked, TutorialDismissal
from hayday.vision import BoardDetector
from hayday.work_scheduling import retry_delay, utc_timestamp


class OrdersCancelled(RuntimeError):
    pass


class OrdersLimit(RuntimeError):
    pass


class OrdersBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class OrderRunResult:
    status: str
    message: str
    deliveries: int = 0
    frame: Screenshot | None = field(default=None, repr=False)
    diagnostics: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"completed", "deferred", "limit"}


def _atomic_json(path: Path, value: Any) -> None:
    """Persist the exact intent before input; rename only after a durable flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OrderRunner:
    """One fixed-device session; only verified orders and known ad controls act.

    A pending send survives cancellation, command failure, and application exit.
    It must receive positive delivery evidence, or a rewarded attempt must be
    reconciled against the same fully ready order, before a later send. No order
    deletion, premium purchase, or Back fallback exists.
    """

    def __init__(
        self, client: AdbClient, diagnostics_root: Path, *,
        cancel_event: threading.Event | None = None, progress=None,
        max_deliveries: int = 10, max_seconds: float = 1200,
        reader=None, navigator=None, worker=None,
        ad_exit_detector=None, ad_text_reader=None, ad_gate_factory=None, tutorials=None,
        reconnect=None, maintenance=None, launcher=None,
    ):
        if type(max_deliveries) is not int or not 1 <= max_deliveries <= 100:
            raise ValueError("Delivery limit must be an integer between 1 and 100.")
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 7200:
            raise ValueError("Session time limit must be positive and at most 7200 seconds.")
        self.client = client
        self.diagnostics_root = Path(diagnostics_root)
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress = progress
        self.max_deliveries = max_deliveries
        self.max_seconds = max_seconds
        self.reader = reader
        self.navigator = navigator
        self.worker = worker
        self.ad_exit_detector = ad_exit_detector
        self.ad_text_reader = ad_text_reader
        self.ad_gate_factory = ad_gate_factory
        self.tutorials = tutorials
        self.reconnect = reconnect
        self.maintenance = maintenance
        self.launcher = launcher
        self.serial = client.serial
        self.deliveries = 0
        self.frame: Screenshot | None = None
        self.diagnostics: Path | None = None
        self.device_root = self.diagnostics_root / ("device_" + hashlib.sha256(self.serial.encode()).hexdigest()[:16])
        self.state_path = self.device_root / "state.json"
        self.poll_seconds = .8
        self.selection_seconds = .35
        self.ad_min_seconds = 30.0
        self.ad_timeout = 60.0
        self.confirmation_timeout = 35.0
        self.truck_timeout = 150.0
        self._deadline = math.inf
        self._size: tuple[int, int] | None = None
        self._sequence = 0
        self._state: dict[str, Any] = {}
        self._details: dict[str, Any] = {"serial": self.serial}
        self._lock_file = None
        self._receipt_detector = None
        self._receipt_recovery_attempted = False
        self._replacement_seen: dict[str, str] = {}
        self._routine_capture_labels = frozenset({'camera', 'panel', 'resource', 'fruit_inventory', 'receipt_navigation'})
        self._routine_capture_interval = 20.0
        self._routine_png_budget_bytes = 64 * 1024 * 1024
        self._routine_last_saved: dict[str, float] = {}
        self._routine_frames_skipped: dict[str, int] = {}
        self._routine_skip_reasons: dict[str, int] = {}
        self._routine_png_bytes = 0
        self._diagnostic_png_bytes = 0

    def cancel(self):
        self.cancel_event.set()
        self.client.close()

    def _check(self):
        if self.cancel_event.is_set():
            raise OrdersCancelled("Orders session cancelled.")
        if self.client.serial != self.serial or not self.serial:
            raise OrdersBlocked("Device selection changed during the orders session.")
        if time.monotonic() >= self._deadline:
            raise OrdersLimit("Session time limit reached.")

    def _wait(self, seconds: float):
        self._check()
        if self.cancel_event.wait(min(max(0, seconds), max(0, self._deadline-time.monotonic()))):
            self._check()
        self._check()

    def _progress(self, message: str):
        self._details["last_message"] = message
        if self.progress:
            self.progress({"message": message, "deliveries": self.deliveries, "frame": self.frame})

    def _maintenance_wait(self, seconds):
        self._check()
        started = time.monotonic()
        try:
            self.cancel_event.wait(seconds)
        finally:
            # Maintenance downtime does not consume the active farming budget.
            self._deadline += time.monotonic()-started
        self._check()

    def _capture(self, label="observation", *, save=True) -> Screenshot:
        frame = self._capture_runtime(label, save=save)
        return self._recover_capture(frame)

    def _capture_resource(self):
        frame = self._capture_runtime('resource')
        # Location popovers expire while three generic dialog matchers search
        # the same cream-colored order panel. A positively recognized resource
        # popup can be returned promptly; unfamiliar screens still go through
        # all normal recovery handlers on this very capture.
        from hayday.resource_vision import ResourceVision
        vision = getattr(self.worker, 'vision', None)
        if (isinstance(vision, ResourceVision) and self._state.get('pending') is None
                and not any(getattr(recovery, '_uncertain', False)
                            for recovery in (self.maintenance, self.reconnect, self.tutorials))):
            scene = vision.observe(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if len(scene.popups) == 1:
                popup = scene.popups[0]
                if (not popup.rows and len(popup.navigation) == 1) or popup.recipe_complete:
                    return frame
        return self._recover_capture(frame)

    def _recover_capture(self, frame):
        if self.maintenance is None:
            self.maintenance = MaintenanceRecovery(
                self.client, capture=lambda: self._capture_runtime('maintenance_observation', save=False),
                cancel_event=self.cancel_event, check=self._check, wait=self._maintenance_wait,
                save=self._save_frame, progress=self._progress)
        try:
            frame = self.maintenance.process(frame)
        except ReconnectBlocked as exc:
            raise OrdersBlocked(str(exc)) from exc
        self._details['maintenance'] = self.maintenance.events
        # Reconnection is safe only while there is no unresolved delivery/ad
        # receipt. Pending recovery has its own strict observation path and must
        # never gain an unrelated retry tap.
        if self._state.get("pending") is None:
            if self.reconnect is None:
                self.reconnect = ReconnectRecovery(
                    self.client,
                    capture=lambda: self._capture_runtime("reconnect_observation", save=False),
                    cancel_event=self.cancel_event,
                    check=self._check,
                    wait=self._wait,
                    save=self._save_frame,
                    progress=self._progress,
                )
            try:
                frame = self.reconnect.process(frame)
            except ReconnectBlocked as exc:
                raise OrdersBlocked(str(exc)) from exc
            self._details["reconnect"] = self.reconnect.events
        if self.tutorials is None:
            self.tutorials = TutorialDismissal(
                self.client, capture=lambda: self._capture_raw('fuel_tutorial_observation', save=False),
                cancel_event=self.cancel_event, check=self._check, wait=self._wait,
                save=self._save_frame, progress=self._progress,
            )
        self._details['tutorials'] = self.tutorials.events
        # Exact free-fuel dismissal is allowed even with an unresolved delivery.
        # Generic dialog recovery remains disabled for receipt/ad navigation.
        return self.tutorials.process(frame)

    def _capture_runtime(self, label, *, save=True):
        frame = self._capture_raw(label, save=save)
        if self.launcher is None:
            from hayday.launcher import LaunchRecovery
            self.launcher = LaunchRecovery(
                self.client, capture=lambda: self._capture_raw('relaunch_observation', save=False),
                cancel_event=self.cancel_event, check=self._check, wait=self._wait,
                save=self._save_frame, progress=self._progress)
        try:
            frame = self.launcher.process(frame)
        except ReconnectBlocked as exc:
            raise OrdersBlocked(str(exc)) from exc
        self._details['launcher'] = self.launcher.events
        return frame

    def _capture_raw(self, label="observation", *, save=True) -> Screenshot:
        self._check()
        frame = self.client.capture()
        # Keep the latest returned image even if cancellation, a device change,
        # or a changed resolution makes this observation unusable for input.
        self.frame = frame
        self._check()
        size = (frame.width, frame.height)
        if self._size is not None and self._size != size:
            raise OrdersBlocked("Device resolution changed during the orders session.")
        self._size = size
        if save:
            self._save_frame(label, frame)
        return frame

    def _save_frame(self, label: str, frame: Screenshot):
        if not self.diagnostics:
            return None
        # Every observation keeps a unique sequence even when its PNG is
        # sampled out, so JSON observations never overwrite one another.
        self._sequence += 1
        routine = label in self._routine_capture_labels
        now = time.monotonic()
        if routine:
            prior = self._routine_last_saved.get(label)
            reason = ('interval' if prior is not None and now-prior < self._routine_capture_interval
                      else 'byte_budget' if self._diagnostic_png_bytes+len(frame.png) > self._routine_png_budget_bytes
                      else None)
            if reason:
                self._routine_frames_skipped[label] = self._routine_frames_skipped.get(label, 0)+1
                self._routine_skip_reasons[reason] = self._routine_skip_reasons.get(reason, 0)+1
                return None
        safe = "".join(char for char in label if char.isalnum() or char == "_")[:60]
        target = self.diagnostics / f"{self._sequence:04d}_{safe}.png"
        target.write_bytes(frame.png)
        self._diagnostic_png_bytes += len(frame.png)
        if routine:
            self._routine_last_saved[label] = now
            self._routine_png_bytes += len(frame.png)
        return target

    def _create_diagnostics(self):
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.diagnostics = self.diagnostics_root / name
        self.diagnostics.mkdir(parents=True)

    def _observe(self, label="observation", *, frame=None, save=True):
        frame = frame if frame is not None else self._capture(label, save=save)
        observation = self.reader.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        self._observe_replacements(observation)
        if save and self.diagnostics:
            _atomic_json(self.diagnostics / f"{self._sequence:04d}_{label}.json", asdict(observation))
        selected = observation.selected
        inventory_hook = getattr(self.worker, "observe_inventory", None) if self.worker is not None else None
        if (callable(inventory_hook) and observation.panel.verified and selected is not None
                and selected.complete and not selected.obstructed and not observation.bonus_prompt):
            inventory_hook(frame, selected.items)
            self._check()
        return observation

    def _persist(self):
        self._state["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(self.state_path, self._state)

    def _acquire_state(self):
        self.device_root.mkdir(parents=True, exist_ok=True)
        stream = (self.device_root / "session.lock").open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise OrdersBlocked("Another orders session already owns this device state.") from exc
        self._lock_file = stream
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (not isinstance(loaded, dict) or type(loaded.get("version")) is not int
                    or loaded["version"] != 1 or loaded.get("serial") != self.serial):
                raise OrdersBlocked("Saved order state is invalid; no input can be sent.")
            self._state = loaded
            if self._state.get("pending") is not None and not isinstance(self._state["pending"], dict):
                raise OrdersBlocked("Saved pending delivery is invalid; no input can be sent.")
            pending = self._state.get("pending")
            if pending is not None and (
                type(pending.get("slot_id")) is not int or not 0 <= pending["slot_id"] <= 8
                or not isinstance(pending.get("id"), str) or not pending["id"]
                or pending.get("mode") not in {"normal", "rewarded_ad"}
                or pending.get("stage") not in {"prepared", "send_attempted", "ad_started", "ad_close_attempted"}
                or pending.get("mode") == "normal" and pending.get("stage") not in {"prepared", "send_attempted"}
                or pending.get("mode") == "rewarded_ad" and pending.get("stage") == "send_attempted"
                or 'ad_control_kind' in pending and (
                    pending['ad_control_kind'] not in ('close', 'skip')
                    or pending.get('mode') != 'rewarded_ad' or pending.get('stage') != 'ad_close_attempted')
                or not isinstance(pending.get("ticket_fingerprint"), str)
                or not fingerprints_match(pending["ticket_fingerprint"], pending["ticket_fingerprint"])
                or not isinstance(pending.get("order_fingerprint"), str)
                or not fingerprints_match(pending["order_fingerprint"], pending["order_fingerprint"])
            ):
                raise OrdersBlocked("Saved pending delivery fields are invalid; no input can be sent.")
            active = self._state.get("active")
            if active is not None and (not isinstance(active, dict)
                    or type(active.get("slot_id")) is not int or not 0 <= active["slot_id"] <= 8
                    or not isinstance(active.get("ticket_fingerprint"), str)
                    or not fingerprints_match(active["ticket_fingerprint"], active["ticket_fingerprint"])
                    or type(active.get("actions", 0)) is not int or active.get("actions", 0) < 0):
                raise OrdersBlocked("Saved active order is invalid; no input can be sent.")
            parked = self._state.get('parked_orders', [])
            if (not isinstance(parked, list) or len(parked) > 9
                    or any(not self._valid_parked_order(entry) for entry in parked)
                    or len({entry['slot_id'] for entry in parked}) != len(parked)
                    or active and not self._valid_item_waits(active.get('deferred_items', []))):
                raise OrdersBlocked('Saved waiting-order state is invalid; no input can be sent.')
            if not isinstance(self._state.get("receipts", []), list):
                raise OrdersBlocked("Saved delivery receipts are invalid; no input can be sent.")
            receipts = self._state.get("receipts", [])
            if any(not isinstance(receipt, dict) or not isinstance(receipt.get("id"), str)
                   or not receipt["id"] or type(receipt.get("slot_id")) is not int
                   or not 0 <= receipt["slot_id"] <= 8
                   or receipt.get("evidence") not in {"sent_stamp", "replacement"}
                   or not isinstance(receipt.get("ticket_fingerprint"), str)
                   or not fingerprints_match(receipt["ticket_fingerprint"], receipt["ticket_fingerprint"])
                   or "awaiting_replacement" in receipt and type(receipt["awaiting_replacement"]) is not bool
                   for receipt in receipts):
                raise OrdersBlocked("Saved delivery receipt identities are invalid; no input can be sent.")
            receipt_ids = [receipt["id"] for receipt in receipts]
            if len(receipt_ids) != len(set(receipt_ids)) or pending and pending["id"] in receipt_ids:
                raise OrdersBlocked("Saved delivery identities conflict; no input can be sent.")
            if self._state.get('ticket_fingerprint_scheme') != TICKET_FINGERPRINT_SCHEME:
                guarded = any(receipt.get('awaiting_replacement', receipt.get('evidence') == 'sent_stamp')
                              for receipt in receipts)
                if pending or active or parked or guarded:
                    raise OrdersBlocked('Saved unresolved work uses an older ticket identity format; '
                                        'it has been preserved and no new input can be sent.')
                self._state['ticket_fingerprint_scheme'] = TICKET_FINGERPRINT_SCHEME
        else:
            self._state = {"version": 1, "serial": self.serial, "pending": None, "active": None, "receipts": [],
                           'ticket_fingerprint_scheme': TICKET_FINGERPRINT_SCHEME}

    def _release_state(self):
        if self._lock_file:
            stream, self._lock_file = self._lock_file, None
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()

    @staticmethod
    def _ticket(observation, slot_id):
        return next((ticket for ticket in observation.tickets if ticket.slot_id == slot_id), None)

    @staticmethod
    def _same_control(first, second):
        return first is not None and second is not None and first.kind == second.kind and all(
            abs(a-b) <= max(5, first.bounds[2+i]*.08) for i, (a, b) in enumerate(zip(first.center, second.center, strict=True))
        )

    @staticmethod
    def _fulfilled(observation):
        selected = observation.selected
        return bool(observation.panel.verified and selected and selected.complete
                    and not selected.obstructed and selected.items
                    and all(item.status == "fulfilled" and not item.red_quantity for item in selected.items)
                    and not observation.bonus_prompt and not getattr(observation, "navigation", None))

    @classmethod
    def _ready(cls, observation):
        return bool(cls._fulfilled(observation) and observation.selected.ready and observation.send_button)

    def _tap(self, point, message):
        self._check()
        self._progress(message)
        self._check()
        if self.frame is None:
            raise OrdersBlocked("Input has no current screenshot.")
        self.client.tap(*point, width=self.frame.width, height=self.frame.height)
        self._check()

    def _panel(self):
        observation = self._observe("panel")
        if observation.panel.verified:
            return observation
        self._progress("Recovering the order board from the current farm view.")
        if self.navigator is None:
            self.navigator = CameraNavigator(
                self.client, verifier=self.reader.verifier, cancel_event=self.cancel_event,
                capture=lambda: self._capture("camera"), progress=self._progress,
                search_timeout=min(360, max(.1, self._deadline-time.monotonic())),
            )
        result = self.navigator.open_board()
        self._check()
        if not result.success:
            raise OrdersBlocked(result.message)
        observation = self._observe("recovered_panel")
        if not observation.panel.verified:
            raise OrdersBlocked("Recovered board could not be read as an order panel.")
        return observation

    def _select(self, ticket):
        latest = self._observe("before_ticket")
        current = self._ticket(latest, ticket.slot_id)
        if (not latest.panel.verified or latest.bonus_prompt or getattr(latest, "navigation", None)
                or current is None or current.sent or not fingerprints_match(ticket.fingerprint, current.fingerprint)):
            raise OrdersBlocked("Ticket changed before selection; no ticket tap sent.")
        self._tap(current.center, f"Reading order ticket {current.slot_id+1}.")
        previous = None
        for _ in range(6):
            self._wait(self.selection_seconds)
            selected = self._observe("selected_order")
            check = self._ticket(selected, current.slot_id)
            if selected.bonus_prompt or getattr(selected, 'navigation', None):
                raise OrdersBlocked('An unexpected popover appeared while selecting the order.')
            if (not selected.panel.verified or check is None or check.sent
                    or not fingerprints_match(current.fingerprint, check.fingerprint)):
                previous = None
                continue
            if (selected.selected is None or not selected.selected.complete or selected.selected.obstructed
                    or current.ready and self._fulfilled(selected) and not check.ready):
                # A neighboring paper can temporarily cover the selected ready
                # check during its tilt. Observe again; never tap the ticket again.
                previous = None
                continue
            if previous and fingerprints_match(previous.selected.fingerprint, selected.selected.fingerprint):
                return check, selected
            previous = selected
        raise OrdersBlocked('The selected ticket and complete order did not settle into two matching observations.')

    def _prepare_intent(self, ticket, selected, mode):
        if self._state.get("pending"):
            raise OrdersBlocked("An earlier delivery still has an unresolved outcome.")
        if self._guarded_ticket(ticket):
            raise OrdersBlocked("This ticket was already delivered and is awaiting a confirmed replacement.")
        self._receipt_recovery_attempted = False
        pending = {
            "id": uuid.uuid4().hex, "slot_id": ticket.slot_id,
            "ticket_fingerprint": ticket.fingerprint, "order_fingerprint": selected.fingerprint,
            "mode": mode, "stage": "prepared", "created_at": datetime.now(UTC).isoformat(),
        }
        self._state["pending"] = pending
        self._persist()  # This must succeed before the send command is attempted.
        return pending

    def _delivery_evidence(self, observation, pending):
        if not observation.panel.verified or observation.bonus_prompt or getattr(observation, "navigation", None):
            return None
        ticket = self._ticket(observation, pending["slot_id"])
        if ticket is None:
            return None  # Absence alone can be failed recognition.
        if ticket.sent:
            return ("sent_stamp", getattr(ticket, "sent_fingerprint", "") or ticket.fingerprint)
        if ticket.fingerprint and not fingerprints_match(ticket.fingerprint, pending["ticket_fingerprint"]):
            return ("replacement", ticket.fingerprint)
        return None

    def _commit_delivery(self, pending, evidence):
        previous_state = copy.deepcopy(self._state)
        receipt = {**pending, "confirmed_at": datetime.now(UTC).isoformat(), "evidence": evidence[0],
                   "awaiting_replacement": evidence[0] == "sent_stamp",
                   "replacement_fingerprint": evidence[1] if evidence[0] == "replacement" else None}
        history = self._state.get("receipts", []) + [receipt]
        recent = {entry["id"] for entry in history[-100:]}
        self._state["receipts"] = [entry for entry in history
                                   if entry["id"] in recent or self._awaiting_replacement(entry)]
        self._state["pending"] = None
        active = self._state.get("active")
        if (active and active.get("slot_id") == pending["slot_id"]
                and fingerprints_match(active.get("ticket_fingerprint", ""), pending["ticket_fingerprint"])):
            self._state["active"] = None
        if 'parked_orders' in self._state:
            self._state['parked_orders'] = [entry for entry in self._state['parked_orders']
                if not self._same_work_ticket(entry, pending)]
        try:
            self._persist()
        except Exception:
            # Durable storage still contains the unresolved intent. Keep the
            # in-memory result equally conservative if receipt persistence fails.
            self._state = previous_state
            raise
        self.deliveries += 1
        self._details["last_receipt"] = receipt
        self._progress(f"Delivery confirmed. {self.deliveries}/{self.max_deliveries} delivered.")

    @staticmethod
    def _awaiting_replacement(receipt):
        return receipt.get("awaiting_replacement", receipt.get("evidence") == "sent_stamp")

    def _guarded_ticket(self, ticket):
        return any(receipt.get("slot_id") == ticket.slot_id and self._awaiting_replacement(receipt)
                   for receipt in self._state.get("receipts", []))

    def _observe_replacements(self, observation):
        """A completed stamp must not turn into a second send of lingering artwork."""
        receipts = self._state.get("receipts", [])
        if not observation.panel.verified or observation.bonus_prompt or getattr(observation, "navigation", None):
            self._replacement_seen.clear()
            return
        changed = False
        for receipt in receipts:
            if not self._awaiting_replacement(receipt):
                continue
            identity = receipt["id"]
            ticket = self._ticket(observation, receipt.get("slot_id"))
            if (ticket is None or ticket.sent or not ticket.fingerprint
                    or fingerprints_match(ticket.fingerprint, receipt.get("ticket_fingerprint", ""))):
                self._replacement_seen.pop(identity, None)
                continue
            previous = self._replacement_seen.get(identity, "")
            if fingerprints_match(previous, ticket.fingerprint):
                receipt["awaiting_replacement"] = False
                receipt["replacement_fingerprint"] = ticket.fingerprint
                self._replacement_seen.pop(identity, None)
                changed = True
            else:
                self._replacement_seen[identity] = ticket.fingerprint
        if changed:
            self._persist()

    def _farm_visible(self, frame):
        """Positive game HUD evidence is required before searching beyond the view."""
        return farm_scene_vision().ready(frame.png) and not CameraNavigator._modal_visible(frame)

    def _receipt_farm_capture(self):
        # Reward particles or a panel opening can briefly hide the HUD. Do not
        # hand an unknown frame to the navigator, but allow it time to settle.
        for attempt in range(4):
            frame = self._capture('receipt_navigation')
            panel = self.reader.verifier.verify(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if panel.verified or self._farm_visible(frame):
                return frame
            if attempt == 0:
                self._save_frame('receipt_scene_unsettled', frame)
                self._progress('Waiting for the farm interface to settle before further camera input.')
            if attempt < 3:
                self._wait(.35)
        raise OrdersBlocked('The farm disappeared during receipt recovery; camera input stopped.')

    def _recover_receipt_panel(self):
        """Find the board on a verified farm, never search an unknown ad.

        Two farm HUD observations allow bounded zoom and pan when the board is
        offscreen. Every subsequent capture must still show the farm or its order
        panel. Without that HUD evidence, only a twice-verified visible board can
        be opened. Neither path replays a send or assumes delivery succeeded.
        """
        if self._receipt_recovery_attempted or self.frame is None:
            return None
        if self._farm_visible(self.frame):
            self._wait(self.selection_seconds)
            fresh = self._capture('receipt_farm_checked', save=False)
            if not self._farm_visible(fresh):
                return None
            self._receipt_recovery_attempted = True
            self._save_frame('receipt_farm_confirmed', fresh)
            self._progress('The farm is visible; finding the board to check the earlier delivery.')
            navigator = CameraNavigator(
                self.client, verifier=self.reader.verifier, cancel_event=self.cancel_event,
                capture=self._receipt_farm_capture, progress=self._progress,
                allow_dialog_recovery=False,
                search_timeout=min(360, max(.1, self._deadline-time.monotonic())),
            )
            result = navigator.open_board()
            self._check()
            self._details['receipt_recovery'] = {'status': result.status, 'message': result.message,
                                                'actions': list(result.actions)}
            if not result.success:
                raise OrdersBlocked(result.message)
            return self._observe('receipt_panel_recovered')
        if self._receipt_detector is None:
            self._receipt_detector = BoardDetector()
        report = self._receipt_detector.detect(self.frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if report.match is None:
            return None
        fresh = self._capture("receipt_farm_checked", save=False)
        checked = self._receipt_detector.revalidate(
            fresh.png, report.match, cancel=self.cancel_event.is_set
        )
        self._check()
        match = checked.match
        if (match is None or abs(match.tap_x-report.match.tap_x) > max(8, report.match.width*.1)
                or abs(match.tap_y-report.match.tap_y) > max(8, report.match.height*.1)):
            return None
        self._receipt_recovery_attempted = True
        self._save_frame("receipt_farm_confirmed", fresh)
        self._progress("The farm is visible; reopening its verified board to confirm the delivery.")
        navigator = CameraNavigator(
            self.client, detector=self._receipt_detector, verifier=self.reader.verifier,
            cancel_event=self.cancel_event, capture=lambda: self._capture("receipt_navigation"),
            progress=self._progress, max_zoom_steps=0, max_pan_steps=0,
            allow_dialog_recovery=False,
            search_timeout=min(25, max(.1, self._deadline-time.monotonic())),
        )
        result = navigator.open_board()
        self._check()
        if not result.success:
            return None
        return self._observe("receipt_panel_recovered")

    def _confirm_delivery(self, pending, *, timeout=None):
        end = time.monotonic() + (self.confirmation_timeout if timeout is None else timeout)
        previous = None
        while time.monotonic() < end:
            observation = self._observe("delivery_check", save=False)
            if (not observation.panel.verified and
                    (pending["mode"] == "normal" or pending["stage"] == "ad_close_attempted"
                     or self._farm_visible(self.frame))):
                observation = self._recover_receipt_panel() or observation
            evidence = self._delivery_evidence(observation, pending)
            if evidence and previous and evidence[0] == previous[0] and fingerprints_match(evidence[1], previous[1]):
                self._save_frame("delivery_confirmed", self.frame)
                self._commit_delivery(pending, evidence)
                return True
            previous = evidence
            self._wait(self.poll_seconds)
        return False

    def _wait_ad(self, pending, *, resumed=False):
        from hayday.ad_exit import AdExitDetector
        from hayday.ad_text import AdTextReader
        from hayday.ad_watch import AdWatchGate

        self._progress("Checking whether the ad or the farm is visible.")
        detector = self.ad_exit_detector or AdExitDetector()
        text_reader = self.ad_text_reader or AdTextReader(cancel=self.cancel_event.is_set)
        gate_factory = self.ad_gate_factory or AdWatchGate
        loading_started = time.monotonic()
        watch_started = None
        gate = None
        last_text_at = -math.inf
        text = None
        while True:
            frame = self._capture("ad_wait", save=False)
            observation = self.reader.observe(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if observation.panel.verified:
                if self._delivery_evidence(observation, pending):
                    return self._confirm_delivery(pending)
                if resumed:
                    # A recognized game panel on restart is not an ad. Give
                    # its receipt a short settling window, then reconcile it.
                    return self._confirm_delivery(pending, timeout=min(10, self.confirmation_timeout))
                # Ad loading can leave the unchanged order panel visible for
                # several frames. It is not delivery evidence or an ad exit.
                if time.monotonic() - (watch_started or loading_started) >= (
                    self.ad_timeout if watch_started is not None else 30
                ):
                    return False
                self._wait(self.poll_seconds)
                continue
            if self._farm_visible(frame):
                recovered = self._recover_receipt_panel()
                if recovered is not None:
                    return self._confirm_delivery(pending, timeout=(
                        min(10, self.confirmation_timeout) if resumed else None))
                if self._receipt_recovery_attempted:
                    return False
                # A farm frame is never ad-watch evidence. If it changed before
                # confirmation, observe again without using its corner controls.
                self._wait(self.poll_seconds)
                continue
            if gate is None:
                self._progress("Watching the rewarded ad for 30–60 seconds; checking both upper corners.")
                watch_started = time.monotonic()
                gate = gate_factory(initial_elapsed=0)
                # A resumed uncertain ad is observed for a fresh watch window;
                # wall-clock changes or time spent in another app cannot skip it.
                pending["ad_seen_at"] = datetime.now(UTC).isoformat()
                self._persist()
            close = detector.detect(frame.png, cancel=self.cancel_event.is_set)
            now = time.monotonic()
            if now-last_text_at >= 2:
                text = text_reader.read(frame.png, cancel=self.cancel_event.is_set)
                last_text_at = time.monotonic()
            self._check()
            decision = gate.observe(frame.png, close, text=text)
            if decision.status == "close" and close is not None:
                # OCR can take time. Detect the actual input target again in a
                # new screenshot instead of clicking coordinates from before OCR.
                fresh = self._capture("ad_exit_recheck", save=False)
                checked = detector.detect(fresh.png, cancel=self.cancel_event.is_set)
                self._check()
                if (checked is None or checked.corner != close.corner
                        or checked.kind != close.kind
                        or abs(checked.center[0]-close.center[0]) > 6
                        or abs(checked.center[1]-close.center[1]) > 6):
                    self._details["ad_exit"] = {"status": "changed", "message": "Ad close control moved before input."}
                    return False
                confirmed = gate.observe(fresh.png, checked, text=text)
                if confirmed.status != "close":
                    self._wait(self.poll_seconds)
                    continue
                self._save_frame("ad_close_confirmed", fresh)
                self._details["ad_exit"] = asdict(confirmed)
                pending["stage"] = "ad_close_attempted"
                pending["ad_control_kind"] = checked.kind
                pending["ad_exit_reason"] = confirmed.message
                self._persist()
                self._tap(checked.center, "Using the skip control after the required ad watch."
                          if checked.kind == 'skip' else
                          "Closing the watched ad after verifying its exit control.")
                self._wait(self.selection_seconds)
                if checked.kind == 'skip':
                    return self._finish_skipped_ad(pending)
                return self._confirm_delivery(pending)
            if decision.status == "timeout":
                self._details["ad_exit"] = asdict(decision)
                self._progress(decision.message)
                return False
            if close is None and time.monotonic()-watch_started >= self.ad_min_seconds:
                recovered = self._recover_receipt_panel()
                if recovered is not None and self._delivery_evidence(recovered, pending):
                    return self._confirm_delivery(pending)
            if time.monotonic()-watch_started >= self.ad_timeout:
                return False
            self._wait(self.poll_seconds)

    def _finish_skipped_ad(self, pending):
        """A skip can reveal an end card; allow one separately confirmed X.

        The saved skip intent prevents repeating it on restart or transport
        failure. Neither a skip nor an X is itself a delivery receipt.
        """
        from hayday.ad_exit import AdExitDetector
        from hayday.ad_text import AdTextReader
        from hayday.ad_watch import AdWatchGate

        detector = self.ad_exit_detector or AdExitDetector()
        text_reader = self.ad_text_reader or AdTextReader(cancel=self.cancel_event.is_set)
        gate = None
        deadline = time.monotonic()+15
        self._progress('The watched ad was skipped; checking for the final close button or delivery receipt.')
        while time.monotonic() < deadline:
            frame = self._capture('ad_after_skip', save=False)
            observation = self.reader.observe(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if observation.panel.verified:
                return self._confirm_delivery(pending)
            if self._farm_visible(frame):
                recovered = self._recover_receipt_panel()
                if recovered is not None:
                    return self._confirm_delivery(pending)
                if self._receipt_recovery_attempted:
                    return False
                self._wait(self.poll_seconds)
                continue
            candidate = detector.detect(frame.png, cancel=self.cancel_event.is_set)
            if candidate is None or candidate.kind != 'close':
                # Do not repeat the skip, follow a store label, or click a CTA.
                candidate = None
                recovered = self._recover_receipt_panel()
                if recovered is not None:
                    return self._confirm_delivery(pending)
            text = text_reader.read(frame.png, cancel=self.cancel_event.is_set)
            self._check()
            if gate is None:
                gate = (self.ad_gate_factory or AdWatchGate)(initial_elapsed=30)
            decision = gate.observe(frame.png, candidate, text=text)
            if decision.status == 'close' and candidate is not None:
                fresh = self._capture('ad_final_close_recheck', save=False)
                checked = detector.detect(fresh.png, cancel=self.cancel_event.is_set)
                self._check()
                if (checked is None or checked.kind != 'close' or checked.corner != candidate.corner
                        or abs(checked.center[0]-candidate.center[0]) > 6
                        or abs(checked.center[1]-candidate.center[1]) > 6):
                    return False
                fresh_text = text_reader.read(fresh.png, cancel=self.cancel_event.is_set)
                self._check()
                confirmed = gate.observe(fresh.png, checked, text=fresh_text)
                if confirmed.status != 'close':
                    self._wait(self.poll_seconds)
                    continue
                self._save_frame('ad_final_close_confirmed', fresh)
                pending['ad_control_kind'] = 'close'
                pending['ad_final_exit_reason'] = confirmed.message
                self._persist()
                self._tap(checked.center, 'Closing the verified end card after the watched ad.')
                return self._confirm_delivery(pending)
            if decision.status == 'timeout':
                return False
            self._wait(self.poll_seconds)
        self._progress('No verified final close or delivery receipt appeared after the skip; saved intent is preserved.')
        return False

    def _reconcile_unfulfilled_ad(self, pending):
        """Retire a failed ad only when its exact order is still fully sendable.

        Run only after receipt observation has failed. Seeing a farm, elapsed
        time, or a ready check alone cannot clear a delivery intent. This records
        an unfulfilled attempt, never a delivery, and preserves resource work.
        """
        if pending['mode'] != 'rewarded_ad' or pending['stage'] not in ('ad_started', 'ad_close_attempted'):
            return False
        observation = self._observe('ad_reconcile_panel', save=False)
        ticket = self._ticket(observation, pending['slot_id'])
        if (not observation.panel.verified or observation.bonus_prompt or observation.navigation
                or ticket is None or ticket.sent or not ticket.ready
                or not fingerprints_match(ticket.fingerprint, pending['ticket_fingerprint'])):
            return False
        self._progress('The earlier ad returned without a receipt; checking whether that exact order is still ready.')
        ticket, selected = self._select(ticket)
        if (not self._ready(selected)
                or not fingerprints_match(selected.selected.fingerprint, pending['order_fingerprint'])):
            return False
        fresh = self._observe('ad_reconcile_ready', save=False)
        current = self._ticket(fresh, pending['slot_id'])
        if (not self._ready(fresh) or current is None or current.sent or not current.ready
                or not fingerprints_match(current.fingerprint, pending['ticket_fingerprint'])
                or not fingerprints_match(fresh.selected.fingerprint, pending['order_fingerprint'])
                or not self._same_control(selected.send_button, fresh.send_button)):
            return False
        self._save_frame('ad_order_still_ready', self.frame)
        previous_state = copy.deepcopy(self._state)
        recovery = {**pending, 'resolved_at': datetime.now(UTC).isoformat(), 'evidence': 'same_ready_order'}
        self._state['unfulfilled_ad_attempts'] = (self._state.get('unfulfilled_ad_attempts', [])+[recovery])[-100:]
        self._state['pending'] = None
        try:
            self._persist()
        except Exception:
            self._state = previous_state
            raise
        self._details['unfulfilled_ad'] = recovery
        self._progress('The same order and all its goods are still ready. The old ad attempt was unfulfilled; resuming orders.')
        return True

    def _dispatch(self, ticket, selection):
        current = self._observe("before_send")
        current_ticket = self._ticket(current, ticket.slot_id)
        if (not self._ready(current) or current_ticket is None or not current_ticket.ready or current_ticket.sent
                or not fingerprints_match(current_ticket.fingerprint, ticket.fingerprint)
                or not fingerprints_match(current.selected.fingerprint, selection.selected.fingerprint)):
            raise OrdersBlocked("Order readiness changed before sending; no send attempted.")
        if current.double_offer:
            self._tap(current.double_offer.center, "Opening the available 2X delivery offer.")
            self._wait(self.selection_seconds)
            first = self._observe("bonus_prompt")
            self._wait(self.selection_seconds)
            second = self._observe("bonus_prompt_confirmed")
            second_ticket = self._ticket(second, ticket.slot_id)
            if (not first.panel.verified or not second.panel.verified
                    or not self._same_control(first.bonus_prompt, second.bonus_prompt)
                    or second_ticket is None or not fingerprints_match(second_ticket.fingerprint, ticket.fingerprint)):
                # A missing offer may fall back only before ad-send, with the
                # original fully ready order and normal control stable twice.
                first_ticket = self._ticket(first, ticket.slot_id)
                fallback = (
                    first.bonus_prompt is None and second.bonus_prompt is None
                    and self._ready(first) and self._ready(second)
                    and first_ticket is not None and second_ticket is not None
                    and first_ticket.ready and second_ticket.ready
                    and fingerprints_match(first_ticket.fingerprint, ticket.fingerprint)
                    and fingerprints_match(second_ticket.fingerprint, ticket.fingerprint)
                    and fingerprints_match(first.selected.fingerprint, current.selected.fingerprint)
                    and fingerprints_match(second.selected.fingerprint, current.selected.fingerprint)
                    and self._same_control(first.send_button, second.send_button)
                )
                if not fallback:
                    raise OrdersBlocked("The 2X offer did not produce a recognized prompt or a stable normal-send fallback.")
                self._progress("The 2X prompt is unavailable; the same order is ready for normal delivery.")
                return self._normal_dispatch(ticket, second)
            pending = self._prepare_intent(ticket, current.selected, "rewarded_ad")
            self._tap(second.bonus_prompt.center, "Starting the rewarded delivery ad.")
            pending["stage"] = "ad_started"
            self._persist()
            return self._wait_ad(pending)
        return self._normal_dispatch(ticket, current)

    def _normal_dispatch(self, ticket, current):
        pending = self._prepare_intent(ticket, current.selected, "normal")
        self._tap(current.send_button.center, "Sending the fully ready truck order once.")
        pending["stage"] = "send_attempted"
        self._persist()
        return self._confirm_delivery(pending)

    def _ensure_resource_worker(self):
        if self.worker is None:
            from hayday.resources import ResourceWorker
            self.worker = ResourceWorker(
                self.client, capture=self._capture_resource,
                cancel_event=self.cancel_event, progress=self._progress,
                state_path=self.device_root / "resources.json",
            )

    @staticmethod
    def _valid_item_waits(items):
        return (isinstance(items, list) and len(items) <= 9 and all(
            isinstance(item, dict) and type(item.get('index')) is int and 0 <= item['index'] <= 8
            and isinstance(item.get('fingerprint'), str)
            and fingerprints_match(item['fingerprint'], item['fingerprint'])
            and type(item.get('retry_at')) in (int, float)
            and math.isfinite(item['retry_at']) and item['retry_at'] >= 0
            and all(item.get(name) is None or type(item[name]) is int and item[name] >= 0
                    for name in ('available', 'required'))
            for item in items) and len({item['index'] for item in items}) == len(items))

    @classmethod
    def _valid_parked_order(cls, entry):
        return (isinstance(entry, dict) and type(entry.get('slot_id')) is int
            and 0 <= entry['slot_id'] <= 8 and isinstance(entry.get('ticket_fingerprint'), str)
            and fingerprints_match(entry['ticket_fingerprint'], entry['ticket_fingerprint'])
            and type(entry.get('actions', 0)) is int and entry.get('actions', 0) >= 0
            and bool(entry.get('deferred_items')) and cls._valid_item_waits(entry['deferred_items']))

    @staticmethod
    def _same_work_ticket(left, right):
        return (left.get('slot_id') == right.get('slot_id') and fingerprints_match(
            left.get('ticket_fingerprint', ''), right.get('ticket_fingerprint', '')))

    def _parked_order(self, ticket):
        identity = {'slot_id': ticket.slot_id, 'ticket_fingerprint': ticket.fingerprint}
        return next((entry for entry in self._state.get('parked_orders', [])
                     if self._same_work_ticket(entry, identity)), None)

    @staticmethod
    def _item_waiting(active, item):
        return any(entry['index'] == item.index and entry['retry_at'] > utc_timestamp()
            and fingerprints_match(entry['fingerprint'], item.fingerprint)
            and entry.get('available') == item.available and entry.get('required') == item.required
            for entry in active.get('deferred_items', []))

    def _park_active(self):
        active = self._state['active']
        parked = [entry for entry in self._state.get('parked_orders', [])
                  if entry['slot_id'] != active['slot_id']]
        self._state['parked_orders'] = [*parked, copy.deepcopy(active)]
        self._state['active'] = None
        self._persist()
        self._progress(f"Order {active['slot_id']+1} is waiting for resources; checking the next order.")

    def _next_work_ticket(self, tickets):
        due = []
        for ticket in tickets:
            parked = self._parked_order(ticket)
            if parked is None:
                return ticket
            retry_at = min(entry['retry_at'] for entry in parked['deferred_items'])
            if retry_at <= utc_timestamp():
                due.append((retry_at, ticket))
        # Visit untouched orders before revisiting growing work, then service
        # the oldest due item so low ticket numbers cannot starve later orders.
        return min(due, key=lambda entry: entry[0])[1] if due else None

    def _resource(self, ticket, observation):
        selected = observation.selected
        if selected is None or not selected.complete or selected.obstructed:
            raise OrdersBlocked("Cannot work on an incomplete order observation.")
        missing = [item for item in selected.items if item.status == "missing" and item.red_quantity]
        if not missing:
            if all(item.status == "fulfilled" for item in selected.items):
                self._progress("Order goods are ready; waiting for the truck to become available.")
                return "truck"
            raise OrdersBlocked("One or more required goods could not be classified.")
        active = self._state.get("active")
        if not active:
            active = copy.deepcopy(self._parked_order(ticket)) or {
                "slot_id": ticket.slot_id, "ticket_fingerprint": ticket.fingerprint,
                "order_fingerprint": selected.fingerprint, "actions": 0}
            if 'parked_orders' in self._state:
                self._state['parked_orders'] = [entry for entry in self._state['parked_orders']
                                              if entry['slot_id'] != ticket.slot_id]
        elif (active.get("slot_id") != ticket.slot_id
              or not fingerprints_match(active.get("ticket_fingerprint", ""), ticket.fingerprint)):
            raise OrdersBlocked("Existing resource work belongs to a different order and has been preserved.")
        if active.get("actions", 0) >= 48:
            raise OrdersBlocked("The active order reached its bounded resource-action limit.")
        self._state['active'] = active
        eligible = [item for item in missing if not self._item_waiting(active, item)]
        if not eligible:
            self._park_active()
            return 'parked'
        item = eligible[0]
        active["item_fingerprint"] = item.fingerprint
        self._state["active"] = active
        self._persist()
        self._ensure_resource_worker()
        self._progress(f"Working on missing item {item.index+1} for order {ticket.slot_id+1}.")
        query = getattr(self.worker, 'pending_reconciliation', None)
        previous_details = (active.get('last_resource') or {}).get('details', {})
        previous_item = previous_details.get('item') if previous_details.get('pending_harvest') is True else None
        result = query(self.frame, item, item_key=previous_item) if callable(query) else None
        from hayday.resources import ResourceResult
        if isinstance(result, ResourceResult) and result.details.get('inventory_only') is True:
            result = self._reconcile_fruit_inventory(ticket, observation, item, result, active)
        elif not isinstance(result, ResourceResult):
            result = self.worker.work(self.frame, item)
        self._check()
        active["last_resource"] = {"status": result.status, "message": result.message, "details": result.details}
        if result.status in {"collected", "queued"}:
            active["actions"] = active.get("actions", 0)+1
        self._persist()
        self._progress(result.message)
        if result.status in {"unsupported", "changed"}:
            raise OrdersBlocked(result.message)
        if result.status not in {"collected", "queued", "waiting"}:
            raise OrdersBlocked("Resource worker returned an unsupported state.")
        delay = retry_delay(result)
        if delay is not None:
            holds = [entry for entry in active.get('deferred_items', []) if entry['index'] != item.index]
            active['deferred_items'] = [*holds, {
                'index': item.index, 'fingerprint': item.fingerprint,
                'available': item.available, 'required': item.required,
                'retry_at': utc_timestamp()+delay, 'reason': result.message,
            }]
            self._persist()
            if all(self._item_waiting(active, candidate) for candidate in missing):
                self._park_active()
                return 'parked'
            self._progress('This item has work in progress; checking the remaining missing items.')
            return result.status
        if result.details.get("defer_session") is True:
            return "deferred"
        if result.status in {"queued", "waiting"}:
            wait = result.details.get("wait_seconds", 30)
            if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait):
                wait = 30
            self._wait(min(30, max(1, wait)))
        return result.status

    def _reconcile_fruit_inventory(self, ticket, observation, item, result, active):
        """Stay at this order for a bounded stock check; never navigate to its tree."""
        from hayday.resources import ResourceResult

        started = time.monotonic()
        unchanged_frames = set()
        key = result.details.get('item')
        operation = result.details.get('operation')
        for attempt in range(12):
            self._check()
            active['last_resource'] = {'status': result.status, 'message': result.message, 'details': result.details}
            self._persist()
            self._progress(result.message)
            if result.status != 'waiting':
                return result
            if result.details.get('operation') != operation:
                return ResourceResult('unsupported', 'The pending fruit intent changed during stock reconciliation; no resource input was sent.')
            observation_id = result.details.get('observation_id')
            if result.details.get('unchanged_stock') is True and observation_id:
                unchanged_frames.add(observation_id)
            else:
                unchanged_frames.clear()
            elapsed = time.monotonic()-started
            if (elapsed >= 10 and len(unchanged_frames) >= 2) or elapsed >= 30 or attempt == 11:
                message = ('Fruit stock is still unchanged after the harvest confirmation wait. '
                           if len(unchanged_frames) >= 2 else
                           'The fruit harvest could not be confirmed from fresh order stock within the bounded wait. ')
                return ResourceResult('unsupported', message+
                    'The saved harvest intent is preserved. Inspect the tree and stock before restarting; no repeat drag or location visit was sent.',
                    {**result.details, 'wait_elapsed': elapsed, 'unchanged_frames': len(unchanged_frames),
                     'reconciliation_stopped': True})
            self._wait(min(3, 30-elapsed))
            current = self._observe('fruit_inventory')
            current_ticket = self._ticket(current, ticket.slot_id)
            selected = current.selected
            if (not current.panel.verified or selected is None or not selected.complete or selected.obstructed
                    or current.bonus_prompt or getattr(current, 'navigation', None)
                    or current_ticket is None or not fingerprints_match(current_ticket.fingerprint, ticket.fingerprint)
                    or not fingerprints_match(selected.fingerprint, observation.selected.fingerprint)):
                return ResourceResult('unsupported',
                    'The order changed or became covered during fruit stock confirmation. The harvest intent is preserved; no location visit was sent.')
            current_item = next((candidate for candidate in selected.items if candidate.index == item.index
                                 and fingerprints_match(candidate.fingerprint, item.fingerprint)), None)
            if current_item is None:
                return ResourceResult('unsupported', 'The fruit item changed during stock confirmation; its saved harvest intent is preserved.')
            result = self.worker.pending_reconciliation(self.frame, current_item, item_key=key)
            if not isinstance(result, ResourceResult):
                return ResourceResult('unsupported', 'The saved fruit harvest could not be reconciled; no new resource input was sent.')
        raise AssertionError('Fruit inventory reconciliation exceeded its observation bound.')

    def _finish(self, status, message):
        if self.frame is not None:
            try:
                if self.diagnostics is None:
                    self._create_diagnostics()
                final_path = self._save_frame('final', self.frame)
                self._details['final_frame'] = str(final_path)
            except OSError as exc:
                # A full/unwritable disk must not obscure the actual session
                # outcome or trigger another device capture after failure.
                self._details['final_frame_error'] = str(exc)
        self._details['routine_frames_skipped'] = sum(self._routine_frames_skipped.values())
        self._details['diagnostic_images'] = {
            'routine_interval_seconds': self._routine_capture_interval,
            'routine_budget_bytes': self._routine_png_budget_bytes,
            'routine_png_bytes': self._routine_png_bytes,
            'total_png_bytes': self._diagnostic_png_bytes,
            'skipped_by_label': dict(self._routine_frames_skipped),
            'skipped_by_reason': dict(self._routine_skip_reasons),
            'evidence_and_final_exempt': True,
        }
        self._details["pending"] = self._state.get("pending")
        self._details["active"] = self._state.get("active")
        self._details['parked_orders'] = self._state.get('parked_orders', [])
        self._details["state_path"] = str(self.state_path)
        result = OrderRunResult(status, message, self.deliveries, self.frame, self.diagnostics, dict(self._details))
        if self.diagnostics:
            try:
                _atomic_json(self.diagnostics / "result.json", {
                    "status": status, "message": message, "deliveries": self.deliveries,
                    "finished_at": datetime.now(UTC).isoformat(), "details": self._details,
                })
            except (OSError, TypeError, ValueError):
                pass
        return result

    def run(self) -> OrderRunResult:
        self._deadline = time.monotonic() + self.max_seconds
        try:
            self._check()
            self._acquire_state()
            self._create_diagnostics()
            self.reader = self.reader or OrderReader()
            if self.reader.reference_error:
                raise OrdersBlocked(self.reader.reference_error)
            if any((self.device_root / name).is_file()
                   for name in ('resources.json', 'fruit.json', 'orchard.json')):
                # Ready orders after a restart still need to acknowledge stock
                # from saved production/harvest before Send consumes that stock.
                self._ensure_resource_worker()
            pending = self._state.get("pending")
            if pending:
                self._progress("Reconciling an earlier delivery before allowing any new send.")
                if pending.get("mode") == "rewarded_ad":
                    if pending.get("stage") == "ad_started":
                        resolved = self._wait_ad(pending, resumed=True)
                    elif pending.get('stage') == 'ad_close_attempted' and pending.get('ad_control_kind') == 'skip':
                        resolved = self._finish_skipped_ad(pending)
                    else:
                        # A previous ad may still cover the game. Camera search
                        # requires fresh farm evidence; ad-send is not replayed.
                        resolved = self._confirm_delivery(pending, timeout=min(10, self.confirmation_timeout))
                else:
                    resolved = self._confirm_delivery(pending, timeout=min(10, self.confirmation_timeout))
                if not resolved:
                    resolved = self._reconcile_unfulfilled_ad(pending)
                if not resolved:
                    return self._finish("uncertain", "An earlier delivery remains unconfirmed. Its saved intent prevents another send.")
            truck_started = None
            while self.deliveries < self.max_deliveries:
                self._check()
                observation = self._panel()
                if observation.bonus_prompt:
                    raise OrdersBlocked("An unexpected bonus prompt is open; no send will be attempted.")
                if getattr(observation, "navigation", None):
                    raise OrdersBlocked("A resource popover covers the orders; no ticket or send will be tapped.")
                ready = [ticket for ticket in observation.tickets if ticket.ready and not ticket.sent
                         and not self._guarded_ticket(ticket)]
                if ready:
                    ticket, selected = self._select(ready[0])
                    if not self._ready(selected):
                        if self._fulfilled(selected) and selected.send_button is None:
                            truck_started = truck_started if truck_started is not None else time.monotonic()
                            if time.monotonic()-truck_started > self.truck_timeout:
                                raise OrdersBlocked("The truck did not become available within the wait limit.")
                            self._progress("Order goods are ready; waiting for the truck to return.")
                            self._wait(3)
                            continue
                        raise OrdersBlocked("Ticket appeared ready, but its complete selected goods were not verified ready.")
                    if not self._dispatch(ticket, selected):
                        return self._finish("uncertain", "A delivery was attempted once but not confirmed. Saved intent prevents a duplicate send.")
                    truck_started = None
                    continue
                active = self._state.get("active")
                if active:
                    ticket = self._ticket(observation, active["slot_id"])
                    if ticket is None or ticket.sent or not fingerprints_match(ticket.fingerprint, active["ticket_fingerprint"]):
                        raise OrdersBlocked("The active order changed; its resource work has been preserved for review.")
                else:
                    available = [item for item in observation.tickets if not item.sent
                                 and not self._guarded_ticket(item)]
                    ticket = self._next_work_ticket(available)
                    if ticket is None and available:
                        return self._finish('deferred',
                            'All remaining orders are waiting for queued or growing resources. '
                            'Their work is saved; later sessions recheck them when due.')
                if ticket is None:
                    self._progress("Waiting for an available truck-order ticket.")
                    self._wait(5)
                    continue
                ticket, selected = self._select(ticket)
                status = self._resource(ticket, selected)
                if status == "deferred":
                    active = self._state.get("active") or {}
                    message = (active.get("last_resource") or {}).get("message")
                    return self._finish(
                        "deferred",
                        message or "Orders paused while a newly placed orchard plant grows.",
                    )
                if status == "truck":
                    truck_started = truck_started if truck_started is not None else time.monotonic()
                    if time.monotonic()-truck_started > self.truck_timeout:
                        raise OrdersBlocked("The truck did not become available within the wait limit.")
                    self._wait(3)
                else:
                    truck_started = None
            return self._finish("limit", f"Delivery limit reached: {self.deliveries} truck orders confirmed.")
        except Exception as exc:
            if self.cancel_event.is_set() or isinstance(exc, OrdersCancelled):
                return self._finish("cancelled", f"Orders stopped. {self.deliveries} deliveries confirmed.")
            if isinstance(exc, OrdersLimit):
                if self._state.get("pending"):
                    return self._finish("uncertain", f"Session time limit reached with an unconfirmed delivery. {self.deliveries} deliveries confirmed; saved intent prevents a duplicate send.")
                return self._finish("limit", f"Session time limit reached. {self.deliveries} deliveries confirmed.")
            status = "uncertain" if self._state.get("pending") else "blocked" if isinstance(exc, (OrdersBlocked, TutorialBlocked)) else "error"
            return self._finish(status, f"Orders stopped: {exc}")
        finally:
            try:
                self.client.close()
            finally:
                self._release_state()

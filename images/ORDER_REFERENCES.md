# Order-reading references

`order_*.png` contains masked original pixels from the fresh BlueStacks captures in
`reference_captures`. The user-provided explanation images are not used. Source
filenames, original crop boxes and SHA-256 values are in `order_manifest.json`.
Reproduce these assets and their installed-build copies with:

```powershell
.venv\Scripts\python.exe scripts/build_order_references.py --sync-package
```

The reader first verifies the order modal and converts its detected bounds to a
common coordinate system. Ticket checks, item checks, Send, the 2X badge, the
rewarded-ad prompt, sent stamp, and resource-navigation arrow are searched in
their relevant panel regions. All returned rectangles and click centers are
converted back to the actual screenshot's pixels. Ticket indices are row-major
from 0 to 8 in the observed three-column panel layout.

Quantity slash geometry identifies visible item rows, including smaller text in
additional rows. Red available quantities indicate missing goods; a matched green
check confirms fulfillment. Every detected quantity and every matched item check
must agree before an order is considered complete. An independent check for
occupied cells retains visible items with unreadable quantities as `unknown`.
The Send artwork alone never establishes readiness. Popovers prevent a partly
hidden order from being classified as ready.

Ticket fingerprints group all neutral black digit components in each reward row
before deskewing it. This includes disconnected digits such as the 4 and 16 in
416. Selected-order fingerprints combine the title
and rewards with item-body and required-quantity signatures; available counts
are excluded. Compare them with `fingerprints_match`, not string equality.
They are conservative visual identifiers, not database IDs or exact OCR. Sending
requires separate positive evidence: the sent stamp or a stable replacement
ticket. `sent_fingerprint` identifies the stamp while it obscures reward digits.

The session saves send intent before input and requires matching evidence from
two observations before saving a receipt. A stamped ticket remains guarded against
another send until two observations establish its replacement, including across
restarts. Receipt-history trimming retains unresolved replacement guards. If a
delivery returns to the farm, two matching HUD observations allow one zoom-out
followed by bounded overlapping row sweeps to locate the board. Every navigation capture must still show the farm
or verified order panel. Without the HUD, recovery opens only an independently verified
visible board and uses no camera zoom or pan. Existing work for another order stays
in the saved state while a ready order takes priority.

An old rewarded-ad intent can also resolve as unfulfilled after a short receipt
check on the verified game panel: its same ready ticket must be selected, its
complete order identity must match, every item must still be fulfilled, and the
Send control must agree in a fresh observation. The old intent moves to the
`unfulfilled_ad_attempts` audit history before a new send is allowed. This is not
a delivery receipt and does not clear unrelated production or harvest work.

Selection waits for two matching complete observations after one ticket tap.
This allows an adjacent ticket to stop covering the selected green check during
the tilt animation. Visible checks are searched in five-degree rotation steps;
Send is searched at 94–109% of its reference size to cover its observed pulse.
The three `reference_captures/orders_susan_*.png` regressions are untouched copies
of 0019, 0020, and 0023 from the authorized live run
`artifacts/live-order-tests/20260906T170149_7628bfa0`.

Saved state declares the `reward_rows_v2` ticket-fingerprint scheme. State with
no unresolved work can migrate automatically. A pending delivery, active resource
order, or guarded unreplaced receipt from an older scheme is preserved and blocks
new input; changing a fingerprint algorithm must never manufacture a replacement.

The 2X flow falls back to normal sending only when opening the badge produces no
bonus prompt and two observations still show the same fully ready order and Send
control. An unfamiliar popup, failed input command, or attempted ad-send prevents
that fallback.

The order session uses `AdExitDetector` to find an isolated X or double-arrow skip
in either upper corner. X detection requires separation from nearby text; skip
detection requires two filled right-pointing triangles and a terminal bar, and
can handle a colored overlay beside a store label. Both use shape and contrast. It requires no
fixed endcard screenshot. `OrderReader.ad_close` and its older gray circular X
reference remain compatibility/regression helpers; they do not gate the current
session's ad handling.

`AdWatchGate` observes an active ad for at least 30 seconds. After that, a stable
X and either a settled screen or explicit reward-granted text permit a close.
At 60 seconds, a stable X permits a close even if the scene is animated, unless
an observed reward countdown remains unresolved. Missing or ambiguous exits and
unfinished countdowns stop the attempt at the watch limit. The timing branches
are heuristics, not proof that the ad reward was granted.

A skip symbol cannot use screen stillness as completion evidence: it requires
explicit reward-granted text after 30 seconds or the full 60-second watch, with
the same countdown guard. After one skip tap, a separate bounded 15-second step
can use a stable, freshly rechecked X on a settled end card. A new reward countdown
blocks this final close. The saved control kind prevents repeating either tap on
restart or uncertain input; neither tap alone is a delivery receipt.

`AdTextReader` adds optional local Windows English OCR with a bounded timeout.
Learn More, Install, Close, and skip timers are not reward evidence. OCR failure
leaves visual/timing handling available but cannot clear an observed countdown.
The exit is revalidated in a fresh frame before one tap; delivery counting still
requires two observations of a sent stamp or replacement ticket afterward.

The references support the observed game UI artwork/layout. A redesign, unusual
localization geometry, unreadable quantities, or an unfamiliar overlay should
produce an incomplete observation and require further inspection rather than
guessing an action. Product names are not OCRed. Optional numeric inventory totals
use the separate strict original-font references described in `quantities/README.md`.

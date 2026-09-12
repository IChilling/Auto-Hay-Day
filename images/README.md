# Hay Day reference images

This folder contains editable source identifiers. The application reads them here
when running from this checkout; installed builds use matching copies under
`src/hayday/assets` in `identifiers`, `resources`, `farming`, `fruit`, `orchard`, `dialogs`, `launcher`, `animals`, and `quantities`.

| Asset | Purpose |
| --- | --- |
| `quest_board_identifier.png` | Transparent board frame, supports, and ledge used by **Test quest board** |
| `panel_*.png` | Static truck-order panel markers used to confirm opening |
| `panel_manifest.json` | Source size, hash, crop locations, and masks for the panel markers |
| `order_*.png`, `order_manifest.json` | Ticket/check/quantity/Send/2X/ad artwork and provenance |
| `resources/` | Masked navigation arrow and exact EMPTY label, with provenance |
| `farming/` | Observed sickle, drag cues, selected soil, seed-page controls, and growth timer masks |
| `fruit/` | Basket/arrow masks and provenance for the verified cherry-harvest gesture |
| `orchard/` | Raspberry catalog, tab, shop, placement, and invalid-space masks with provenance |
| `dialogs/` | Exact fuel-tutorial, Connection Lost, and Server Offline / Retry Login text groups |
| `launcher/` | Hay Day icon/label and farm HUD markers that distinguish the farm from loading screens |
| `animals/` | Active/gray sheep controls, wooden trough, and ready fleece; excludes decorative lambs |
| `resources/products/` | Complete Blue Sweater artwork, order identifier, title, and provenance |
| `quantities/` | Original-font glyph masks and their source quantity regions |
| `learned_items/` | Runtime item/name crops and source metadata for durable resource identities |
| `reference_captures/` | Original screenshots retained to inspect and regenerate references |
| `ticket_identifier.png` | Supplied ticket reference; not used by the current board test |

Keep original colors in visible template pixels and erase changing content to
transparency. The detector derives its mask from the PNG alpha channel; a separate
black-and-white mask file is unnecessary. The board identifier excludes order
slips, truck, and scenery. Its four spatial regions must retain their original
relative positions, with at least three agreeing before a match can be accepted.
There are no fixed farm coordinates.

The panel markers exclude changing order contents and identify static UI artwork.
They verify panel opening, not truck availability or whether an order can be sent.
The separate order reader verifies all requirements before allowing Send. Resource
icon matches locate candidates; they do not establish that an object is collectible.
See [PANEL_REFERENCES.md](PANEL_REFERENCES.md) for the individual markers and limits.

`reference_captures/resource_violet_dress_*_bright.png` and their JSON provenance
record direct BlueStacks board, location prompt, Sewing Machine, and recipe states.
They cover a brighter rendering of the existing UI and the two-line location label;
the detector calibrates expected palette brightness from matched arrow whites while
preserving the captured RGB in all learned crops.

To regenerate panel references from the original **unscaled 1920 × 1080** capture:

```powershell
.\.venv\Scripts\python.exe scripts\build_panel_references.py images\reference_captures\truck_order_panel.png --sync-package
```

The script writes `panel_*` files to `images/` by default; `--output` selects another
directory and `--sync-package` refreshes the packaged copies. It preserves the board
and ticket identifiers. Inspect the crop specifications before using a capture of
a changed panel layout. When editing the board identifier directly, also copy the
finished PNG to `src/hayday/assets/identifiers/quest_board_identifier.png` for builds.

Other reproducible reference builders, run from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts\build_order_references.py --sync-package
.\.venv\Scripts\python.exe scripts\build_quantity_references.py
.\.venv\Scripts\python.exe scripts\build_dialog_references.py
.\.venv\Scripts\python.exe scripts\build_fruit_references.py
.\.venv\Scripts\python.exe scripts\build_orchard_references.py
```

Order references are cropped original RGB pixels with alpha masks. Quantity
references are isolated masks from actual brown/red glyphs, normalized with their
aspect ratio preserved. The quantity manifest defines the supported glyphs; unseen
or ambiguous digits must return unknown instead of being guessed. See the
[resource developer guide](../docs/RESOURCE_WORKFLOW.md) for precise crops,
provenance requirements, positive/negative examples, and state verification.

Dialog references preserve original RGB with alpha around the actual lettering;
several text groups and their panel layout must agree. The source screenshots are
`reference_captures/dialog_fuel_tutorial.png` and `dialog_connection_lost.png`.
The builder retains their original capture hashes and crop/mask specifications,
then refreshes source and packaged copies. Similar artwork or a standalone
Try Again label is insufficient. `reference_captures/dialog_reconnect_loading.png`
is a negative return-state replay: reconnect must wait past that loading illustration
for playable farm scenery. These references do not authorize other tutorials or advertisements.

Farming references cover the observed selected-crop controls. A crop icon alone
does not identify a harvestable plot. Runtime gestures require fresh tool/target
agreement and seed-stock evidence; source crop coordinates are never farm targets.

Fruit masks retain original pixels that agree across independently captured farm
zooms. This excludes scenery inside the basket handle. `fruit/manifest.json`
records the successful cherry gesture's before/result hashes and its reference
geometry. A basket, matching gold arrow, and selected-object highlight must agree
before input. The guide remaining or disappearing afterward does not establish
collection; two inventory observations must confirm a gain.

Orchard masks preserve the observed Raspberry catalog and placement artwork across
two farm layouts and an idle-animation pose. The manifest records original hashes,
crop coordinates, masks, and the successful live placement's before/result evidence.
Runtime placement uses two current screenshots to rank clear full-size footprints;
the source coordinates never select a farm location. The invalid-space capture is a
negative control, and a saved purchase cannot be repeated until stock evidence retires it.

The active ad workflow uses generic X and double-arrow skip geometry in both upper corners and a bounded
watch rule, with optional local text recognition. It requires no fixed endcard
image. The older `order_ad_close.png` and captured ads remain regression examples;
they do not restrict sessions to one advertisement or one close-button style.
`reference_captures/orders_ad_skip.png` preserves the actual BlueStacks ad capture
with a skip symbol next to the Google Play label for scaled replay tests.
See [ORDER_REFERENCES.md](ORDER_REFERENCES.md) for watch limits and receipt checks.

`reference_captures/farm_zoomed_board_clipped.png` and
`reference_captures/farm_zoomed_out_recovery.png` cover receipt recovery before
and after zooming. The diamond HUD mask uses only the gem's interior facets;
its manifest records the crop's own capture and hash. Grass, buildings behind
the gem, the currency amount, and the user's farm layout are excluded.

`src/hayday/assets/farm-reference.png` is only the labeled reference preview shown
before connecting. It is not a detection template or coordinate map. Recognition
of new artwork, skins, or game redesigns has not been established. Order sessions
can perform bounded camera recovery using current screenshots; reference crop
coordinates are provenance, not saved action coordinates for a particular farm.

`animals/chicken/` contains original-pixel masked chicken-feed and hen artwork,
plus the Egg identifier cropped from Cookie's recipe. `resource_chicken_menu`
and `resource_chicken_relocated` captures cover the old and newly spaced pens.
The basket and arrow reuse the existing fruit assets; the extra paintbrush makes
the chicken menu outline unsuitable as a fruit-menu match. A red comb and an
independently detected fenced soil polygon exclude sheep and lamb lookalikes.
Paths are rebuilt from current pixels; no pen coordinates are saved for input.

`resource_cookie_board`, `resource_cookie_prompt`, and `resource_cookie_source`
replay the short-lived location prompt and Bakery arrival. Their JSON sidecars
record direct ADB provenance and hashes. These full captures are regression
fixtures, not whole-farm templates.

# MuMu Player

## Connect on Windows

1. Start your MuMu Android device and install Hay Day inside it.
2. Launch this project using **Launch Hay Day.cmd**.
3. Open **Emulators**, then **Discover**.
4. Find the MuMu device by name and endpoint, and select **Connect instance**.
   This selects the matching ADB executable, connects its actual port, and captures the screen.
5. Start the desired game workflow.

## Wheat on multiple instances

On **Emulators**, tick **Wheat** beside each named farm, or use **Select all for
wheat**. Then open **Overview** and choose **Start wheating (N)**. Selecting farms
does not require connecting each one manually. **Clear selection** returns to
the existing single-device workflow.

Each selected instance has its own worker thread, ADB connection, crop map,
growth timer, saved actions, and shop state. Overview shows separate progress
and counters. **View** switches the preview; **Stop** on a farm's row stops only
that farm. **Stop wheating** stops all selected farms. Stop before changing the
selection. A disconnected or blocked farm does not interrupt healthy workers.

The application accepts up to 32 selected instances. Four simultaneous MuMu
instances have been tested live; practical capacity depends on host resources.
Lossless raw screen capture reduces transfer/compression overhead, and OpenCV's
internal thread count is bounded while the independent emulator workers overlap
their I/O. Detection caches are separate for each worker.

Discovery reads MuMu's installation location and `MuMuManager.exe info -v all`.
It supports the current `nx_main` layout and the MuMu 12 `shell` layout, including
custom installation folders registered by the installer. Only running Android
instances with valid loopback ADB endpoints appear. Discovery never starts or
reconfigures an emulator.

For a portable installation that is not detected, browse to its `adb.exe`, apply
the path, and Discover again. You can also enter the local ADB endpoint manually.
Use the port reported by your particular instance; do not assume that every
MuMu release or instance uses the same port. Existing saved connections remain
unchanged until you choose a different connection.

## Coexistence with BlueStacks

Both players can remain installed. Use the named **Connect instance** buttons
to select the intended installation and device. Android model names can be
identical across emulators, and `emulator-5554` is not a reliable brand identifier.

MuMu's bundled ADB uses local server port **5038** in this application. BlueStacks
and generic ADB keep the default **5037**. This prevents the bundled ADB versions
from replacing one another's servers. Server port 5038 is separate from the
Android instance endpoint, such as `127.0.0.1:16384`.

When using MuMu's bundled ADB in a terminal alongside this application, include
`-P 5038`. Device commands must also include `-s` with the selected endpoint.

## Input and recovery

Continuous drags and pinch zoom require one discoverable, writable type-B
multitouch screen. The application reads its supported button events, axis
ranges, and display rotation. Unknown/ambiguous input devices, mismatched display
sizes, and unavailable input permissions stop the gesture with an error. The
application does not enable root or change Android input permissions.

MuMu launcher recovery resolves the current Android HOME activity, confirms it
on fresh captures, and launches the installed Hay Day package. It then verifies
the game screen. BlueStacks keeps its existing icon-based recovery. Retry limits,
cancellation, and contact release on failed gestures remain in effect. The
BlueStacks Smart Downloads handler requires a verified BlueStacks guest.

The English Play Games profile invitation is dismissed with **Cancel** after
two fresh captures confirm its text, button arrangement, and Google Play Games
foreground package. This works over either the game or the Android launcher.
Startup then verifies the returned app and positions the field and roadside shop
before crop actions or full-silo sales. Camera recovery uses a bounded zoom and
fresh farm landmarks rather than waiting for the first field or shop failure.
The recorded English level-9 Event Board introduction is completed through its
poster and information screens, with fresh visual confirmation before every tap.
The camera is then prepared again before interrupted crop work resumes. Brief
portrait frames during a known launch are waited out without input; other
resolution changes still stop the session. A fully growing field can establish
the startup workspace when every saved plot and the shop are freshly recognized.

## Field detection

Soil detection searches both supplied furrow orientations alongside the original
soil reference. Reference masks exclude the surrounding grass and white corners,
and overlapping matches are merged before building the planting grid. Field
group detection uses the same references so it can expand across mixed rotations.

The crop picker also recognizes the single-page seed menu shown at early levels.
Planting still requires a selected soil outline and confirmed seed availability.
Live tests cover nine-plot, 15-plot, and irregular 51-plot farms with mixed
orientations, nearby production buildings, and objects overlapping field edges.
The observed bare-soil footprint bounds growth scheduling, preventing overhanging
wheat or neighboring objects from being counted as extra growing plots. The
planting grid also checks furrows inside each candidate tile, rejecting brown
animal-pen floors that happen to align with the field's grid. The
detector measures row spacing instead of assuming a particular farm size;
regressions include layouts larger than 98 plots, holes, and different scales.
The processing bound is 512 visible tiles. Fully hidden soil cannot be inferred
reliably; fresh crop controls and visible tile evidence remain required.
Each visible lattice cell is inspected independently, so a missed cell or a
grass gap cannot cut off another part of the field. Aligned soil observations
are combined; newly revealed cells expand coverage without exceeding confirmed
seed stock. Partially covered picker borders also use the full outline shape
and soil interior to retain the correct tile dimensions.

Harvesting and planting follow the diagonal tile axes. Harvest lanes overlap
and cover every pixel in the detected footprint, with extra coverage around
the verified field grid. The latest wheat image is always inspected, even when
a previous planting map is available. Planting visits exact tile centers and
connects rows through diagonal turns. Route planning uses bounded sampling and
sorting instead of repeated quadratic random searches. ADB accepts up to 2,049
waypoints for the 512-tile bound and its diagonal turns, while retaining its
touch-sample limit, cancellation checks, and contact release.
Crop-arrow searches refresh the complete fan after the camera moves. Before a
full-silo recovery sells wheat from an untracked bare field, it measures that
field and reserves enough seeds to replant it.
The same reserve check protects unfinished field repairs during ordinary shop
visits. After cancellation, two matching observations distinguish planted and
bare tiles before fresh bare-soil work resumes.
Before selecting ripe wheat, the worker takes a new capture after layout and
route calculations, translates the selected point using farm landmarks, and
confirms wheat pixels at that point. The five-second input limit remains in
place; an expired selection can be reobserved up to three times before any tap.
ADB errors and uncertain input outcomes are not retried by this check.
Smoke-darkened selection borders are recognized from their complete shape and
soil interior. If a harvest leaves ripe tiles, two fresh observations and a
verified sickle authorize a bounded repair before replanting. Moving reward
particles cannot authorize that repair, and interrupted attempts remain recorded.
Opening the sickle can obscure those verified crops; their positions are
translated through the freshly verified camera movement for the repair.
Exhausting a repair pass defers remaining ripe tiles while cleared soil is
planted. Partial planting is reconciled per tile across two captures: confirmed
plants are counted, and bare or unstable cells remain scheduled. An obscured
seed-reserve picker retains a conservative reserve and lets the field retry
instead of stopping the session.

The September 2026 reliability regressions include recorded missed tiles,
interrupted seed drags, five image resolutions, irregular large layouts, and
wide or tall viewports. `scripts/verify_wheating_multi.py --audit-crops` records
crop evidence for live diagnosis. Its optional `--miss-drag-once seed` or
`--miss-drag-once harvest`, restricted by `--fault-endpoint`, deliberately
shortens one gesture to exercise recovery. These options are test-only and
are never enabled by the application.

## Wheating and the shop

On **Overview**, choose **Max** or **Min** in the **Sell price** dropdown next to
**Start wheating**. Max is the default (36 coins for 10 wheat); Min lists the
stack for 1 coin. The saved choice applies to all selected farms and partial
stacks, while preserving seeds for replanting. Stop Wheating to change it.

The shop recognizer handles MuMu's title and world-scale differences while
checking the counter and wooden platform independently. Crop controls are
cleared before selecting a shop they cover. A locked shop reports its level-7
requirement without restarting the game.

The Android notification request for Hay Day is declined after fresh visual and
foreground checks, then the workflow resumes. Repeated recognition failures on
the same visible field stop with saved evidence instead of retrying indefinitely.
Recognized achievement animations are allowed to clear before farm input resumes.
Large camera shifts use distributed farm landmarks to bring the shop back into
view, followed by fresh counter and platform checks before opening it.
After shop visits, camera positioning uses the complete verified field footprint
before tracing another harvest. Repeated tree textures cannot establish a camera
shift without a dominant agreement among widely distributed farm landmarks.

The current simultaneous-run results, screenshots, capture benchmarks, and
interruption findings are in [the multi-instance report](artifacts/mumu-multi/report.md).
The subsequent cold-launch and popup tests are in
[the startup report](artifacts/mumu-startup/report.md).
Four-instance recovery, live cycle results, and per-worker timings are in
[the four-instance report](artifacts/mumu-four/report.md).

The level-10 Neighborhood House introduction is recognized by its complete
recorded message and speech-bubble geometry. A fresh confirmation permits one
continuation tap; two clear farm observations must follow before crop work
resumes. Startup, level-up, and reconnection recovery use the same handler, and
the field/shop camera is restored afterward. The normal farm check skips OCR.
Live dismissal, field selection, and shop access are documented in
[the level-10 tutorial report](artifacts/level10-tutorial/report.md).

Scarecrow tutorial recognition normalizes the recorded decorative-font OCR
spellings word by word, so variations such as `have`/`haue` and `the`/`bhe`
can vary independently. The complete known message must still match inside
the speech bubble, followed by fresh confirmation in Hay Day before input.
Both live tutorial stalls and their recovery are documented in
[the tutorial OCR report](artifacts/tutorial-prompts/report.md).

## Validation

### Start orders

Truck-order resource work recognizes the recorded MuMu cherry clusters at both
farm zoom levels. Shape and fruit-color checks use the same neutral cloud-light
correction. Fruit collection uses the discovered touchscreen with a brief hold
on the basket and the drop target. Two fresh positive inventory readings still
confirm each harvest; cancellation, input errors, or unchanged stock retain the
pending intent and prevent another automatic drag.

An exhausted linked apple or cherry tree is identified from the requested fruit and the
saw, help sign, and arrow layout in two fresh frames. That requirement is deferred
for ten minutes while other missing items and tickets continue. This path does
not cut or revive trees, and an existing unconfirmed harvest still takes priority.

Missing items rotate within each order. After each remaining requirement has a
matching deferral, the runner visits another ticket even if navigation has already
outlasted an earlier retry delay. Unconfirmed harvests keep priority until stock
is reconciled, including over older feeding tasks.

Animal collection tools report readiness across every pen of their type. Enabled
blue shears, an orange-handled milk bucket, or a colored egg basket mean at least
one pen has goods ready. Fully greyed-out tools cannot authorize collection.
Arrange matching pens next to each other: the worker groups independently verified
adjacent pens and sweeps each fenced outline, without inspecting individual animals.
Two fresh grey-tool observations record that collection is exhausted; two fresh
inventory gains still confirm what was collected.

For wool, individual sheep poses, fleece artwork and seasonal
accessories are not used. The requested wool icon, sheep-feed menu, wooden
trough and white fence identify the production pen; a sweep stays inside its
verified outline, and inventory gains still require two fresh confirmations.
Sheep-pen recognition measures soil occupancy inside the verified fence outline.
Soil extending past its simplified corners no longer makes occupancy exceed
100 percent and reject the pen when sheep move. A live feed sweep reduced stock
from 1 to 0 in two readings; the next order session recognized growing wool and
continued to the next order.

After an animal harvest is confirmed, feeding is scheduled separately for each
pen included in that collection gesture. The worker relocates the saved pen from
current farm imagery, selects its
trough or coop, and checks the feed bag independently of the harvest tool.
Disabled feed is skipped. Feed-stock decreases confirm each sweep; a pen stays
unfinished while its feed bag remains enabled, including after a partial feed.
If stock runs out, the worker follows that bag's own Feed Mill link and verifies
the ingredients before queuing one batch. Existing batches are reconciled before
another batch or consumption. Restarted sessions retain the pen and any uncertain
feed gesture. Other pens are only requested when their own order work needs them;
there is no whole-farm feeding pass.

Milk collection recognizes the enabled bucket, cow-feed menu, metal trough, and
wooden fence before sweeping complete adjacent cow pens. A greyed-out bucket cannot
authorize collection: its normally orange handle must still be colored, rather
than matching only the silver body shared by both states. If an animal timer
or the viewport hides the enclosure, the worker dismisses the menu and performs
a bounded search before reopening the pen. Collection requires readable stock
before input, saves its intent before the gesture, and waits for two fresh
inventory gains. Feeding has a separate saved intent and requires two matching
stock decreases. Growing or unavailable milk is deferred while other orders run.
Milk identity also has a complete recipe-icon reference without quantity digits,
so it remains recognizable when a recipe exposes more of the can than an older
saved item crop.
When cows cover part of their trough, its visible metal rim and the same complete
wooden fence identify the enclosure. Native references cover the wider camera
view and passing clouds. Individual cow poses and accessories do not determine
readiness: the enabled bucket does. The bucket, cow-feed menu, requested milk icon,
fence, and stock checks still precede collection. World searches omit HUD rows
to stay within the existing freshness limits.
A milk timer may cover the lower fence after opening the pen. Its freshly
observed boundary remains usable only while the metal trough and most of the
enclosed world still match; a shifted camera invalidates it. Decorative arrows
are not required to recognize the enabled bucket.
Menu dismissal excludes the translucent accessories wedge even when grass is
visible underneath it. The worker verifies that the cow controls disappeared
before searching or panning the farm.
The cow menu is also dismissed after a feeding-only visit so its growth timer
does not obstruct recovery of the order board.

Order navigation closes accidentally opened Achievements, County Fair, County
Fair tutorial, catalog and Expand your Farm pages using their exit buttons. Farm expansion
reuses the existing generic Achievements exit reference. Page text and the visible exit
must match recorded references in two fresh frames; an uncleared exit is not
repeated. No achievement reward or County Fair delivery is selected. Recovery
stays disabled during an unresolved truck delivery or advertisement. A dimmed incidental popup can also be closed using
that same original red exit image at the observed scale, masking its background
pixels so shifted layouts remain recognizable. This fallback excludes
the verified order panel, rejects dimmed exits behind another overlay, and stays
disabled while a delivery receipt is unresolved.

Recipe quantities are positioned from the row's stable right edge and height.
An ingredient covering the left background no longer cuts off the first digit
in compact recipes such as butter's milk requirement. Regression captures also
check that brown cloth artwork cannot enter the quantity reading.
Two-line recipe names, including Blueberry Cheesecake, retain their full title
when the initial header crop cuts through the second line. Stock gains observed
before a production-slot revisit are retained across restarts and later truck
deliveries; the batch is retired only after its slot is observed empty.
Before a dependent recipe consumes collected goods, its second stock observation
is saved to the ingredient's production receipt. The recipe and queue are then
captured again to keep the actual production input fresh.
Location and recipe headers use the independently detected action-row bounds.
This preserves the full Red Berry Cake title when its popup connects to the
cream order paper, while excluding the neighboring customer name in the White
Sugar popup.

Fishing now recognizes the requested fish fillet, the spot's lure picker, and
red-lure stock before casting. A held touch follows fresh line observations for
at most 46 seconds. Contact release is guaranteed on cancellation and exceptions.
Fishing captures retain the session deadline, device and resolution checks while
deferring dialog recovery until the touch is released. A saved cast intent must
receive two positive stock observations before another cast is allowed.
Bent fishing lines are followed across connected segments, and a visible catch
ring is required before switching from casting to fish control. Brief line
detection gaps keep the touch held while the catch ring remains visible. A completed
failed cast may be retired only after two unchanged fish-stock readings and two
picker readings showing the same positive lure stock as before the cast.
Interrupted casts retain their original intent. Per-cast tracking evidence records
contact positions, line endpoints, ring centers, and elapsed time.

An empty red-lure supply follows the red lure's own workbench link. Two fresh
observations must verify the free recipe and the first EMPTY queue slot before
one held drag. The durable attempt is confirmed by red-lure artwork in the queue,
not just a disappearing EMPTY label. Existing batches are reused, and interrupted
or unconfirmed production cannot start another batch. Once a confirmed batch
vacates its slot, the worker checks the recognized workbench for collection and
requires two positive stock readings before allowing replacement production.
Workbench references exclude menu arrows, background trees, and the changing
shore behind the queue. Both the open and closed workbench and the queued lure
have recorded checks at the wider camera view used during collection.
Collection allows a bounded wait for swaying lure artwork and closes the
translucent producer menu using a broad, freshly verified open-water patch.

Resting linked fishing spots trigger one water-validated zoom and at most six
inspection attempts for another picker. Moving ripples invalidate individual
inspection points without authorizing a touch. When no picker is verified,
the worker returns to the farm and defers that requirement so other orders can
proceed. Colored lures are not implemented. Unknown stock and unconfirmed catches stop
instead of repeating resource input. The lure workbench is distinguished from
the spot picker by the latter's paging controls. Crop work likewise requires
selected soil, because it shares those controls.

Live validation on the level-33 MuMu farm confirmed cherry stock changing from
0/1 to 1/1, then a feedback-controlled fish catch with fish stock changing from
0/1 to 1/1. A subsequent bounded order session queued missing production and
confirmed one truck delivery (diagnostics `20261001T041123_9b1e244b`). A follow-up
test queued one free red lure, recovered its queue intent after restarting, and
revisited the workbench without adding another batch. After the 90-minute timer,
reopening the verified workbench collected the lure, and two picker readings
confirmed stock 1. The integrated fishing worker then confirmed fish stock
increasing from 1/1 to 2/1 (`artifacts/orders-milk/cast_hold_ring_070050`).
Two earlier failed casts were reconciled using unchanged fish stock and the
returned lure; neither was reported as a catch.
The follow-up session (`20261001T050730_8d278bbc`) advanced past growing wool.
The milk follow-up (`artifacts/orders-milk/milk_work_053807`) then confirmed milk
stock changing from 0/2 to 5/2, cow feed from 6 to 1, and one butter batch queued
after the two recipe confirmations. A one-hour milk growth timer appeared after
feeding. The bounded order session also collected eggs and queued the dependent
recipe. See `artifacts/orders-milk/report.json` for the milk validation evidence.
The next session confirmed butter stock at 1/1 and queued cheese through the
Blueberry Cheesecake dependency chain. Follow-up fishing evidence and regression
results are in `artifacts/orders-next/report.json`. Subsequent live sessions
recognized exhausted apple and cherry sources, deferred those ingredients, and
continued into the Red Berry Cake milk dependency. See
`artifacts/orders-apple/report.json` for the follow-up evidence and remaining limits.

For a bounded live order test, run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_orders_live.py --seconds 180 --deliveries 1
```

The script selects the sole running MuMu instance, or requires `--endpoint` when
several are running. It uses the application's existing device state and session
lock. Optional `--capture-all artifacts/orders-mumu/frames` saves every capture
for debugging; normal application sessions retain their existing capture limits.

### Compatibility and regression checks

The compatibility tests use recorded MuMu Android 15 input descriptors and cover
rotation, gesture release, permissions, instance ports, separate ADB servers,
foreground detection, and bounded launcher recovery. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\ui_smoke.py --hidden
```

Regressions cover both supplied soil orientations, recorded fields, crop-menu
rejection, shop recognition, seed reserves, interrupted-work recovery, concurrent
workers, individual cancellation, selection persistence, and preview isolation.
The hidden native UI smoke test covers 15 route/state combinations, including
an active multi-instance session. See the report for current test totals.

Windows MuMu Android 15 is the live validation target for this change. MuMu 12
installation discovery is covered with fixtures; other Android builds still need
live verification. The macOS installer continues to install BlueStacks Air;
MuMu for macOS is not validated by this Windows change.

Camera recovery requires clear ground at both ends of each swipe. A shorter
swipe is used when the full distance has no clear endpoint, including while
locating a pen for feeding.
Camera-edge checks compare stable farm features so clouds or an accidentally
opened production picker cannot be mistaken for camera movement. Feeding searches
dismiss production controls before another camera gesture and defer an unfound
pen for ten minutes, preserving its request while truck-order work continues.
Clear farm views observed before returning to the truck board can serve as
intermediate references for a saved pen. Each reference must match the original
pen image through a broad farm transform; references are never chained together.
They survive restart and guide camera movement only. Feeding still requires the
same freshly recognized enclosure, tool, and stock checks. Feeding searches also
wait for two distinct, stationary camera observations after each swipe.
The sheep reader checks feed-tool color separately from its stock count. Two
stable observations of grey shears and grey feed defer the wool requirement even
when no growth timer is visible. An earlier feed attempt with unchanged stock
can be reconciled as already fed only when the disabled feed control is also
confirmed in two distinct observations; it does not authorize another sweep.
If farm navigation reaches the fishing area, it reuses the recorded Home button,
confirms it in two fresh frames, taps once, and verifies its disappearance before
resuming from the returned farm view. Normal fishing work retains its own navigation.

Order planting uses a held drag with a brief pause over each selected
soil, matching the input path used by the crop worker. The planting record saves
the original plot and seed image. If a failed attempt leaves the empty picker and
the same seed count visible in two fresh frames, the worker records a no-effect
receipt before allowing another attempt. Changed or unreadable stock keeps the
original intent pending. Cotton planting was verified live with two growth frames
after this recovery; its remaining growth time still defers dependent orders.
Production-menu dismissal excludes the translucent product palette, preventing
ground visible through it from selecting another recipe instead of closing it.
Ready truck deliveries take priority over scheduled feeding searches.
When an order is reread, retry entries for fulfilled or changed requirements are
removed. Their old timestamps can no longer keep that order ahead of other due
tickets while its remaining ingredients are growing. Saved production and
unconfirmed harvest intents are preserved for their separate inventory checks.
The follow-up MuMu run confirmed cotton planting, normal deferral of cotton-dependent
orders, chicken feed consumption from 3 to 0, and the wider Feed Mill menu exit.
Remaining collection and growth dependencies and the latest validation results
are recorded in `artifacts/orders-truck/report.json`.

Order crops now receive the ingredient's actual required quantity. A saved
planting plan accounts for seed consumption, the two-crop harvest yield, crops
already planted, and one seed retained beyond the recipe requirement. Each
additional empty plot must expose a fresh, verified seed picker before its own
planting gesture. Interrupted plans retain their remaining quantity; a growing
plot alone does not satisfy a larger recipe. With only one cotton seed, the first
harvest supplies two seeds for the next planting cycle. Once stock covers the
recipe and seed reserve, harvesting returns to production without spending that
stock on unnecessary replanting.

Crop harvesting starts on the opaque sickle handle and ends near the base of
the full selection outline, including cotton's outline below its white bolls.
Confirmation checks newly exposed soil inside that selected crop; soil revealed
by the disappearing tool palette cannot establish a harvest. Two saved native
post-harvest observations can reconcile a completed harvest after interruption.
The worker relocates the same soil in fresh observations before reopening its
seed picker, without repeating the harvest drag or decrementing growth twice.
An ineffective harvest requires two immediate observations of the unchanged
crop before another attempt becomes eligible.
Seed-picker dismissal excludes its entire translucent palette. Additional plots
can use any soil candidate verified in both fresh observations, so a snowflake
briefly covering the top match does not interrupt the remaining quantity.

Incidental page exits now wait for opening animations to settle. The generic
exit can become an exact Expand Your Farm match as the panel expands; two stable
observations are still required before one exit tap. No new exit image is used.

A corrupt desktop activity database is preserved in place. Startup can write new
activity to `logs/activity-recovered.sqlite3`, with a visible warning, while the
separate order, crop, and feeding state remains available. The bounded order
verifier does not open the desktop activity database.

Production recipe selection requires a readable stock badge beside the matched
palette icon. The same artwork in an occupied production slot is excluded, even
when it has a higher match score. The badge search includes the full height of
stock counters beside partially cropped product icons such as the wool hat.
Brown Sugar collection also verifies the visible output beside two Sugar Mill
features. Blue Woolly Hat collection similarly checks the Loom frame, cloth,
and finished hat. This prevents a small ingredient icon from selecting a similar
piece of another building. Other close-scoring collection matches are deferred.
When the first board tap collects the returned truck's reward, the retry verifies
the known board location in two fresh frames. The same feature thresholds apply
even if a coarse whole-screen search misses the board during the reward animation.

MuMu confirmed expansion-page dismissal, harvesting the initial cotton plot and
planting both resulting seeds into two plots for the three-cotton requirement,
Woolly Chaps collection from 0 to 1 in two inventory
observations, and skipping an already-fed sheep pen. Restart an already-open
desktop tool to load these code changes.

The October 1 follow-up confirmed a School truck delivery after collecting
butter, verifying its stock in fresh observations, watching the available 2X ad,
and recognizing the replacement order. Blue Woolly Hat collection was also
confirmed live: Loom collection changed stock from zero to one in two fresh
inventory observations and retired the pending batch. The latest test results
and native evidence are recorded in `artifacts/orders-truck/report.json`.
The final regression run passed all 656 tests. Both cotton plots have confirmed
growth; harvesting them and queuing the dependent recipe still await maturity.

The sheep follow-up fixes enabled shears being deferred by an individual animal's
growth timer. Ready wool now takes priority over that timer. When the menu hides
the pen, the worker closes it, verifies the complete production enclosure in two
fresh views, and reopens it before shearing. White fence edges and the wooden
trough establish the pen even when full wool coats cover its soil corners.
The clear outline can survive the reopened timer only while the trough and
visible pen pixels still agree. Grey shears continue to prevent collection.
MuMu collection increased wool stock from zero to five, confirmed in two fresh
inventory readings, and queued the dependent product. The follow-up passed 149
related animal, recipe, and inventory tests; native screenshots and results are
in `artifacts/orders-truck/sheep-fix/report.json`. The same run exposed a separate
cotton seed-picker recovery issue after harvest, so the full truck loop still
needs further work. Restart an already-open desktop tool to load the sheep fix.

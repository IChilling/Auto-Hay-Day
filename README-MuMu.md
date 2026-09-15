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

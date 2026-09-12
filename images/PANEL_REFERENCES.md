# Truck-order panel references

The editable source references live directly in this `images` folder. The panel
verifier uses this folder when running from the project checkout. Matching copies
under `src/hayday/assets/identifiers` support installed builds. A missing or invalid
reference causes verification to fail with a useful message.

The four `panel_*.png` files contain original game UI pixels and binary alpha masks:
opaque pixels count, transparent pixels do not. They identify the header's left
wooden edge, close button, lower-left board frame, and upper-left detail-pane edge.
The header text, ticket contents, truck, and farm scenery are excluded. The original
source screenshot is retained in `reference_captures/truck_order_panel.png` for
inspection; the verifier does not match against this full screenshot.

`panel_manifest.json` records the source image size, SHA-256, original crop
coordinates, and mask coordinates. Regenerate after inspecting an **unscaled
1920 × 1080 capture** using:

```powershell
.venv\Scripts\python.exe scripts/build_panel_references.py images/reference_captures/truck_order_panel.png --sync-package
```

The script modifies only generated `panel_*` files. `--output` can direct a trial
generation to another directory; `--sync-package` refreshes the installed-build
copies. Original quest-board and ticket identifiers are preserved.

Verification requires three spatially separated markers to agree at the same
scale and position, plus the expected golden divider/bottom border, wooden order
area, and cream detail area. It works without any specific order slips or English
text and does not infer whether the truck is home, whether goods are available,
or whether an order can be sent. Recognition covers the observed modal style;
future game redesigns or a different panel arrangement may require new references.

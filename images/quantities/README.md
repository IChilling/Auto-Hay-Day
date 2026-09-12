# Quantity glyph references

These masks are derived from the original brown/red quantity characters in local
BlueStacks captures. They cover `/` and digits **0, 1, 2, 3, 4, 5, 6, 7**. Digits
8, 9 and unfamiliar shapes remain unknown until a clearly labeled live capture is
added. The reader does not substitute a similar known digit or use Windows OCR.

`manifest.json` records the source screenshot, its SHA-256 and dimensions, the
complete labeled quantity, each character's position, and its original component
rectangle. Masks preserve aspect ratio within a 40 × 32 canvas. Their color is
only a shape mask; original screenshots remain in `images/reference_captures`.

Rebuild from the inspected samples with:

```powershell
.venv\Scripts\python.exe scripts/build_quantity_references.py
```

The script refreshes the manifest-listed installed-build copies. Restart the
application after updating references because templates are cached during a run.

`read_quantity(png, (x, y, width, height))` returns `Quantity(available, required)`
only when every glyph passes both an absolute shape threshold and a margin over
the next candidate. It also checks a common baseline, one slash, valid numeric
fields, and a positive requirement. Invalid screenshots, bounds, separators,
leading zeros, or unrecognized shapes return `None`.

Order items expose optional `available` and `required` fields from this strict
reader. Numeric totals help verify partial inventory increases; green checks and
complete item observations remain sufficient to establish order readiness when
counts are unknown.

The separate `outlined_*.png` group comes from the white stock counters with
black outlines in field, production, and chicken-feed captures.
It covers **0, 1, 2, 3, 4, 5, 6, 9** and the exact **11** ligature whose two black
outlines touch. The single 1 is cropped from the first digit of that inspected
11, with another native 1 from the chicken-feed counter. These templates are never substituted for the brown/red quantity font.

`read_count(png, (x, y, width, height), font='outlined')` returns an integer,
including **0** for a positively recognized zero. Unclear counts return `None`.
The caller must locate the actual stock bubble. Every component must match a
known shape; the reader does not split touching digits into guessed numbers.
Separated known digits can form longer counts.

The outline mask uses pixel value below 90 and retains enclosed dark components
such as the center of 0. Matching permits one pixel of outline variation after
normalization. Exact deterministic reference resamplings cover tested screenshot
scales 0.66–1.5; 0.5-scale outlines fragment and are deliberately rejected. The
manifest records the source crop and resize scale, and component bounds are in
that resampled screenshot's coordinate space. The source SHA-256 always refers
to the untouched original capture.

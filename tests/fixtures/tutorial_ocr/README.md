# MuMu tutorial OCR stalls, September 14, 2026

Captured directly from the two stopped workers at 1920 × 1080. `text.json`
contains their actual local Windows OCR lines, bounds, foreground package, and
the pre-fix detector results. No device input was sent during capture.

- `neighborhood.png`: endpoint 16448. The existing recorded message contained
  `have`; this capture reads `haue`, causing launch readiness to reject the popup.
- `event_info.png`: endpoint 16480. The existing recorded message contained
  `the inpo bubbon bo`; this capture reads `bhe inpo bubbon bo`, causing the
  Event Board sequence to stop with an unrecognized next page.

The regressions replay these recorded readings without depending on Windows OCR
availability or nondeterministic OCR results during tests.

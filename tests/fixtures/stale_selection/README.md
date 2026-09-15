# Expired wheat selection on MuMu, September 14, 2026

`ripe.png` is the saved terminal screenshot from Android Device-1, endpoint
127.0.0.1:16416, run `20260914T182706_fdd58a`. The farm is clear and ripe, but
the worker stopped before input with "The observed control became stale before
input" followed by "Field recovery made no progress". Its crop intent was empty.

The freshness regressions use this frame and a simulated monotonic clock to
cover aged and superseded captures without sending emulator input.

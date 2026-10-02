Fruit collection references
===========================

The basket and arrow are shared controls. The requested item determines whether
the worker searches for ripe Cherry or Raspberry artwork; recognizing the basket
alone never authorizes a drag.

The MuMu cherry references include two native small-canopy clusters from
`tests/fixtures/orders/cherry_small.png`. The shared RGB gain/offset fitted to
each fruit pattern also corrects its color check, so neutral cloud haze does not
reject an otherwise matching ripe cluster. Other hues cannot be corrected
independently. Recorded cloudy, small-canopy, wrong-item, and hue regressions
are covered by `tests/test_orders_fruit.py`.

The MuMu live comparison found that Android's synthetic swipe could move the
camera without collecting fruit. A held touchscreen contact on the basket,
followed by the verified target and a brief drop hold, produced an observed
cherry inventory change from 0/1 to 1/1. Runtime still requires two positive
stock observations before retiring the durable harvest intent.

Raspberry references include recipe and order-board item variants and two native
berry patterns. They match at world scales independent of the tool menu, require
ripe color and supporting fruit or textured foliage, and remain beyond the
detected arrow. This allows the compact bush beside the translucent guide without
assuming a farm layout. The target is freshly verified before each gesture.

`manifest.json` records original capture names, crop boxes, hashes, masks, and the
live Raspberry inventory change from 0/1 to 1/1. Source coordinates describe asset
provenance only. Runtime positions are detected in the current screenshot.

Rebuild with `.venv\Scripts\python.exe scripts/build_fruit_references.py` from the
project root. The builder preserves native RGB and mirrors assets into
`src/hayday/assets/fruit`. Two positive inventory readings confirm a harvest;
until then, the durable intent prevents repeating an uncertain basket drag.

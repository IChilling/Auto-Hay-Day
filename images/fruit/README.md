Fruit collection references
===========================

The basket and arrow are shared controls. The requested item determines whether
the worker searches for ripe Cherry or Raspberry artwork; recognizing the basket
alone never authorizes a drag.

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

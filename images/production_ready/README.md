# Finished production references

The Cookie and Bakery cutouts use original pixels from our direct BlueStacks
captures in `images/reference_captures/resource_cookie_ready*.png`. Alpha includes
only each feature; opaque white means included and transparent means excluded.
No user attachment is used as a runtime reference.

The manifest relates the finished product to two stable machine features and a
tap point on the machine body. These are object-relative coordinates, independent
of the farm's layout. The detector searches several scales, dismisses the
production menu if it obscures output, and rechecks the machine before clicking.
Inventory must increase in two subsequent observations before the batch clears.

Cookie/Bakery is currently the validated combination. Add observed output and
machine cutouts with their provenance for other combinations; empty queue slots
or a product icon alone do not establish that a machine is ready to collect.

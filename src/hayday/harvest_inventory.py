"""Positive stock confirmation shared by fruit and fishing harvests."""

def observe_harvest_inventory(worker, key, status, available, required, observation_id):
    """Accept two consistent positive stock observations, including partial gains."""
    worker._check()
    entry = worker.state['items'].get(key)
    if not entry or entry.get('stage') != 'attempted':
        return
    before = entry.get('inventory_before', {})
    same_requirement = type(required) is int and required > 0 and before.get('required') == required
    increased = (same_requirement and type(available) is int
                 and type(before.get('available')) is int and available > before['available'])
    fulfilled = same_requirement and status == 'fulfilled' and before.get('status') == 'missing'
    if not observation_id or not (increased or fulfilled):
        if entry.get('inventory_confirmation'):
            worker._record(key, inventory_confirmation=None)
        return
    evidence = [status, available, required]
    confirmation = entry.get('inventory_confirmation') or {}
    if confirmation.get('observation_id') == observation_id:
        return
    count = confirmation.get('count', 0)+1 if confirmation.get('evidence') == evidence else 1
    worker._record(key, inventory_confirmation={'evidence': evidence, 'count': count,
                                              'observation_id': observation_id})
    if count >= 2:
        worker._record(key, stage='confirmed', inventory_confirmation=None,
                     confirmed_available=available, confirmed_required=required)

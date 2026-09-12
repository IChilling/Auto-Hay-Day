"""Replace an interrupted harvest only after freshly selecting mature wheat."""
from __future__ import annotations

import uuid

from hayday.camera import CameraNavigator
from hayday.wheating_selection import PreparedHarvest


def can_reselect(fields, key, frame):
    worker, run = fields.worker, fields.run
    entry = worker.state['items'][key]
    # Seed gestures and Silo Full have separate, durable recovery paths.
    if (entry.get('stage') != 'harvest_attempted'
            or any(entry.get(name) for name in ('replant_evidence', 'plant_before', 'points', 'replanted'))
            or run.state.get('pending') or run.state.get('silo_recovery') is not None):
        return False
    worker._bind_size(frame)
    before = worker._saved_frame(entry.get('harvest_before'))
    if before is None or worker.vision.harvest(before.png) is None:
        return False
    return (run.vision.farm(frame) and not CameraNavigator._modal_visible(frame)
            and bool(run.vision.plots(frame, 'ripe')))


def retire_reselected(fields, key, frame):
    """Archive the old attempt before the ordinary worker records a new one.

    The selector's two fresh sickle/crop observations and current sweep gate
    this transition. No old coordinates or seed gesture are replayed. A crash
    after retirement is harmless: the new harvest has not been sent yet.
    """
    worker, run = fields.worker, fields.run
    entry = worker.state['items'].get(key)
    prepared = worker._prepared_harvest
    if (entry is None or entry.get('stage') != 'harvest_attempted'
            or any(entry.get(name) for name in ('replant_evidence', 'plant_before', 'points', 'replanted'))
            or run.state.get('pending') or run.state.get('silo_recovery') is not None
            or not isinstance(prepared, PreparedHarvest)
            or not prepared.current(worker, frame) or not prepared.path):
        run.block('The interrupted harvest has no fresh verified wheat selection; its intent was retained.')
    run.check()
    archive = uuid.uuid4().hex
    proof = worker._evidence(frame, archive, 'harvest_reselected')
    run.save_json(worker.state_path.parent/'farming_evidence'/f'{archive}_superseded_harvest.json',
                  {'previous_key': key, 'previous': entry, 'current_wheat': proof,
                   'reason': 'Fresh mature wheat selected for a new harvest; old gesture not replayed.'})
    state = {**worker.state, 'items': {k: v for k, v in worker.state['items'].items() if k != key}}
    run.save_json(worker.state_path, state)
    worker.state = state
    fields._retry_harvest_key = None
    run.publish('Wheating: verified mature wheat after an interrupted harvest. Starting a fresh field sweep.')

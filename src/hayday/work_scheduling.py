"""Scheduling hints for confirmed work; these never settle action intent."""

import math
from datetime import UTC, datetime


def utc_timestamp() -> float:
    return datetime.now(UTC).timestamp()


def retry_delay(result) -> float | None:
    """Only positively queued/growing work may yield to another requirement."""
    details = result.details
    if result.status not in {"queued", "waiting"} or not isinstance(details, dict):
        return None
    if (details.get("placement_mode") or details.get("inventory_only")
            or details.get("pending_harvest")
            or details.get("pending_stage") in {"uncertain", "attempted", "placement_attempted"}):
        return None
    orchard = details.get("orchard_pending") is True and details.get("pending_stage") == "growing"
    if details.get("defer_item") is not True and not orchard:
        return None
    delay = details.get("retry_after_seconds", 300 if orchard else 60)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay):
        delay = 60
    return min(600., max(1., float(delay)))

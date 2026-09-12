"""Read visible barn capacity and full-storage notices without guessing free space."""
import re
from dataclasses import dataclass

from hayday.ad_text import AdTextReader


@dataclass(frozen=True)
class StorageStatus:
    used: int | None = None
    capacity: int | None = None
    full: bool = False
    error: str | None = None

    @property
    def free(self):
        return None if self.used is None or self.capacity is None else max(0, self.capacity-self.used)


def parse_storage(text):
    normalized = re.sub(r'\s+', ' ', text).lower()
    full = bool(re.search(r'\b(?:your )?barn (?:is )?full\b|\bstorage is full\b', normalized))
    amounts = re.findall(r'\bbarn(?: storage)?\s*:?\s*(\d{1,5})\s*/\s*(\d{1,5})\b', normalized)
    if len(amounts) == 1:
        used, capacity = map(int, amounts[0])
        if 0 <= used <= capacity and capacity > 0:
            return StorageStatus(used, capacity, full or used == capacity)
    return StorageStatus(full=full)


class StorageReader:
    def __init__(self):
        self.reader = AdTextReader(timeout_seconds=2)

    def read(self, png, cancel):
        observed = self.reader.read(png, cancel=cancel)
        if observed.error:
            return StorageStatus(error=observed.error)
        return parse_storage(observed.text)

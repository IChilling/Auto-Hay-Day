"""Local settings, captures, and activity for the desktop application.

No farm layout or game state is inferred by this module. Everything it stores is
application configuration or an explicitly recorded session event.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENDPOINT = re.compile(r"^(?:127\.0\.0\.1|localhost|\[::1\]):([0-9]{1,5})$", re.IGNORECASE)
_SERIAL = re.compile(r"^[A-Za-z0-9\[][A-Za-z0-9_.:\[\]-]{0,127}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class Settings:
    """User-editable settings; constructing an invalid instance raises ValueError."""

    adb_path: str = ""
    endpoint: str = "127.0.0.1:5555"
    selected_serial: str = ""
    preview_interval_seconds: float = 3.0
    farm_name: str = "My farm"
    wheating_instances: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate and normalize fields, including instances edited after creation."""
        keys = self.wheating_instances
        if (not isinstance(keys, (list, tuple)) or len(keys) > 32
                or any(not isinstance(k, str) or not k or len(k) > 1024
                       or any(ord(c) < 32 for c in k) for k in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError('wheating_instances must contain unique emulator instance keys.')
        self.wheating_instances = tuple(keys)
        for name in ("adb_path", "endpoint", "selected_serial", "farm_name"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be text.")
            if any(ord(character) < 32 for character in value):
                raise ValueError(f"{name} cannot contain control characters.")
            setattr(self, name, value.strip())

        endpoint_match = _ENDPOINT.fullmatch(self.endpoint)
        if not endpoint_match or not 1 <= int(endpoint_match.group(1)) <= 65535:
            raise ValueError("endpoint must be a loopback address and port, such as 127.0.0.1:5555.")
        self.endpoint = self.endpoint.lower()
        if self.selected_serial and not _SERIAL.fullmatch(self.selected_serial):
            raise ValueError("selected_serial must be a valid ADB device identifier (up to 128 characters).")
        interval = self.preview_interval_seconds
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 1 <= interval <= 30:
            raise ValueError("preview_interval_seconds must be a number between 1 and 30.")
        self.preview_interval_seconds = float(interval)
        if not self.farm_name or len(self.farm_name) > 80:
            raise ValueError("farm_name must contain between 1 and 80 characters.")


@dataclass(frozen=True)
class Activity:
    id: int
    timestamp: str
    level: str
    category: str
    message: str


def _timestamp_name() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]


def _default_root() -> Path:
    configured = os.environ.get("HAYDAY_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local = Path(local_app_data).expanduser()
        # Packaged Windows apps can expose a package-local LOCALAPPDATA path.
        # Keep automation state in the user's stable Local directory so a
        # packaged launch and a checkout launch share one recovery history.
        parts = list(local.parts)
        package_index = next((i for i, part in enumerate(parts)
                              if part.casefold() == 'packages'), None)
        if package_index is not None and package_index > 0:
            local = Path(*parts[:package_index])
        return local / "HayDayAutomation"
    return Path.home() / ".local" / "share" / "HayDayAutomation"


class AppData:
    """Application data owned by one process, with thread-safe SQLite access.

    The default location is ``%LOCALAPPDATA%/HayDayAutomation`` on Windows.
    ``HAYDAY_DATA_DIR`` can select an isolated directory for development/testing.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root if root is not None else _default_root()).expanduser().resolve()
        self.settings_path = self.root / "settings.json"
        self.captures_dir = self.root / "captures"
        self.exports_dir = self.root / "exports"
        self.logs_dir = self.root / "logs"
        self.warnings: list[str] = []
        self._lock = threading.RLock()
        self._closed = False
        for directory in (self.root, self.captures_dir, self.exports_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(self.logs_dir / "activity.sqlite3", check_same_thread=False, timeout=10)
        self._database.row_factory = sqlite3.Row
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS activity ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "level TEXT NOT NULL, category TEXT NOT NULL, message TEXT NOT NULL)"
        )
        self._database.commit()
        if not self.settings_path.exists():
            try:
                self.save_settings(Settings())
            except OSError as exc:
                self.warnings.append(f"Could not create settings; using defaults: {exc}")

    def load_settings(self) -> Settings:
        """Read the latest settings, recovering invalid files without losing them.

        Missing fields receive defaults. Invalid individual fields receive defaults
        while valid fields are retained. An invalid original is backed up before
        its repaired replacement is written; a backup failure leaves it untouched.
        Read/write errors are available through ``warnings`` and never prevent a
        default configuration being returned.
        """
        with self._lock:
            defaults = Settings()
            try:
                raw: Any = json.loads(self.settings_path.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                self._save_recovered_settings(defaults)
                return defaults
            except (json.JSONDecodeError, UnicodeError) as exc:
                self.warnings.append(f"Settings could not be read; using defaults: {exc}")
                self._recover_settings(defaults)
                return defaults
            except OSError as exc:
                self.warnings.append(f"Settings could not be read; using defaults: {exc}")
                return defaults

            if not isinstance(raw, dict):
                self.warnings.append("Settings must be a JSON object; using defaults.")
                self._recover_settings(defaults)
                return defaults

            accepted = asdict(defaults)
            invalid = False
            for field in fields(Settings):
                if field.name not in raw:
                    continue
                try:
                    candidate = Settings(**{**accepted, field.name: raw[field.name]})
                except (ValueError, TypeError) as exc:
                    invalid = True
                    self.warnings.append(f"Invalid setting {field.name}; using its default. {exc}")
                else:
                    accepted = asdict(candidate)
            settings = Settings(**accepted)
            if invalid:
                self._recover_settings(settings)
            return settings

    def _recover_settings(self, settings: Settings) -> None:
        backup_path = self.root / f"settings.corrupt.{_timestamp_name()}.json"
        try:
            self.settings_path.replace(backup_path)
        except OSError as exc:
            self.warnings.append(f"Could not preserve invalid settings; original file was left untouched: {exc}")
            return
        self.warnings.append(f"Original settings preserved at {backup_path}")
        self._save_recovered_settings(settings)

    def _save_recovered_settings(self, settings: Settings) -> None:
        try:
            self.save_settings(settings)
        except OSError as exc:
            self.warnings.append(f"Could not save recovered settings: {exc}")

    def save_settings(self, settings: Settings) -> None:
        """Atomically persist valid settings; never replace a file with a partial write."""
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance.")
        settings.validate()
        with self._lock:
            self._atomic_json(self.settings_path, asdict(settings))

    @staticmethod
    def _atomic_json(destination: Path, value: Any) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Application data is closed.")

    def log(self, level: str, message: str, category: str = "app") -> None:
        if not all(isinstance(value, str) for value in (level, category, message)):
            raise TypeError("Activity level, category, and message must be text.")
        with self._lock:
            self._ensure_open()
            with self._database:
                self._database.execute(
                    "INSERT INTO activity(timestamp, level, category, message) VALUES (?, ?, ?, ?)",
                    (datetime.now(UTC).isoformat(timespec="milliseconds"), level.strip().upper(), category.strip(), message),
                )

    def recent_activity(self, limit: int = 200, query: str = "") -> list[Activity]:
        """Return newest records first, optionally matching a literal search string."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer.")
        if not isinstance(query, str):
            raise ValueError("query must be text.")
        with self._lock:
            self._ensure_open()
            sql = "SELECT id, timestamp, level, category, message FROM activity"
            parameters: list[Any] = []
            if query:
                literal = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                sql += " WHERE (message LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\' OR level LIKE ? ESCAPE '\\')"
                parameters.extend([f"%{literal}%"] * 3)
            sql += " ORDER BY id DESC LIMIT ?"
            parameters.append(limit)
            return [Activity(**dict(row)) for row in self._database.execute(sql, parameters).fetchall()]

    def export_activity(self) -> Path:
        """Export the complete activity history as a UTF-8 JSON array."""
        with self._lock:
            self._ensure_open()
            rows = self._database.execute(
                "SELECT id, timestamp, level, category, message FROM activity ORDER BY id DESC"
            ).fetchall()
            destination = self.exports_dir / f"activity_{_timestamp_name()}.json"
            self._atomic_json(destination, [dict(row) for row in rows])
            return destination

    def save_capture(self, png: bytes) -> Path:
        """Store an explicit screenshot without assuming any fixed game dimensions."""
        if not isinstance(png, bytes) or not png.startswith(_PNG_SIGNATURE):
            raise ValueError("Capture must contain PNG image bytes.")
        with self._lock:
            destination = self.captures_dir / f"capture_{_timestamp_name()}.png"
            with destination.open("xb") as stream:
                stream.write(png)
            return destination

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._database.close()
                self._closed = True

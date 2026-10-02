"""A corrupt diagnostic log must not prevent the desktop from opening."""

import sqlite3

import pytest

from hayday.storage import AppData


def test_corrupt_log_is_preserved_and_replacement_persists(tmp_path):
    logs = tmp_path/'logs'
    logs.mkdir()
    original = logs/'activity.sqlite3'
    damaged = b'corrupt original activity log'
    original.write_bytes(damaged)
    data = AppData(tmp_path)
    assert any('preserved' in warning for warning in data.warnings)
    data.log('info', 'retained', 'test')
    data.close()
    assert original.read_bytes() == damaged
    restored = AppData(tmp_path)
    assert restored.recent_activity()[0].message == 'retained'
    restored.close()
    assert original.read_bytes() == damaged


def test_locked_log_does_not_silently_create_another_database(tmp_path, monkeypatch):
    class Locked:
        def execute(self, *args):
            error = sqlite3.OperationalError('database is locked')
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error

        def close(self):
            self.closed = True

    database = Locked()
    monkeypatch.setattr(sqlite3, 'connect', lambda *args, **kwargs: database)
    with pytest.raises(sqlite3.OperationalError, match='locked'):
        AppData(tmp_path)
    assert database.closed
    assert not (tmp_path/'logs/activity-recovered.sqlite3').exists()

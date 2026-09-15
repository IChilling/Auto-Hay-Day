"""One independent, cancellable wheat worker per explicitly selected instance."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cv2

from hayday.adb import AdbClient, AdbError
from hayday.wheating import WheatingResult, WheatingRunner


@dataclass(frozen=True)
class WheatingFleetResult:
    results: dict[str, WheatingResult]

    @property
    def success(self):
        return all(result.success for result in self.results.values())

    @property
    def status(self):
        return 'cancelled' if self.success else 'blocked'

    @property
    def message(self):
        failed = sum(not result.success for result in self.results.values())
        return (f'Wheating stopped on {len(self.results)} instance(s).'
                + (f' {failed} need attention; see their status.' if failed else ''))

    @property
    def planted(self):
        return sum(r.planted for r in self.results.values())

    @property
    def listed(self):
        return sum(r.listed for r in self.results.values())

    @property
    def collected(self):
        return sum(r.collected for r in self.results.values())

    @property
    def advertisements(self):
        return sum(r.advertisements for r in self.results.values())


class WheatingFleet:
    def __init__(self, instances, root, *, progress=None, cancel_event=None,
                 max_seconds=None, reset_restart_cooldown=False,
                 client_factory=AdbClient, runner_factory=WheatingRunner):
        self.instances = tuple(instances)
        if not self.instances or len(self.instances) > 32:
            raise AdbError('Select between 1 and 32 emulator instances for Wheating.')
        endpoints = [i.endpoint.lower().replace('localhost:', '127.0.0.1:') for i in self.instances]
        if len(set(endpoints)) != len(endpoints):
            raise AdbError('The same emulator endpoint was selected more than once.')
        self.root, self.progress = Path(root), progress
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.max_seconds, self.reset_restart_cooldown = max_seconds, reset_restart_cooldown
        self.client_factory, self.runner_factory = client_factory, runner_factory
        self._lock = threading.RLock()
        self._runners = {}
        self._clients = {}
        self._stops = {i.key: threading.Event() for i in self.instances}
        self._sequence = 0
        self.states = {i.key: dict(key=i.key, name=i.display_name, serial=i.endpoint,
            status='starting', message='Connecting…', planted=0, listed=0, collected=0,
            advertisements=0) for i in self.instances}

    def _publish(self, instance, update):
        with self._lock:
            self.states[instance.key].update(update)
            self._sequence += 1
            states = {key: dict(value) for key, value in self.states.items()}
            payload = dict(sequence=self._sequence, instance_key=instance.key,
                instances=states, message=f'{instance.display_name}: {update.get("message", "")}',
                **{key: sum(s.get(key, 0) for s in states.values())
                   for key in ('planted', 'listed', 'collected', 'advertisements')})
        if self.progress:
            self.progress(payload)

    def cancel(self, key=None):
        with self._lock:
            keys = [key] if key is not None else list(self._stops)
            for selected in keys:
                if selected in self._stops:
                    self._stops[selected].set()
                runner = self._runners.get(selected)
                if runner is not None:
                    runner.cancel()
                elif selected in self._clients:
                    self._clients[selected].close()
            if key is None:
                self.cancel_event.set()

    def _worker(self, instance):
        client = None
        stop = self._stops[instance.key]
        try:
            if stop.is_set() or self.cancel_event.is_set():
                result = WheatingResult('cancelled', 'Stopped before connecting.')
            else:
                client = self.client_factory(instance.adb_path, serial=instance.endpoint, timeout_seconds=30)
                with self._lock:
                    self._clients[instance.key] = client
                    if stop.is_set() or self.cancel_event.is_set():
                        client.close()
                        raise AdbError('Stopped before connecting.')
                client.connect(instance.endpoint)
                if not any(d.online and d.serial == instance.endpoint for d in client.list_devices()):
                    raise AdbError('This emulator endpoint is not online.')
                runner = self.runner_factory(client, self.root,
                    cancel_event=stop, progress=lambda u: self._publish(instance, {'status': 'running', **u}),
                    max_seconds=self.max_seconds, reset_restart_cooldown=self.reset_restart_cooldown)
                with self._lock:
                    self._runners[instance.key] = runner
                    if stop.is_set() or self.cancel_event.is_set():
                        runner.cancel()
                result = runner.run()
        except Exception as exc:
            result = WheatingResult('cancelled' if stop.is_set() else 'blocked', str(exc))
        finally:
            if client is not None:
                client.close()
            with self._lock:
                self._runners.pop(instance.key, None)
                self._clients.pop(instance.key, None)
        self._publish(instance, {key: value for key, value in result.__dict__.items()
                                 if key != 'diagnostics'})
        return result

    def run(self):
        # OpenCV otherwise starts a host-sized pool inside every Python worker.
        # Bound its internal parallelism while independent ADB I/O overlaps.
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=len(self.instances), thread_name_prefix='wheating') as pool:
            futures = {pool.submit(self._worker, instance): instance.key for instance in self.instances}
            pending = set(futures)
            try:
                while pending:
                    _, pending = wait(pending, timeout=.2)
                    if self.cancel_event.is_set():
                        self.cancel()
            finally:
                if pending:
                    self.cancel()
            return WheatingFleetResult({key: future.result() for future, key in futures.items()})

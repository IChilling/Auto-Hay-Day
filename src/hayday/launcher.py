"""Relaunch the installed Hay Day icon only on a confirmed BlueStacks home screen."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hayday.dialogs import DialogFeature, DialogObservation, DialogVision
from hayday.reconnect import ReconnectBlocked, ReconnectRecovery
from hayday.resource_vision import ResourceVision, _decode


class LauncherVision:
    def __init__(self, reference_path=None):
        source = Path(__file__).resolve().parents[2]/'images/launcher'
        root = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/launcher')
        self.manifest = json.loads((root/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported launcher references.')
        self.references = {name: _decode((root/spec['file']).read_bytes(), True)
                           for name, spec in self.manifest['features'].items()}
        self.matcher = ResourceVision()
        self.feature_matcher = DialogVision()

    def observe(self, png, cancel=lambda: False):
        try:
            image = _decode(png)
        except ValueError:
            return None
        # Only a cheap rejection; icon, label, and foreground launcher are all
        # required before input. An icon displayed inside an ad is insufficient.
        if np.mean(image.max(axis=2) < 90) < .3:
            return None
        base = image.shape[0]/self.manifest['reference_height']
        scales = np.unique(np.r_[np.geomspace(base*.5, base*1.6, 23), base])
        icon_ref = self.references['hay_day_icon']
        icons = self.matcher._search(image, icon_ref, scales, .95, cancel, max_peaks=4)
        found = []
        for icon in icons:
            scale = icon.width/icon_ref.shape[1]
            # Keep the small label at native resolution; coarse full-screen
            # matching loses thin launcher-font strokes.
            labels = [self.feature_matcher._feature(image, self.references['hay_day_label'],
                        self.manifest['features']['hay_day_label']['box'], icon,
                        self.manifest['features']['hay_day_icon']['box'], value)
                      for value in (scale, (icon.width-1)/icon_ref.shape[1],
                                    (icon.width+1)/icon_ref.shape[1])]
            label = max((m for m in labels if m is not None), key=lambda m: m.score, default=None)
            if label is not None:
                bounds = (min(icon.x, label.x), icon.y, max(icon.width, label.width),
                          label.y+label.height-icon.y)
                found.append(DialogObservation('app_launcher', icon, bounds,
                    min(icon.score, label.score), (DialogFeature('hay_day_icon', icon),
                                                 DialogFeature('hay_day_label', label))))
        return found[0] if len(found) == 1 else None


class LaunchRecovery(ReconnectRecovery):
    def __init__(self, *args, clock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = clock or time.monotonic
        self._launches = []
        self._dialog_vision = None

    def _observe(self, frame):
        self._check()
        if self.vision is None:
            self.vision = LauncherVision()
        result = self.vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        if result is None:
            return None
        if self.client.foreground_package() != 'com.uncube.launcher3':
            return None
        self._check()
        return result

    def _returned_game(self, frame):
        if self.client.foreground_package() != 'com.supercell.hayday':
            return False
        if self._possible_playable_scene(frame) and not self._possible_connection(frame):
            return True
        if self._dialog_vision is None:
            self._dialog_vision = DialogVision()
        dialog = self._dialog_vision.observe(frame.png, cancel=self.cancel_event.is_set)
        return dialog is not None and dialog.kind in {'server_maintenance', 'connection_lost', 'fuel_tutorial'}

    def _wait_for_launch(self):
        self._launches = [stamp for stamp in self._launches if self.clock()-stamp < 3600]
        if len(self._launches) >= 3:
            raise ReconnectBlocked('Hay Day repeatedly returned to the launcher; the launch limit was reached.')
        while self._launches and self.clock()-self._launches[-1] < 60:
            self.progress('Hay Day returned to the home screen. Waiting before another launch attempt.')
            self.wait(min(10., 60-(self.clock()-self._launches[-1])))
            self._check()

    def _record_launch(self):
        self._launches.append(self.clock())

    def process(self, frame):
        self._check()
        if self._uncertain:
            raise ReconnectBlocked('An earlier Hay Day launch is uncertain; no repeated icon tap was sent.')
        observed = self._observe(frame)
        if observed is None:
            return frame
        self.save('launcher_detected', frame)
        for _ in range(3):
            self._wait_for_launch()
            frame, confirmed = self._confirm(frame, observed)
            if confirmed is None:
                if self._returned_game(frame):
                    return frame
                raise ReconnectBlocked('The home screen changed before Hay Day could be relaunched.')
            self._check()
            if self.client.foreground_package() != 'com.uncube.launcher3':
                raise ReconnectBlocked('The foreground app changed before the Hay Day icon tap.')
            self._uncertain = True
            self._record_launch()
            self.events.append({'kind': 'app_launcher', 'stage': 'launch_attempted',
                                'captured_at': frame.captured_at})
            self.save('launcher_before_tap', frame)
            self.progress('Opening the confirmed Hay Day app icon after a crash.')
            self.client.tap(*confirmed.target.center, width=frame.width, height=frame.height)
            self._check()
            self._uncertain = False
            deadline = self.clock()+60
            ready_count = 0
            observed = None
            while self.clock() < deadline:
                previous = frame
                frame = self._fresh(previous)
                if not self._fresh_capture(previous, frame):
                    raise ReconnectBlocked('App relaunch requires fresh screenshots.')
                ready_count = ready_count+1 if self._returned_game(frame) else 0
                if ready_count >= 2:
                    self.events.append({'kind': 'app_launcher', 'stage': 'launched',
                                        'captured_at': frame.captured_at})
                    self.save('launcher_recovered', frame)
                    return frame
                observed = self._observe(frame)
                if observed is not None and self.clock()-self._launches[-1] >= 10:
                    break
            if observed is None:
                raise ReconnectBlocked('Hay Day did not reach a recognized game screen after relaunch.')
        raise ReconnectBlocked('Hay Day could not remain open after three launch attempts.')

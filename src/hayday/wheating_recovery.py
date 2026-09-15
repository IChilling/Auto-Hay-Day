"""Reuse exact game recovery controls without slowing ordinary wheat frames."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from hayday.dialogs import DialogVision
from hayday.launcher import LauncherVision, LaunchRecovery
from hayday.reconnect import ReconnectBlocked, ReconnectRecovery
from hayday.resource_vision import ResourceVision, _decode
from hayday.tutorials import TutorialBlocked, TutorialDismissal
from hayday.wheating_event_board import WheatEventBoard
from hayday.wheating_level_up import WheatLevelUp
from hayday.wheating_neighborhood import WheatNeighborhood
from hayday.wheating_notifications import WheatNotifications
from hayday.wheating_play_games import WheatPlayGames
from hayday.wheating_popup import WheatHostPopup
from hayday.wheating_silo import WheatSiloFull
from hayday.wheating_transient import WheatAchievement


class WheatServerDialog(DialogVision):
    def __init__(self):
        root = Path(__file__).parent/'assets/wheating/recovery'
        spec = json.loads((root/'manifest.json').read_text('utf-8'))
        self.specifications = {'connection_lost': spec}
        self.references = {'connection_lost': [_decode((root/f['file']).read_bytes(), True)
                                             for f in spec['features']]}
        self._matcher = ResourceVision()


class WheatConnectionVision:
    def __init__(self):
        self.server, self.original = WheatServerDialog(), DialogVision()

    def observe(self, png, cancel=lambda: False):
        return self.server.observe(png, cancel) or self.original.observe(png, cancel)


class WheatLauncherVision:
    def __init__(self):
        self.current = LauncherVision(Path(__file__).parent/'assets/wheating/launcher')
        self.original = LauncherVision()

    def observe(self, png, cancel=lambda: False):
        return self.current.observe(png, cancel) or self.original.observe(png, cancel)


class WheatLaunchRecovery(LaunchRecovery):
    """Use Wheating's persistent cooldown for both restarts and launcher retries."""
    def __init__(self, run, **options):
        super().__init__(run.client, **options)
        self.run = run

    def _wait_for_launch(self):
        if self.run._failsafe:
            self.run._failsafe.before_launch()
        else:
            super()._wait_for_launch()

    def _record_launch(self):
        self.run._workspace_needs_restore = True
        self.run._workspace_zoom_attempted = False
        self.run._launch_rotation_until = time.monotonic()+15.
        if self.run._failsafe:
            self.run._failsafe.launch_attempted()
        super()._record_launch()
        self._launches = self._launches[-3:]

    def _returned_game(self, frame):
        if super()._returned_game(frame):
            return True
        recovery = getattr(self.run, '_recovery', None)
        return (self.client.foreground_package() == 'com.supercell.hayday'
                and any(handler is not None and handler.observe(frame) is not None
                        for handler in (getattr(recovery, 'event_board', None),
                                        getattr(recovery, 'neighborhood', None))))

    def process(self, frame):
        restart = self.run._failsafe
        if restart is None:
            return super().process(frame)
        active = restart.active
        restart.active = True  # A deliberate cooldown is not stalled farming.
        try:
            result = super().process(frame)
            if result is not frame:
                restart.reset_progress()
            return result
        finally:
            restart.active = active


class WheatRecovery:
    def __init__(self, run):
        self.run = run
        self.host_popup = WheatHostPopup(run)
        self.notifications = WheatNotifications(run)
        self.play_games = WheatPlayGames(run)
        self.event_board = WheatEventBoard(run)
        self.neighborhood = WheatNeighborhood(run)
        self.level_up = WheatLevelUp(run)
        self.silo_full = WheatSiloFull(run)
        self.achievement = WheatAchievement(run)
        options = dict(capture=run._capture_raw, cancel_event=run.cancel_event,
                       check=run.check, wait=run.wait, progress=run.publish, save=self.save)
        self.launch = WheatLaunchRecovery(run, **options)
        self.launch.vision = WheatLauncherVision()
        dialogs = WheatConnectionVision()
        self.launch._dialog_vision = dialogs
        self.tutorial = TutorialDismissal(run.client, **options)
        self.reconnect = ReconnectRecovery(run.client, ready=self.ready, vision=dialogs, **options)

    def ready(self, frame):
        # Login may return directly to the tutorial. Recognize it here; the
        # normal tutorial helper still owns its fresh confirmation and dismissal.
        return (self.run.vision.farm(frame) or self.level_up.observe(frame) is not None
                or self.event_board.observe(frame) is not None
                or self.neighborhood.observe(frame) is not None
                or self.tutorial._observe(frame) is not None)

    def save(self, label, frame):
        if self.run.diagnostics:
            (self.run.diagnostics/(label+'.png')).write_bytes(frame.png)

    def possible(self, frame):
        # These coarse masks only reject ordinary gameplay. Recovery still
        # requires the existing exact text/icon features on fresh captures.
        image = self.run.vision._image_for(frame.png)
        small = cv2.resize(image, (640, round(640*frame.height/frame.width)),
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        launcher = float(np.mean(cv2.inRange(hsv, (100, 90, 0), (130, 255, 95)) > 0)) > .35
        if not launcher and not self.run.vision.farm(frame):
            launcher = self.launch._package_home() is not None
        cream = cv2.inRange(hsv, (15, 0, 160), (45, 125, 255))
        cream = cv2.morphologyEx(cream, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(cream)
        height, width = cream.shape
        dialog = any(w > width*.35 and h > height*.22 and area > width*height*.12
                     for _, _, w, h, area in stats[1:count])
        if dialog and self.run.vision.shop(frame).kind in {'overview', 'composer', 'edit'}:
            dialog = False
        return launcher, dialog

    def process(self, frame):
        try:
            frame = self.achievement.process(frame)
            frame = self.silo_full.process(frame)
            frame = self.introductions(frame)
            launcher, dialog = self.possible(frame)
            if launcher and not self.run.state.get('pending'):
                frame = self.launch.process(frame)
                frame = self.introductions(frame)
                _, dialog = self.possible(frame)
            if dialog:
                if not self.run.state.get('pending'):
                    frame = self.reconnect.process(frame)
                frame = self.tutorial.process(frame)
                # Reconnection can land directly on a level-up or introduction.
                frame = self.introductions(frame)
            return self.level_up.process(frame)
        except (ReconnectBlocked, TutorialBlocked) as exc:
            self.run.block(str(exc))

    def introductions(self, frame):
        frame = self.level_up.process(frame)
        frame = self.event_board.process(frame)
        return self.neighborhood.process(frame)

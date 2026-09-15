"""Advance only recorded Event Board introduction pages on fresh game frames."""
import re
from pathlib import Path

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import ResourceVision, VisualTarget, _decode
from hayday.wheating import WheatingPrerequisite
from hayday.wheating_tutorial_panel import scarecrow_panel, tutorial_message

PAGES = {
    'congratulations you just unlocked the event board': 'unlocked',
    'here you can find information about events tap on the highlighted poster to unveil what s inside': 'posters',
    'tap on the info button to show more details and rewards': 'info',
    'remember to check back every day for new and exciting events': 'complete',
}


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip()


class WheatEventBoard:
    def __init__(self, run):
        self.run = run
        self.uncertain = None
        self.refs = None
        self.close_ref = None
        self.ribbon_ref = None
        self.info_ref = None

    def menu_control(self, frame):
        image = self.run.vision._image_for(frame.png)
        header = image[:round(frame.height*.18), round(frame.width*.25):round(frame.width*.78)]
        hsv = cv2.cvtColor(header, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 90, 100), (12, 255, 255))
        if float((red > 0).mean()) < .45:
            return None
        text = self.run.vision.text.read(frame.png)
        titles = [line for line in text.lines if normalized(line.text) == 'event board' and line.bounds
                  and frame.width*.35 < line.bounds[0] < frame.width*.65
                  and line.bounds[1]+line.bounds[3] < frame.height*.15]
        if text.error or len(titles) != 1:
            return None
        if self.ribbon_ref is None:
            self.ribbon_ref = _decode((Path(__file__).parent/'assets/wheating/event_board/ribbon.png').read_bytes(), True)
        poster = image[round(frame.height*.55):round(frame.height*.90), round(frame.width*.06):round(frame.width*.24)]
        pink = cv2.inRange(cv2.cvtColor(poster, cv2.COLOR_BGR2HSV), (155, 100, 100), (179, 255, 255))
        if float((pink > 0).mean()) > .85:
            left_panel = image[round(frame.height*.20):round(frame.height*.55), :round(frame.width*.34)]
            ribbons = ResourceVision()._search(left_panel, self.ribbon_ref,
                np.linspace(.88,1.12,25)*frame.height/1080, .85, self.run.cancel_event.is_set, 2)
            if len(ribbons) == 1:
                t = ribbons[0]
                return 'unveil_poster', VisualTarget(t.x,t.y+round(frame.height*.20),t.width,t.height,t.score)
            return None
        if self.info_ref is None:
            self.info_ref = _decode((Path(__file__).parent/'assets/wheating/event_board/info.png').read_bytes(), True)
        left, top = round(frame.width*.22), round(frame.height*.26)
        infos = ResourceVision()._search(image[top:round(frame.height*.40), left:round(frame.width*.32)],
            self.info_ref, np.linspace(.97,1.03,7)*frame.height/1080, .94, self.run.cancel_event.is_set, 2)
        if len(infos) == 1:
            t = infos[0]
            return 'open_details', VisualTarget(left+t.x,top+t.y,t.width,t.height,t.score)
        if self.close_ref is None:
            self.close_ref = _decode((Path(__file__).parent/'assets/wheating/event_board/close.png').read_bytes(), True)
        left = round(frame.width*.80)
        crop = image[:round(frame.height*.18), left:round(frame.width*.95)]
        hits = ResourceVision()._search(crop, self.close_ref, np.linspace(.97,1.03,7)*frame.height/1080,
                                        .94, self.run.cancel_event.is_set, 2)
        if len(hits) == 1:
            t = hits[0]
            return 'close_board', VisualTarget(left+t.x,t.y,t.width,t.height,t.score)
        return None

    def details(self, frame):
        image = self.run.vision._image_for(frame.png)
        center = image[round(frame.height*.25):round(frame.height*.55), round(frame.width*.36):round(frame.width*.76)]
        cream = cv2.inRange(cv2.cvtColor(center, cv2.COLOR_BGR2HSV), (15, 10, 195), (35, 85, 255))
        if float((cream > 0).mean()) < .80:
            return None
        text = self.run.vision.text.read(frame.png)
        labels = [normalized(line.text) for line in text.lines]
        if text.error or 'top rewards' not in labels or not any(t.startswith('leaves in ') for t in labels):
            return None
        headings = [normalized(line.text) for line in text.lines if line.bounds
                    and frame.width*.4 < line.bounds[0] < frame.width*.65
                    and frame.height*.07 < line.bounds[1] < frame.height*.18]
        if len(headings) != 1 or labels.count(headings[0]) != 2:
            return None
        if self.close_ref is None:
            self.close_ref = _decode((Path(__file__).parent/'assets/wheating/event_board/close.png').read_bytes(), True)
        left = round(frame.width*.80)
        hits = ResourceVision()._search(image[:round(frame.height*.18), left:round(frame.width*.95)],
            self.close_ref, np.linspace(.97,1.03,7)*frame.height/1080, .94, self.run.cancel_event.is_set, 2)
        if len(hits) == 1:
            t = hits[0]
            return 'close_details', VisualTarget(left+t.x,t.y,t.width,t.height,t.score)
        return None

    def pointed_board(self, frame):
        image = self.run.vision._image_for(frame.png)
        small = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        yellow = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (18, 180, 245), (29, 255, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
        if not any(10 < w < 45 and 20 < h < 80 and h > w*1.2 and area > w*h*.35
                   and 60 < x < 400 and 40 < y < 210 for x,y,w,h,area in stats[1:]):
            return None
        if self.refs is None:
            root = Path(__file__).parent/'assets/wheating/event_board'
            self.refs = {name:_decode((root/f'{name}.png').read_bytes(), True) for name in ('pointer','board')}
            self.matcher = ResourceVision()
        base = frame.height/1080
        pointers = self.matcher._search(image, self.refs['pointer'], np.r_[base, np.linspace(.65,1.3,27)*base],
                                        .85, self.run.cancel_event.is_set, 2)
        if len(pointers) != 1:
            return None
        pointer = pointers[0]
        scale = pointer.width/self.refs['pointer'].shape[1]
        left = max(0, round(pointer.x-120*scale))
        top = max(0, round(pointer.y+190*scale))
        crop = image[top:round(pointer.y+490*scale), left:round(pointer.x+240*scale)]
        # The tutorial arrow bounces and stretches independently of the board.
        boards = self.matcher._search(crop, self.refs['board'], np.linspace(.94,1.06,13)*base,
                                     .86, self.run.cancel_event.is_set, 2)
        if len(boards) != 1:
            return None
        board = boards[0]
        return VisualTarget(left+board.x, top+board.y, board.width, board.height, board.score)

    def panel(self, frame):
        return scarecrow_panel(self.run.vision._image_for(frame.png), frame.width, frame.height)

    def observe(self, frame):
        details = self.details(frame)
        if details is not None:
            return details
        panel = self.panel(frame)
        if panel is None:
            close = self.menu_control(frame)
            if close is not None:
                return close
            board = self.pointed_board(frame)
            return ('board', board) if board is not None else None
        result = self.run.vision.text.read(frame.png)
        if result.error:
            return None
        lines = [line for line in result.lines if line.bounds is not None
                 and panel.x < line.bounds[0] < line.bounds[0]+line.bounds[2] < panel.x+panel.width
                 and panel.y < line.bounds[1] < line.bounds[1]+line.bounds[3] < panel.y+panel.height]
        page = PAGES.get(tutorial_message(' '.join(line.text for line in lines)))
        if page is None:
            return None
        return page, panel

    def fresh(self):
        self.run.wait(.25)
        return self.run._capture_raw()

    def process(self, frame):
        seen = set()
        for _ in range(10):
            observed = self.observe(frame)
            if observed is None:
                return frame
            if self.run.client.foreground_package() != 'com.supercell.hayday':
                return frame
            page = observed[0]
            if page in seen:
                raise WheatingPrerequisite('The Event Board introduction repeated a completed step; input paused.')
            seen.add(page)
            frame = self._step(frame, observed)
            if page == 'complete':
                self.run._workspace_needs_restore = True
                self.run._workspace_zoom_attempted = False
                return frame
            # Text and highlights animate between introduction steps. Keep
            # crop work suspended until the next control or clear farm settles.
            for _ in range(8):
                frame = self.fresh()
                if self.observe(frame) is not None:
                    break
            else:
                if not self.run.vision.farm(frame) or CameraNavigator._modal_visible(frame):
                    raise WheatingPrerequisite('The Event Board introduction has an unrecognized next page.')
                return frame
        raise WheatingPrerequisite('The Event Board introduction exceeded its bounded recovery steps.')

    def _step(self, frame, observed):
        if self.uncertain is not None:
            raise WheatingPrerequisite('The Event Board tutorial tap is unconfirmed; no repeated tap was sent.')
        if self.run.client.foreground_package() != 'com.supercell.hayday':
            return frame
        page, panel = observed
        for _ in range(4):
            fresh = self.fresh()
            checked = self.observe(fresh)
            if (checked is not None and checked[0] == page and fresh.captured_at
                    and fresh.captured_at != frame.captured_at
                    and max(abs(a-b) for a,b in zip(panel.box, checked[1].box, strict=True)) <= (30 if page == 'unveil_poster' else 8)
                    and self.run.client.foreground_package() == 'com.supercell.hayday'):
                break
        else:
            raise WheatingPrerequisite('The Event Board tutorial changed before its continuation tap.')
        if self.run.diagnostics:
            (self.run.diagnostics/f'event_board_{page}_before.png').write_bytes(fresh.png)
        self.uncertain = page
        self.run.publish(f'Wheating: continuing the verified Event Board introduction ({page}).')
        self.run.tap(checked[1].center, fresh)
        clear = 0
        for _ in range(12):
            after = self.fresh()
            current = self.observe(after)
            distinct = bool(after.captured_at) and after.captured_at != fresh.captured_at
            clear = clear+1 if distinct and (current is None or current[0] != page) else 0
            fresh = after
            if clear >= 2:
                self.uncertain = None
                if self.run.diagnostics:
                    (self.run.diagnostics/f'event_board_{page}_after.png').write_bytes(after.png)
                return after
        raise WheatingPrerequisite('The Event Board introduction did not advance; no repeated tap was sent.')

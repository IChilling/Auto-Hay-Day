"""Close only recorded incidental pages by their independently verified exits."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hayday.resource_vision import VisualTarget, _decode
from hayday.tutorials import TutorialBlocked, TutorialDismissal


@dataclass(frozen=True)
class PageExit:
    kind: str
    target: VisualTarget


class PageExitVision:
    """Three original-pixel features: page identity, second heading, red exit.

    These pages occupy a fixed viewport independent of the farm camera. Require
    the observed 16:9 layout; unfamiliar layouts never acquire a target.
    """

    def __init__(self):
        source = Path(__file__).resolve().parents[2]/'images/page_exits'
        root = source if source.is_dir() else Path(__file__).parent/'assets/page_exits'
        manifest = json.loads((root/'manifest.json').read_text('utf-8'))
        if (manifest.get('version') != 1 or set(manifest['pages']) !=
                {'achievements', 'county_fair_tutorial', 'county_fair', 'farm_expansion', 'catalog'}):
            raise ValueError('Unsupported page exit references.')
        self.references = {kind: [(entry['box'], _decode((root/entry['file']).read_bytes()))
                                 for entry in page['features']]
                           for kind, page in manifest['pages'].items()}

    def observe(self, png, cancel=lambda: False):
        image = _decode(png)
        height, width = image.shape[:2]
        if abs(width/height-16/9) > .01 or cancel():
            return None
        scale = height/1080
        found = []
        for kind, references in self.references.items():
            offset = (0, 0)
            for index, (box, reference) in enumerate(references):
                if cancel():
                    return None
                x, y, w, h = box
                x, y, w, h = (round(value*scale) for value in (x, y, w, h))
                reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
                radius = max(2, round((12 if index == 0 else 3)*scale))
                left, top = x+offset[0]-radius, y+offset[1]-radius
                patch = image[top:y+offset[1]+h+radius, left:x+offset[0]+w+radius]
                if patch.shape[0] < h or patch.shape[1] < w:
                    break
                response = cv2.matchTemplate(patch, reference, cv2.TM_CCOEFF_NORMED)
                _, correlation, _, (dx, dy) = cv2.minMaxLoc(response)
                actual = patch[dy:dy+h, dx:dx+w]
                difference = np.abs(actual.astype(float)-reference).mean()
                if correlation < .965 or difference > 14:
                    break
                offset = (left+dx-x, top+dy-y)
            else:
                # Last independently matched feature is the visible exit.
                found.append(PageExit(kind, VisualTarget(x+offset[0], y+offset[1], w, h, correlation)))
        if found:
            return found[0] if len(found) == 1 else None
        return self._generic_exit(image, png, cancel)

    def _generic_exit(self, image, png, cancel):
        """Reuse the red exit only on a dimmed modal outside the order panel."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hud = hsv[round(height*.014):round(height*.144), round(width*.013):round(width*.0875)]
        blue = hud[(hud[:, :, 0] > 90) & (hud[:, :, 0] < 125) & (hud[:, :, 1] > 75)]
        if len(blue) < hud.shape[0]*hud.shape[1]*.015 or np.quantile(blue[:, 2], .9) >= 190:
            return None
        from hayday.panel import PanelVerifier
        if PanelVerifier().verify(png, cancel=cancel).verified:
            return None
        reference = self.references['achievements'][2][1]
        colors = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        button_mask = ((colors[:, :, 1] > 100) & (colors[:, :, 2] > 120)).astype(np.uint8)*255
        left, bottom = round(width*.78), round(height*.24)
        region = image[:bottom, left:]
        candidates = []
        for factor in np.linspace(.5, 1.25, 76)*height/1080:
            if cancel():
                return None
            scaled = cv2.resize(reference, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            h, w = scaled.shape[:2]
            if h > region.shape[0] or w > region.shape[1]:
                continue
            mask = cv2.resize(button_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            response = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED, mask=mask)
            response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
            _, score, _, (x, y) = cv2.minMaxLoc(response)
            if score < .975 or np.abs(region[y:y+h, x:x+w].astype(float)-scaled)[mask > 0].mean() > 8:
                continue
            candidates.append(VisualTarget(left+x, y, w, h, score))
        if not candidates:
            return None
        best = max(candidates, key=lambda target: target.score)
        if any(np.linalg.norm(np.subtract(target.center, best.center)) > 12*height/1080
               for target in candidates):
            return None
        return PageExit('incidental_popup', best)


class PageExitRecovery(TutorialDismissal):
    """Fresh confirmation, one exit tap, then two frames without that page."""

    def _observe(self, frame):
        self._check()
        if self.vision is None:
            self.vision = PageExitVision()
        result = self.vision.observe(frame.png, cancel=self.cancel_event.is_set)
        self._check()
        return result

    def process(self, frame):
        self._check()
        if self._uncertain:
            raise TutorialBlocked('An earlier page exit remains uncertain; no repeated tap is allowed.')
        handled = set()
        for _ in range(3):
            first = self._observe(frame)
            if first is None:
                return frame
            if first.kind in handled:
                raise TutorialBlocked('An incidental page returned after its exit; navigation paused.')
            self.save('page_exit_detected', frame)
            previous = first
            for _ in range(5):
                fresh = self._fresh(frame)
                current = self._observe(fresh)
                distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
                if current is None and distinct:
                    return fresh
                if (distinct and current.kind == previous.kind and self._same(previous, current)
                        and self._valid_target(current, fresh)):
                    break
                # The opening animation changes the exit's scale and position;
                # its generic match may become an exact page match as it settles.
                # Accept that transition only as observation, never as tap proof.
                compatible = (current is not None and previous is not None
                    and (current.kind == previous.kind
                         or 'incidental_popup' in (current.kind, previous.kind))
                    and np.linalg.norm(np.subtract(current.target.center, previous.target.center))
                        <= 100*fresh.height/1080
                    and .8 <= current.target.width/max(1, previous.target.width) <= 1.25)
                if not distinct or not compatible:
                    break
                previous, frame = current, fresh
            else:
                current = None
            if (not distinct or current is None or current.kind != previous.kind
                    or not self._same(previous, current) or not self._valid_target(current, fresh)):
                raise TutorialBlocked('The page exit changed before confirmation; no tap was sent.')
            self._check()
            kind = current.kind
            self.events.append({'kind': kind, 'stage': 'tap_attempted',
                                'captured_at': fresh.captured_at, 'target': list(current.target.center)})
            self.save('page_exit_before_tap', fresh)
            self._uncertain = True
            self.progress(f"Closing the confirmed {kind.replace('_', ' ')} page using its exit.")
            self.client.tap(*current.target.center, width=fresh.width, height=fresh.height)
            self._check()
            frame, clear = fresh, 0
            for _ in range(8):
                fresh = self._fresh(frame)
                current = self._observe(fresh)
                distinct = bool(fresh.captured_at) and fresh.captured_at != frame.captured_at
                clear = clear+1 if distinct and (current is None or current.kind != kind) else 0
                frame = fresh
                if clear >= 2:
                    self._uncertain = False
                    handled.add(kind)
                    self.events.append({'kind': kind, 'stage': 'dismissed', 'captured_at': frame.captured_at})
                    self.save('page_exit_dismissed', frame)
                    break
            else:
                raise TutorialBlocked('The page remained after its exit attempt; no repeated tap was sent.')
        if self._observe(frame) is not None:
            raise TutorialBlocked('Too many incidental pages interrupted order navigation.')
        return frame

"""Return from an accidental fishing-area visit before farm navigation."""

from hayday.fishing_vision import FishingVision
from hayday.page_exits import PageExit
from hayday.tutorials import TutorialBlocked, TutorialDismissal


class FarmHomeRecovery(TutorialDismissal):
    """Reuse the recorded Home icon, with one freshly confirmed navigation tap."""

    def _observe(self, frame):
        self._check()
        self.vision = self.vision or FishingVision()
        target = self.vision.home(frame.png, self.cancel_event.is_set)
        self._check()
        return PageExit('fishing_area', target) if target is not None else None

    def process(self, frame, *, allow_input=True):
        self._check()
        if self._uncertain:
            raise TutorialBlocked('The previous Home tap remains unconfirmed; no repeat navigation was sent.')
        first = self._observe(frame)
        if first is None:
            return frame
        if not allow_input:
            raise TutorialBlocked('The fishing area is open while unrelated navigation is disabled.')
        fresh = self._fresh(frame)
        current = self._observe(fresh)
        if (fresh.captured_at == frame.captured_at or not self._same(first, current)
                or not self._valid_target(current, fresh)):
            raise TutorialBlocked('The Home button changed before confirmation; no tap was sent.')
        self.save('home_before_tap', fresh)
        self.progress('Returning from the fishing area using the confirmed Home button.')
        self._check()
        self._uncertain = True
        self.client.tap(*current.target.center, width=fresh.width, height=fresh.height)
        clear = 0
        for _ in range(12):
            after = self._fresh(fresh)
            present = self._observe(after)
            clear = clear+1 if after.captured_at != fresh.captured_at and present is None else 0
            fresh = after
            if clear >= 2:
                self._uncertain = False
                self.save('home_returned', fresh)
                return fresh
        raise TutorialBlocked('Home did not return to the farm; no camera movement or repeated tap was sent.')

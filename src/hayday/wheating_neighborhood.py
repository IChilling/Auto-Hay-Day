"""Dismiss the recorded level-ten Neighborhood House introduction."""
from hayday.camera import CameraNavigator
from hayday.wheating import WheatingPrerequisite
from hayday.wheating_tutorial_panel import scarecrow_panel, tutorial_message

MESSAGES = {
    'my my what do we have here repair your neighborhood house to chat and play with your closest friends',
}


class WheatNeighborhood:
    def __init__(self, run):
        self.run = run
        self.uncertain = False

    def observe(self, frame):
        self.run.check()
        panel = scarecrow_panel(self.run.vision._image_for(frame.png), frame.width, frame.height)
        if panel is None:
            return None
        result = self.run.vision.text.read(frame.png)
        if result.error:
            return None
        lines = [line for line in result.lines if line.bounds is not None
                 and panel.x < line.bounds[0] < line.bounds[0]+line.bounds[2] < panel.x+panel.width
                 and panel.y < line.bounds[1] < line.bounds[1]+line.bounds[3] < panel.y+panel.height]
        message = tutorial_message(' '.join(line.text for line in lines))
        return panel if message in MESSAGES else None

    def fresh(self):
        self.run.wait(.25)
        return self.run._capture_raw()

    def save(self, label, frame):
        if self.run.diagnostics:
            (self.run.diagnostics/f'neighborhood_{label}.png').write_bytes(frame.png)

    def process(self, frame):
        self.run.check()
        if self.uncertain:
            raise WheatingPrerequisite('The Neighborhood introduction tap is unconfirmed; no repeated tap was sent.')
        previous = self.observe(frame)
        if previous is None or self.run.client.foreground_package() != 'com.supercell.hayday':
            return frame
        for _ in range(4):
            fresh = self.fresh()
            current = self.observe(fresh)
            if (current is not None and fresh.captured_at and fresh.captured_at != frame.captured_at
                    and max(abs(a-b) for a,b in zip(previous.box, current.box, strict=True)) <= 8
                    and self.run.client.foreground_package() == 'com.supercell.hayday'):
                break
        else:
            raise WheatingPrerequisite('The Neighborhood introduction changed before its continuation tap.')
        self.save('before', fresh)
        self.uncertain = True
        self.run.publish('Wheating: continuing the verified level-10 Neighborhood House introduction.')
        self.run.tap(current.center, fresh)
        # The introduction moves the camera away from the saved fields.
        self.run._workspace_needs_restore = True
        self.run._workspace_zoom_attempted = False
        clear = 0
        for _ in range(20):
            after = self.fresh()
            distinct = bool(after.captured_at) and after.captured_at != fresh.captured_at
            ready = (distinct and self.run.client.foreground_package() == 'com.supercell.hayday'
                     and self.observe(after) is None and self.run.vision.farm(after)
                     and not CameraNavigator._modal_visible(after))
            clear = clear+1 if ready else 0
            fresh = after
            if clear >= 2:
                self.uncertain = False
                self.save('completed', after)
                self.run.publish('Wheating: Neighborhood introduction completed; restoring the field workspace.')
                return after
        self.save('unconfirmed', fresh)
        raise WheatingPrerequisite('The Neighborhood introduction did not return to a clear farm; no repeated tap was sent.')

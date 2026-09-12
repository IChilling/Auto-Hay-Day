"""Pure timing/visual evidence gate for caller-owned rewarded-ad sessions.

This module never identifies the start of an ad or sends input. The caller starts
a new gate only after observing the ad, and supplies a freshly detected corner control.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

from hayday.ad_exit import AdExitCandidate
from hayday.ad_text import AdTextObservation, parse_ad_text


@dataclass(frozen=True)
class AdWatchDecision:
    status: str
    message: str
    heuristic: bool
    elapsed: float


class AdWatchGate:
    """Wait at least 30 seconds, then decide from fresh pixels through 60 seconds.

    ``text=None`` means no successful text read. It is allowed for ads without
    countdown evidence, but cannot clear a previously observed reward countdown.
    An empty string is a successful read with no recognized text. Candidate and
    motion stability are separate so an animated end card may close at 60 seconds.
    """

    min_seconds = 30.0
    max_seconds = 60.0
    candidate_seconds = .8
    quiet_seconds = 2.0

    def __init__(self, initial_elapsed: float = 0.0):
        if (isinstance(initial_elapsed, bool) or not isinstance(initial_elapsed, (int, float))
                or not math.isfinite(initial_elapsed) or not 0 <= initial_elapsed <= self.max_seconds):
            raise ValueError('Initial ad watch time must be between 0 and 60 seconds.')
        self.initial_elapsed = float(initial_elapsed)
        self._started: float | None = None
        self._last_now: float | None = None
        self._last_image: np.ndarray | None = None
        self._last_size: tuple[int, int] | None = None
        self._quiet_since: float | None = None
        self._candidate: AdExitCandidate | None = None
        self._candidate_since: float | None = None
        self._candidate_frames = 0
        self._countdown_seen = False
        self._countdown_resolved = False
        self._countdown_deadline: float | None = None
        self._reward_granted = False
        self._terminal: AdWatchDecision | None = None

    @staticmethod
    def _text_evidence(text: str) -> tuple[bool, float | None, bool]:
        """Return countdown-present, seconds (None if unreadable), reward-granted.

        Only reward-related countdowns arm the guard. A skip timer is not proof
        that watching is complete and does not impose a reward deadline.
        """
        normalized = re.sub(r'\s+', ' ', text.casefold()).strip()
        granted = parse_ad_text(text).reward_granted
        expressions = []
        for match in re.finditer(
            r"\brewards?(?:\s+[a-z']+){0,7}?\s+(?:in|after)\s*[:\-]?\s*", normalized
        ):
            if 'skip' not in match.group(0).split():
                expressions.append(normalized[match.end():])
        for match in re.finditer(
            r'\b(?:watch|wait)\s+([\d:?….-]+)\s*(?:more\s+)?(?:seconds?|secs?|s)\b'
            r'[^.!?]{0,40}\breward\b', normalized
        ):
            expressions.append(match.group(1))
        for match in re.finditer(
            r'(\d{1,3}(?::[0-5]\d)?)\s*(?:seconds?|secs?|s)\s+(?:(?:remaining|left)\s+)?'
            r'(?:until|for|to)\s+(?:(?:get|receive|earn|your|the|a)\s+){0,4}rewards?\b'
            r'|\brewards?\s*:?\s*(\d{1,2}:[0-5]\d)\b'
            r'|\brewards?\s*:\s*([^.!?]{1,25}(?:seconds?|secs?|s)\s+(?:remaining|left)\b)',
            normalized,
        ):
            if re.search(r'\bskip(?:\s+ad)?\s+in\s*:?\s*$', normalized[max(0, match.start()-25):match.start()]):
                continue
            expressions.append(next(group for group in match.groups() if group is not None))
        values = []
        unreadable = False
        for expression in expressions:
            value = re.match(r'^(\d{1,3}(?::[0-5]\d)?)(?![:\d])(?=\s*(?:seconds?|secs?|s)\b|\b)', expression)
            if value:
                pieces = value.group(1).split(':')
                seconds = float(int(pieces[0])*60+int(pieces[1]) if len(pieces) == 2 else int(pieces[0]))
                values.append(seconds)
                continue
            # A known countdown cue with missing/damaged digits is unfinished.
            if (not expression or re.match(r'^[?…:.-]', expression)
                    or re.match(r'^\S*\d\S*\s*(?:seconds?|secs?|s)\b', expression)
                    or re.match(r'^\S{1,8}\s+(?:seconds?|secs?|s)\b', expression)
                    or re.match(r'^(?:xx|seconds?|secs?)\b', expression)):
                unreadable = True
        if unreadable:
            return True, None, granted
        if values:
            return True, max(values), granted
        return False, None, granted

    def _observe_text(self, text: str | AdTextObservation | None, now: float):
        if isinstance(text, AdTextObservation):
            if text.error:
                return
            present, seconds, _ = self._text_evidence(text.text)
            granted = text.reward_granted
            explicit = text.reward_countdown_seconds
            if (isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0
                    and (not present or seconds is not None)):
                present, seconds = True, max(explicit, seconds or 0)
        elif isinstance(text, str):
            present, seconds, granted = self._text_evidence(text)
        else:
            return
        if present:
            self._countdown_seen = True
            self._countdown_resolved = seconds == 0
            if seconds is not None:
                self._countdown_deadline = now+seconds
            if seconds != 0:
                self._reward_granted = False
        elif self._countdown_seen and not self._countdown_resolved:
            if granted or self._countdown_deadline is None or now >= self._countdown_deadline:
                self._countdown_resolved = True
        if granted and (not present or seconds == 0):
            self._reward_granted = True

    @staticmethod
    def _decode(png: bytes):
        try:
            image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, TypeError, cv2.error):
            return None
        if image is None or min(image.shape[:2]) < 32:
            return None
        return image

    @staticmethod
    def _valid_candidate(candidate, width, height):
        if candidate is None or getattr(candidate, 'corner', None) not in {'top_left', 'top_right'}:
            return False
        if getattr(candidate, 'kind', 'close') not in {'close', 'skip'}:
            return False
        bounds = getattr(candidate, 'bounds', None)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
            return False
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in bounds):
            return False
        score = getattr(candidate, 'score', None)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 < score <= 1:
            return False
        x, y, w, h = bounds
        if min(x, y) < 0 or min(w, h) < 3 or x+w > width or y+h > height*.25:
            return False
        return x+w <= width*.30 if candidate.corner == 'top_left' else x >= width*.70

    @staticmethod
    def _same_candidate(first, second):
        def center(candidate):
            x, y, width, height = candidate.bounds
            return x+width/2, y+height/2

        return (first is not None and second is not None and first.corner == second.corner
                and getattr(first, 'kind', 'close') == getattr(second, 'kind', 'close')
                and all(abs(a-b) <= max(3, min(first.bounds[2:], default=3)*.15)
                        for a, b in zip(center(first), center(second), strict=True))
                and all(abs(a-b) <= max(2, min(a, b)*.15)
                        for a, b in zip(first.bounds[2:], second.bounds[2:], strict=True)))

    def _result(self, status, message, elapsed, heuristic=False):
        result = AdWatchDecision(status, message, heuristic, elapsed)
        # Close authorization is deliberately re-evaluated on every fresh frame.
        # A disappearing/moving control or new countdown must revoke it before input.
        if status == 'timeout':
            self._terminal = result
        return result

    def observe(self, png: bytes, candidate: AdExitCandidate | None,
                text: str | AdTextObservation | None = None, now: float | None = None) -> AdWatchDecision:
        if self._terminal is not None:
            return self._terminal
        now = time.monotonic() if now is None else now
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('Ad observation time must be finite.')
        if self._started is None:
            self._started = now
        elapsed = max(0.0, self.initial_elapsed+now-self._started)
        expired = elapsed >= self.max_seconds
        if self._last_now is not None and now < self._last_now:
            return self._result('timeout', 'Ad observation clock moved backwards; no exit authorized.', elapsed)
        self._last_now = now
        image = self._decode(png)
        if image is None:
            self._last_image = self._candidate = None
            self._candidate_since = self._quiet_since = None
            self._candidate_frames = 0
            return self._result('timeout' if expired else 'wait',
                                'The ad screenshot could not be read; no exit authorized.', elapsed)
        height, width = image.shape[:2]
        size = width, height
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        changed = self._last_image is None or self._last_size != size
        if not changed:
            difference = np.abs(small.astype(np.int16)-self._last_image.astype(np.int16)).mean(axis=2)
            changed = float(difference.mean()) > 2.2 or float((difference > 12).mean()) > .02
        if changed or self._quiet_since is None:
            self._quiet_since = now
        if self._last_size is not None and self._last_size != size:
            self._candidate = None
        self._last_image, self._last_size = small, size
        candidate = candidate if self._valid_candidate(candidate, width, height) else None
        if self._same_candidate(self._candidate, candidate):
            self._candidate_frames += 1
        else:
            self._candidate_since = now if candidate else None
            self._candidate_frames = 1 if candidate else 0
            self._quiet_since = now
            self._candidate = candidate
        # A grant belongs to the text observed on this screen. Cached countdowns
        # survive read failures, but an earlier grant cannot authorize a new scene.
        self._reward_granted = False
        self._observe_text(text, now)
        stable = (candidate is not None and self._candidate_frames >= 2
                  and self._candidate_since is not None and now-self._candidate_since >= self.candidate_seconds)
        countdown_blocked = self._countdown_seen and not self._countdown_resolved
        if elapsed < self.min_seconds:
            return self._result('wait', 'Waiting for at least 30 seconds of this ad.', elapsed)
        if countdown_blocked:
            return self._result('timeout' if expired else 'wait',
                                'The reward countdown is unfinished or has not been observed cleared.', elapsed)
        if not stable:
            return self._result('timeout' if expired else 'wait',
                                'No single stable corner exit has been confirmed.', elapsed)
        if self._reward_granted:
            return self._result('close', 'The reward was granted and the exit is stable.', elapsed)
        if expired:
            return self._result('close', 'The 60-second watch limit was reached with a stable exit.', elapsed, True)
        if getattr(candidate, 'kind', 'close') == 'skip':
            return self._result('wait',
                'A skip symbol is visible; waiting for reward confirmation or the full 60-second watch.', elapsed)
        if now-self._quiet_since >= self.quiet_seconds:
            return self._result('close', 'The ad has been watched for 30 seconds and its screen has settled.', elapsed, True)
        return self._result('wait', 'The ad is still moving; waiting for a settled screen or the watch limit.', elapsed)

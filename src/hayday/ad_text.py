"""Local English ad text recognition; this module never sends device input.

Explicit reward wording is evidence for the caller's ad state machine. A generic
close button, CTA, skipped timer, or completed OCR call is not a reward grant.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdTextLine:
    text: str
    bounds: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class AdTextObservation:
    text: str = ''
    lines: tuple[AdTextLine, ...] = ()
    reward_granted: bool = False
    reward_countdown_seconds: int | None = None
    error: str | None = None


_GRANTED = re.compile(
    r'(?:(?:your |the )?reward (?:has been |was |is )?(?:granted|earned|received)'
    r'|you (?:have |have successfully )?(?:earned|received) (?:your |a |the )?reward'
    r'|you\'ve (?:earned|received) (?:your |a |the )?reward)'
)
_SECONDS = r'(?P<seconds>[0-9]{1,3})\s*(?:s|secs?|seconds?)'
_COUNTDOWNS = tuple(re.compile(pattern) for pattern in (
    rf'(?:your |the )?reward (?:available |granted |earned |received |will be granted )?in\s*:?\s*{_SECONDS}',
    rf'(?:your |the )?reward\s*:?\s*{_SECONDS} (?:remaining|left)',
    rf'{_SECONDS} (?:remaining |left )?(?:until|for|to earn|to receive) (?:your |a |the )?reward',
))
_CLOCK = re.compile(r'(?:your |the )?reward (?:available )?in\s*:?\s*(?P<minutes>[0-9]{1,2}):(?P<seconds>[0-5][0-9])')


def _normalized(text):
    return re.sub(r'\s+', ' ', text.replace('\u2019', "'").replace('\u00a0', ' ')).strip(' .!\t\r\n').lower()


def _near(first, second):
    if first.bounds is None or second.bounds is None:
        return True
    ax, ay, aw, ah = first.bounds
    bx, by, bw, bh = second.bounds
    vertical = max(0, by-ay-ah, ay-by-bh)
    horizontal = max(0, bx-ax-aw, ax-bx-bw)
    return vertical <= max(ah, bh)*1.3 and horizontal <= max(ah, bh)*3


def parse_ad_text(value: str | Iterable[AdTextLine | str]) -> AdTextObservation:
    """Parse complete status lines without OCR substitutions or guessed digits.

    A standalone '5 seconds remaining' or 'skip in 5 seconds' has no reward
    association. A nearby explicit Reward label can qualify a remaining timer.
    """
    raw = value.splitlines() if isinstance(value, str) else value
    lines = tuple(line if isinstance(line, AdTextLine) else AdTextLine(str(line))
                  for line in raw if (line.text if isinstance(line, AdTextLine) else str(line)).strip())
    texts = [_normalized(line.text) for line in lines]
    candidates = [(index, index, text) for index, text in enumerate(texts)]
    for index in range(len(lines)-1):
        if _near(lines[index], lines[index+1]):
            candidates.append((index, index+1, texts[index]+' '+texts[index+1]))
    granted = False
    for first, last, text in candidates:
        if not _GRANTED.fullmatch(text):
            continue
        neighbors = [index for index in (first-1, last+1) if 0 <= index < len(lines)
                     and _near(lines[first if index < first else last], lines[index])]
        qualified = any(re.match(r'(?:no|not|never)$|(?:only if|only after|if|when|once|after|unless|until)\b',
                                 texts[index]) for index in neighbors)
        if not qualified:
            granted = True
    remaining = []
    for _, _, text in candidates:
        found = next((match for pattern in _COUNTDOWNS if (match := pattern.fullmatch(text))), None)
        if found:
            seconds = int(found['seconds'])
            if 0 <= seconds <= 600:
                remaining.append(seconds)
        elif found := _CLOCK.fullmatch(text):
            seconds = int(found['minutes'])*60+int(found['seconds'])
            if seconds <= 600:
                remaining.append(seconds)
    countdown = max(remaining) if remaining else None
    # Contradictory positive unfinished countdowns take precedence over a grant.
    if countdown is not None and countdown > 0:
        granted = False
    return AdTextObservation('\n'.join(line.text for line in lines), lines, granted, countdown)


async def _recognize_local(png: bytes):
    """Run inside the disposable OCR process, with no filesystem/network API."""
    if sys.platform == 'darwin':
        return _recognize_macos(png)
    from PIL import Image
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter

    engine = OcrEngine.try_create_from_language(Language('en-US'))
    if engine is None:
        raise RuntimeError('The local English Windows OCR recognizer is unavailable.')
    with Image.open(io.BytesIO(png)) as original:
        width, height = original.size
        if original.format != 'PNG' or min(width, height) <= 0 or width*height > 32_000_000:
            raise ValueError('A bounded PNG screenshot is required.')
        limit = min(4096, OcrEngine.max_image_dimension)
        ratio = min(1.0, limit/max(width, height))
        image = original.convert('RGBA')
        if ratio < 1:
            image = image.resize((max(1, round(width*ratio)), max(1, round(height*ratio))), Image.Resampling.LANCZOS)
        iw, ih = image.size
        pixels = image.tobytes('raw', 'BGRA')
    with DataWriter() as writer:
        writer.write_bytes(pixels)
        buffer = writer.detach_buffer()
        with SoftwareBitmap.create_copy_from_buffer(buffer, BitmapPixelFormat.BGRA8, iw, ih) as bitmap:
            result = await engine.recognize_async(bitmap)
            lines = []
            for line in result.lines:
                words = list(line.words)
                if not words:
                    continue
                left = max(0, math.floor(min(word.bounding_rect.x for word in words)*width/iw))
                top = max(0, math.floor(min(word.bounding_rect.y for word in words)*height/ih))
                right = min(width, math.ceil(max(word.bounding_rect.x+word.bounding_rect.width for word in words)*width/iw))
                bottom = min(height, math.ceil(max(word.bounding_rect.y+word.bounding_rect.height for word in words)*height/ih))
                lines.append({'text': line.text, 'bounds': [left, top, right-left, bottom-top]})
            return {'lines': lines, 'language': engine.recognizer_language.language_tag}


def _recognize_macos(png: bytes):
    """Use Apple's on-device Vision recognizer in the same bounded worker."""
    import Vision
    from Foundation import NSData
    from PIL import Image

    with Image.open(io.BytesIO(png)) as original:
        width, height = original.size
        if original.format != 'PNG' or min(width, height) <= 0 or width*height > 32_000_000:
            raise ValueError('A bounded PNG screenshot is required.')
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(['en-US'])
    request.setUsesLanguageCorrection_(False)
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f'Apple Vision text recognition failed: {error}')
    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        box = observation.boundingBox()
        # Vision uses a bottom-left origin; the app uses top-left pixels.
        left = max(0, math.floor(box.origin.x*width))
        top = max(0, math.floor((1-box.origin.y-box.size.height)*height))
        right = min(width, math.ceil((box.origin.x+box.size.width)*width))
        bottom = min(height, math.ceil((1-box.origin.y)*height))
        if right > left and bottom > top:
            lines.append({'text': str(candidates[0].string()),
                          'bounds': [left, top, right-left, bottom-top]})
    return {'lines': lines, 'language': 'en-US'}


class AdTextReader:
    """Read ad screenshots with a hard native-worker deadline and cancellation."""

    def __init__(self, timeout_seconds: float = 3.0, cancel: Callable[[], bool] | None = None):
        if not .1 <= timeout_seconds <= 10:
            raise ValueError('Ad OCR timeout must be between 0.1 and 10 seconds.')
        self.timeout_seconds = timeout_seconds
        self.cancel = cancel or (lambda: False)
        self._lock = threading.Lock()

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                return
        try:
            process.communicate(timeout=.25)
        except subprocess.TimeoutExpired:
            # No OCR result from this process will ever be read or acted on.
            process.kill()

    def read(self, png: bytes, cancel: Callable[[], bool] | None = None) -> AdTextObservation:
        cancelled = cancel or self.cancel
        if cancelled():
            return AdTextObservation(error='Ad OCR cancelled.')
        if sys.platform not in ('win32', 'darwin'):
            return AdTextObservation(error='Local ad OCR is unavailable on this platform.')
        if not isinstance(png, bytes) or not png.startswith(b'\x89PNG\r\n\x1a\n') or len(png) > 32_000_000:
            return AdTextObservation(error='A bounded PNG screenshot is required for ad OCR.')
        if not self._lock.acquire(blocking=False):
            return AdTextObservation(error='An ad OCR read is already active.')
        process = None
        try:
            deadline = time.monotonic()+self.timeout_seconds
            with tempfile.TemporaryDirectory(prefix='hayday-ad-ocr-') as directory:
                source = Path(directory)/'screen.png'
                source.write_bytes(png)
                result_path = Path(directory)/'result.json'
                frozen = getattr(sys, 'frozen', False)
                command = ([sys.executable, '--hayday-ocr-worker', str(source), str(result_path)]
                           if frozen else [sys.executable, str(Path(__file__).resolve()), '--worker', str(source)])
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                while True:
                    if cancelled():
                        self._stop(process)
                        return AdTextObservation(error='Ad OCR cancelled.')
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        self._stop(process)
                        return AdTextObservation(error='Local ad OCR exceeded its time limit.')
                    try:
                        output, _ = process.communicate(timeout=min(.05, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    return AdTextObservation(error='Local ad OCR failed before producing a result.')
                # Windowed executables have no reliable stdout. A private result
                # file keeps the same bounded worker protocol in packaged builds.
                payload = json.loads(result_path.read_text('utf-8') if frozen else output)
                if payload.get('error'):
                    return AdTextObservation(error=payload['error'])
                lines = tuple(AdTextLine(str(item['text']), tuple(item['bounds'])) for item in payload['lines'])
                return parse_ad_text(lines)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return AdTextObservation(error=f'Local ad OCR is unavailable: {exc}')
        finally:
            if process is not None and process.poll() is None:
                self._stop(process)
            self._lock.release()


def _worker_main(source=None, destination=None):
    try:
        result = asyncio.run(_recognize_local(Path(source or sys.argv[2]).read_bytes()))
    except Exception as exc:
        result = {'error': f'Local English OCR failed: {exc}'}
    output = json.dumps(result, ensure_ascii=True)
    if destination is not None:
        Path(destination).write_text(output, encoding='utf-8')
    else:
        print(output)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('This module is an internal local OCR worker.')
    _worker_main()

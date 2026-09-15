"""Cheap geometry gate for Scarecrow speech bubbles; never authorizes input."""
import re

import cv2

from hayday.resource_vision import VisualTarget


def tutorial_message(text):
    """Normalize only recorded OCR spellings before a complete-message match.

    Windows OCR varies individual decorative glyphs between captures, so whole
    misread sentence aliases miss otherwise identical prompts. Keep this local
    to the known Scarecrow messages; do not fuzzy-match or discard extra words.
    """
    words = re.sub(r'[^a-z0-9]+', ' ', text.casefold()).split()
    spellings = {
        'mg': 'my', 'haue': 'have', 'gour': 'your', 'bo': 'to',
        'chab': 'chat', 'plag': 'play', 'gou': 'you', 'jusb': 'just',
        'goujusb': 'you just', 'bhe': 'the', 'inpormabion': 'information',
        'inpo': 'info', 'bubbon': 'button', 'everg': 'every', 'dag': 'day', 'por': 'for',
    }
    return ' '.join(spellings.get(word, word) for word in words)


def scarecrow_panel(image, width, height):
    small = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    cream = cv2.inRange(hsv, (15, 10, 195), (35, 85, 255))
    _, _, stats, _ = cv2.connectedComponentsWithStats(cream)
    parts = [(x,y,w,h) for x,y,w,h,area in stats[1:]
             if w > 480*.45 and h > 270*.40 and area > w*h*.88
             and x > 480*.20 and x+w < 480*.98 and y > 270*.06 and y+h < 270*.92]
    if len(parts) != 1:
        return None
    x,y,w,h = parts[0]
    return VisualTarget(round(x*width/480), round(y*height/270),
                        round(w*width/480), round(h*height/270), 1.)

"""Inspect visible water for another fishing picker without spending a lure."""
from __future__ import annotations

import time

import cv2
import numpy as np

from hayday.camera import CameraNavigator
from hayday.resource_vision import _decode


def water_clearance(png, *, left=.22):
    """Distance from clear pond water to objects, cloud cover, or the HUD."""
    image = _decode(png)
    height,width = image.shape[:2]
    water = cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(85,60,65),(105,200,225))
    allowed = np.zeros_like(water)
    allowed[round(height*.22):round(height*.80),round(width*left):round(width*.80)] = 255
    water &= allowed
    edges = cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),60,130)
    water[cv2.dilate(edges,np.ones((7,7),np.uint8)) > 0] = 0
    return cv2.distanceTransform(water,cv2.DIST_L2,5)


def water_points(png, *, left=.22, minimum=None):
    clearance = water_clearance(png, left=left)
    height = clearance.shape[0]
    points = []
    if minimum is None:
        # A spot's small white sparkle is surrounded almost entirely by water.
        # Prefer adjacent clear water over arbitrary large empty pond regions.
        image = _decode(png)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, (0,0,225), (179,45,255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(bright)
        scale = height/1080
        radius = round(30*scale)
        for x,y,width,h,area in stats[1:]:
            if not (4*scale <= width <= 50*scale and 4*scale <= h <= 50*scale
                    and .4 <= width/h <= 2.5 and 12*scale**2 <= area <= 700*scale**2):
                continue
            cx,cy = int(x+width//2),int(y+h//2)
            point = (cx,cy+round(40*scale))
            if not (radius <= cx < image.shape[1]-radius and radius <= cy < height-radius
                    and point[1] < height and clearance[point[1],point[0]] >= max(12,height*.017)):
                continue
            patch = hsv[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
            if np.mean(cv2.inRange(patch,(85,60,65),(105,200,225)) > 0) < .95:
                continue
            points.append(point)
            cv2.circle(clearance,point,round(height*.18),0,-1)
            if len(points) == 3:
                break
    for _ in range(6-len(points)):
        _,radius,_,point = cv2.minMaxLoc(clearance)
        if radius < (minimum if minimum is not None else max(12,height*.017)):
            break
        points.append(point)
        cv2.circle(clearance,point,round(height*.18),0,-1)
    return tuple(points)


def water_pinch(png):
    clearance = water_clearance(png)
    height,width = clearance.shape
    for fraction in (.30,.24,.18):
        span = round(width*fraction)
        paired = np.minimum(clearance[:,:-span],clearance[:,span:])
        _,radius,_,point = cv2.minMaxLoc(paired)
        if radius >= max(12,height*.017):
            return (point[0]+span//2,point[1]),span
    return None


class FishingNavigator:
    def __init__(self, worker):
        self.worker = worker

    def _water_view(self, frame):
        return (self.worker.vision.home(frame.png,self.worker.cancel_event.is_set) is not None
                and not CameraNavigator._modal_visible(frame))

    def _fresh_water(self, points, *, left=.22, minimum=None):
        owner = self.worker
        fresh = owner._frame()
        if not self._water_view(fresh):
            return None
        clearance = water_clearance(fresh.png, left=left)
        if any(not (0 <= x < fresh.width and 0 <= y < fresh.height)
               or clearance[y,x] < (minimum if minimum is not None else max(12,fresh.height*.017)) for x,y in points):
            return None
        owner._check()
        return fresh

    def find_spot(self, frame):
        """Return a recognized picker, or stop after one zoom and six probes.

        Water only supplies inspection targets. The caller must independently
        revalidate the picker and stock before any cast or lure production.
        """
        owner = self.worker
        owner._check()
        if not self._water_view(frame):
            return None
        deadline = time.monotonic()+35
        pinch = water_pinch(frame.png)
        if pinch:
            center,span = pinch
            points = ((center[0]-span//2,center[1]),(center[0]+span//2,center[1]))
            fresh = self._fresh_water(points)
            if fresh is None:
                return None
            owner.progress('Widening the fishing view to inspect other visible water.')
            owner.client.pinch_zoom_out(width=fresh.width,height=fresh.height,center=center,
                                         span=span,cancel_event=owner.cancel_event)
            owner.cancel_event.wait(.4)
            frame = owner._frame()
        for attempt in range(6):
            owner._check()
            if time.monotonic() >= deadline or not self._water_view(frame):
                return None
            menu = owner.vision.menu(frame.png,owner.cancel_event.is_set)
            if menu is not None:
                return frame
            points = water_points(frame.png)
            if attempt >= len(points):
                return None
            point = points[attempt]
            fresh = self._fresh_water((point,))
            if fresh is None:
                # Moving ripples can invalidate one inspection point. Spend
                # no input there, then use a newly observed candidate within
                # the same six-probe bound.
                frame = owner._frame()
                continue
            owner.progress(f'Inspecting another visible fishing area ({attempt+1}/6).')
            owner.client.tap(*point,width=fresh.width,height=fresh.height)
            owner.cancel_event.wait(.4)
            frame = owner._frame()
        if self._water_view(frame) and owner.vision.menu(frame.png,owner.cancel_event.is_set):
            return frame
        return None

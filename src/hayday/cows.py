"""Recognize global milk readiness, cow feed, and complete neighboring cow pens."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from hayday.animals import SheepVision
from hayday.quantities import read_count
from hayday.resource_vision import ResourceVision, VisualTarget, _decode


class CowVision:
    def __init__(self, reference_path=None):
        source = Path(__file__).resolve().parents[2]/'images/animals/cow'
        self.root = Path(reference_path) if reference_path else (
            source if source.is_dir() else Path(__file__).parent/'assets/animals/cow')
        self.manifest = json.loads((self.root/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported cow references.')
        self.item = (self.root/self.manifest['desired_item']).read_bytes()
        self.item_variants = (self.item,) + tuple((self.root/spec['file']).read_bytes()
            for spec in self.manifest.get('desired_variants', ()))
        self.references = {name:_decode((self.root/spec['file']).read_bytes(),True)
                           for name,spec in self.manifest['features'].items()}
        self.matcher = ResourceVision()

    def supports(self, icon, cancel):
        return icon is not None and any(self.matcher.find_item(
            icon,reference,min_scale=.45,max_scale=2.2,threshold=.94,cancel=cancel)
            for reference in self.item_variants)

    def _search(self, image, name, cancel, *, world=False):
        base = image.shape[0]/1080
        sizes = np.unique(np.r_[np.geomspace(base*.55,base*2.3,32),base]) if world else (
            np.unique(np.r_[np.linspace(base*.8,base*1.2,13),base]))
        names = (name,) + tuple(name+'_'+variant for variant in ('wide','rim')
            if world and name+'_'+variant in self.references)
        # Complete pens must lie between the HUD margins checked below. Keep
        # extra room for animal artwork while avoiding full-frame HUD searches.
        # Preserve the frame width and even row offset for the matcher's 2x
        # downsampling, so native screenshot references retain their phase.
        top = 2*round(image.shape[0]*.12/2) if world else 0
        bottom = round(image.shape[0]*.90) if world else image.shape[0]
        found = [VisualTarget(target.x,target.y+top,target.width,target.height,target.score)
                 for variant in names
                 for target in self.matcher._search(image[top:bottom],self.references[variant],sizes,
                    .93 if world else .94,
                    cancel,max_peaks=6 if world else 3)]
        return self._distinct(found)

    @staticmethod
    def _distinct(targets):
        result = []
        for target in sorted(targets,key=lambda item:item.score,reverse=True):
            if not any(np.linalg.norm(np.subtract(target.center,old.center))
                       < max(15,min(target.width,old.width)*.75) for old in result):
                result.append(target)
        return tuple(result)

    @staticmethod
    def same_pen(first, second):
        old,new = np.array(first,np.float32),np.array(second,np.float32)
        intersection,_ = cv2.intersectConvexConvex(old,new)
        return intersection/max(1.,cv2.contourArea(old)+cv2.contourArea(new)-intersection) >= .86

    def _feeds(self, image, cancel):
        return tuple((target,enabled) for name,enabled in (('feed',True),('feed_inactive',False))
                     for target in self._search(image,name,cancel))

    def _bucket_enabled(self, image, bucket):
        """The metal body matches in both states; its orange handle does not."""
        reference = self.references['bucket']
        hsv = cv2.cvtColor(reference[:,:,:3],cv2.COLOR_BGR2HSV)
        colored = ((reference[:,:,3] > 0) & (hsv[:,:,0] < 35)
                   & (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100)).astype(np.uint8)
        mask = cv2.resize(colored,(bucket.width,bucket.height),interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask,np.ones((3,3),np.uint8)) > 0
        patch = image[bucket.y:bucket.y+bucket.height,bucket.x:bucket.x+bucket.width]
        if patch.shape[:2] != mask.shape or np.count_nonzero(mask) < 30:
            return False
        pixels = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)[mask]
        return np.mean((pixels[:,0] < 35) & (pixels[:,1] > 70) & (pixels[:,2] > 70)) >= .60

    def menu_ground(self, frame, cancel):
        """Choose grass outside the translucent cow tool/accessory wedge."""
        from hayday.camera import CameraNavigator
        image = _decode(frame.png)
        feed = self._menu_feed(image,cancel)
        if feed is None:
            return None
        source = self.manifest['features']['feed_inactive']['box']
        scale = feed.width/(source[2]-source[0])
        # Grass visible through the wedge still belongs to a menu hit area.
        # The accessories control extends left and below the milk bucket.
        left = max(0,round(feed.center[0]-450*scale))
        return CameraNavigator._grass_start(frame,0,0,
            exclude_regions=((left,0,frame.width-left,frame.height),))

    def menu_visible(self, png, cancel):
        return self._menu_feed(_decode(png),cancel) is not None

    def _menu_feed(self, image, cancel):
        feeds = self._feeds(image,cancel)
        if len(feeds) != 1:
            return None
        feed,_ = feeds[0]
        source = self.manifest['features']['feed_inactive']['box']
        scale = feed.width/(source[2]-source[0])
        fx,fy = (source[0]+source[2])/2,(source[1]+source[3])/2
        bx,by = self.manifest['tool_point']
        center = (feed.center[0]+(bx-fx)*scale,feed.center[1]+(by-fy)*scale)
        # The same feed bag appears in the Feed Mill. Only the adjacent milk
        # bucket (enabled or grey) identifies a cow-pen menu for dismissal.
        buckets = [b for b in self._search(image,'bucket',cancel)
                   if np.linalg.norm(np.subtract(b.center,center)) < 18*scale]
        return feed if len(buckets) == 1 else None

    def enclosures(self, png, cancel, *, pen_hint=None):
        image = _decode(png)
        found = []
        troughs = self._search(image,'trough',cancel,world=True)
        old_troughs = ()
        if pen_hint is not None:
            old,previous = _decode(pen_hint[0]),pen_hint[1]
            if old.shape == image.shape and len(previous) == 4:
                old_troughs = self._search(old,'trough',cancel,world=True)
        for trough in troughs:
            polygon = SheepVision.pen(image,trough,fence='wood')
            if not polygon and old_troughs:
                # Opening a pen can expose a timer over its lower fence. Keep
                # the just-observed boundary only if the trough and a majority
                # of the enclosed world independently remain in place.
                stable = any(np.linalg.norm(np.subtract(t.center,trough.center)) <= 5*image.shape[0]/1080
                             and abs(t.width-trough.width) <= 3*image.shape[0]/1080 for t in old_troughs)
                mask = np.zeros(image.shape[:2],np.uint8)
                cv2.fillConvexPoly(mask,np.array(previous,np.int32),255)
                pixels = np.max(np.abs(image.astype(np.int16)-old.astype(np.int16)),axis=2)[mask > 0]
                if (stable and len(pixels) and np.mean(pixels < 30) >= .55
                        and cv2.pointPolygonTest(np.array(previous,np.int32),trough.center,True) >= -trough.width*.25):
                    polygon = previous
            if not polygon:
                continue
            points = np.array(polygon,np.int32)
            # A partial enclosure touching the viewport/HUD cannot authorize
            # a sweep whose unseen portion might cross a different pen.
            h,w = image.shape[:2]
            if (points[:,0].min() < .15*w or points[:,0].max() > .89*w
                    or points[:,1].min() < .18*h or points[:,1].max() > .84*h):
                continue
            mask = np.zeros(image.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask,points,255)
            soil = cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(18,100,100),(32,255,245))
            mask &= soil
            distance = cv2.distanceTransform(mask,cv2.DIST_L2,5)
            _,radius,_,point = cv2.minMaxLoc(distance)
            if radius >= max(4,trough.width*.08):
                if not any(self.same_pen(polygon,old[2]) for old in found):
                    found.append((trough,VisualTarget(point[0]-5,point[1]-5,10,10,trough.score),polygon))
        return tuple(found)

    def harvest(self, png, icon, cancel, *, feeding=False, pen_hint=None):
        from hayday.fruit import FruitTarget
        if not self.supports(icon,cancel):
            return None
        image = _decode(png)
        feeds = self._feeds(image,cancel)
        if len(feeds) != 1:
            return None
        feed,enabled = feeds[0]
        source = self.manifest['features']['feed_inactive']['box']
        scale = feed.width/(source[2]-source[0])
        fx,fy = (source[0]+source[2])/2,(source[1]+source[3])/2
        bx,by = self.manifest['tool_point']
        bucket_center = (feed.center[0]+(bx-fx)*scale,feed.center[1]+(by-fy)*scale)
        buckets = [b for b in self._search(image,'bucket',cancel)
                   if np.linalg.norm(np.subtract(b.center,bucket_center)) < 18*scale
                   and self._bucket_enabled(image,b)]
        if not feeding and len(buckets) != 1:
            return None
        # Feeding can use an inactive bucket; its own cow artwork identifies
        # the tool, and an enabled colored bag is required before input.
        x,y,w,h = self.manifest['feed_count_box']
        available = read_count(png,(round(feed.center[0]+(x-fx)*scale),
                                    round(feed.center[1]+(y-fy)*scale),round(w*scale),round(h*scale)))
        tool = feed if feeding else VisualTarget(round(bucket_center[0])-6,round(bucket_center[1])-6,12,12,buckets[0].score)
        # The shared harvest guide stores a second control for revalidation.
        # Use the actual enabled bucket, not a decorative arrow which animals
        # can obscure and which does not establish milk readiness.
        arrow = feed if feeding else buckets[0]
        guide = FruitTarget(tool,None,min(feed.score,arrow.score),scale,arrow,'milk',
                            action='feed' if feeding else 'harvest',available=available,enabled=enabled if feeding else True)
        pens = self.enclosures(png,cancel,pen_hint=pen_hint)
        supported = []
        for trough,ground,polygon in pens:
            center = np.mean(polygon,axis=0)
            if (bucket_center[0]+100*scale < center[0] < bucket_center[0]+620*scale
                    and bucket_center[1]+110*scale < center[1] < bucket_center[1]+370*scale):
                supported.append((trough,ground,polygon))
        if not supported:
            return guide
        seed = min(supported, key=lambda pen: np.linalg.norm(np.subtract(pen[0].center, feed.center)))
        trough,ground,polygon = seed
        if not feeding:
            from hayday.animal_groups import grouped_harvest
            return grouped_harvest(guide, pens, seed, .75)
        # Feeding remains scoped to the selected enclosure.
        target = trough if feeding else ground
        return FruitTarget(tool,target,guide.score,scale,arrow,'milk',(target,),
            SheepVision.sweep(polygon,trough.width*.75),polygon,guide.action,available,guide.enabled)

    def enclosure(self, png, icon, cancel):
        if not self.supports(icon,cancel):
            return None
        enclosures = self.enclosures(png,cancel)
        return (enclosures[0][1],enclosures[0][2]) if len(enclosures) == 1 else None


def _dismiss_menu(owner, vision):
    from hayday.resources import ResourceResult
    frame = owner._frame()
    ground = vision.menu_ground(frame,owner.cancel_event.is_set)
    if ground is None:
        return ResourceResult('unsupported','No clear ground was found to inspect the cow pen.')
    owner._check()
    owner.client.tap(*ground[:2],width=frame.width,height=frame.height)
    owner.cancel_event.wait(.3)
    frame = owner._frame()
    if vision.menu_visible(frame.png,owner.cancel_event.is_set):
        return ResourceResult('changed','The cow menu remained open; no camera drag was sent.')
    return frame


def prepare_cows(owner, frame, icon, key):
    """Expose one complete cow pen, then reopen its controls from fresh soil."""
    from hayday.camera import CameraNavigator
    from hayday.resources import ResourceResult
    vision = owner.vision._cow
    frame = _dismiss_menu(owner,vision)
    if isinstance(frame,ResourceResult):
        return frame
    deadline = time.monotonic()+40
    for step in range(4):
        owner._check()
        if time.monotonic() >= deadline:
            break
        candidates = vision.enclosures(frame.png,owner.cancel_event.is_set)
        if candidates:
            _,_,polygon = candidates[0]
            fresh = owner._frame()
            observed_at = time.monotonic()
            checked = [(t,g,p) for t,g,p in vision.enclosures(fresh.png,owner.cancel_event.is_set)
                       if vision.same_pen(p,polygon)]
            if len(checked) != 1 or time.monotonic()-observed_at > 3:
                return ResourceResult('changed','The cow pen moved before its controls could be reopened.')
            _,target,_ = checked[0]
            owner._cow_pen_hint = (fresh.png,checked[0][2])
            owner._check()
            owner.client.tap(*target.center,width=fresh.width,height=fresh.height)
            owner.cancel_event.wait(.4)
            opened = owner._frame()
            observed = owner._read(opened,icon)
            if observed and observed.species == 'milk':
                if observed.action == 'harvest' and observed.target is not None:
                    return opened,observed
                if observed.action == 'harvest':
                    return ResourceResult('waiting','Milk is ready, but the complete pen is obscured; checking other requirements.',
                        {'item':key,'defer_item':True,'retry_after_seconds':60})
                fed = owner._feed_animal(opened,icon,key,'cow')
                if owner.state['items'].get(key,{}).get('feed_stage') == 'attempted':
                    return ResourceResult('unsupported','Cow feeding remains unconfirmed; the saved intent prevents another sweep.')
                # A growth timer is a cream overlay which board recovery treats
                # as an unknown dialog. Close the verified cow menu before
                # yielding to another order, even when no feed was available.
                dismissed = _dismiss_menu(owner,vision)
                if isinstance(dismissed,ResourceResult):
                    return dismissed
                return ResourceResult('waiting',
                    'Cow feed was used and its stock decrease confirmed; checking other requirements.' if fed else
                    'No ready milk was verified in this cow pen; checking other requirements.',
                    {'item':key,'defer_item':True,'retry_after_seconds':600})
            return ResourceResult('changed','The reopened cow controls could not be verified.')
        if step == 3 or CameraNavigator._modal_visible(frame):
            break
        drag = CameraNavigator._grass_start(frame,-.22,0)
        if drag is None:
            break
        owner._check()
        owner.progress('Moving the partially visible cow pen into view.')
        owner.client.swipe(*drag,width=frame.width,height=frame.height,duration_ms=650)
        owner.cancel_event.wait(.4)
        frame = owner._frame()
    return ResourceResult('waiting','A complete cow pen was not verified during the bounded search; checking other requirements.',
                          {'item':key,'defer_item':True,'retry_after_seconds':600})

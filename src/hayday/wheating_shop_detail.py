"""Cloud-tolerant shop texture evidence with bounded color/contrast checks."""
import cv2
import numpy as np


def detail_image(image, scale=1.):
    """Remove slow lighting changes while retaining the rim and plank edges."""
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.clip(128+gray-cv2.GaussianBlur(gray, (0, 0), 6*scale), 0, 255).astype(np.uint8)
    color = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)
    return np.dstack((color, image[:, :, 3])) if image.shape[2] == 4 else color


def plausible_lighting(image, target, refs, matcher):
    """Texture alone is insufficient: both parts must still have wood colors.

    A translucent cloud changes local brightness and contrast. Fit that limited
    change independently to each part; unrelated colors or a missing/flat part
    cannot satisfy both the texture match and this original-pixel check.
    """
    for name in ('counter', 'deck'):
        rgba = refs[name]
        color, mask = matcher._scaled(rgba, target.width/rgba.shape[1])
        patch = image[target.y:target.y+len(mask), target.x:target.x+mask.shape[1]]
        if patch.shape != color.shape:
            return False
        expected = color[mask > 0].astype(np.float32)
        observed = patch[mask > 0].astype(np.float32)
        a, b = expected-expected.mean(axis=0), observed-observed.mean(axis=0)
        variance = float(np.sum(a*a))
        if variance < 1:
            return False
        gain = float(np.sum(a*b)/variance)
        bias = (observed-gain*expected).mean(axis=0)
        residual = float(np.abs(observed-gain*expected-bias).mean())
        if not (.6 <= gain <= 1.35 and bias.min() >= -25 and bias.max() <= 115
                and np.ptp(bias) <= 45 and residual <= 25):
            return False
    return True

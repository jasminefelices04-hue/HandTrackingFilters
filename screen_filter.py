import time
import cv2
import numpy as np


# ---------------------------------------------------------------- filters
def f_dual_tone(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255.0
    dark = np.array([140, 0, 90], np.float32)
    light = np.array([0, 210, 255], np.float32)
    return (dark * (1 - g) + light * g).astype(np.uint8)

def f_cartoon(img):
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 7, 150, 150)
    return cv2.bitwise_and(color, color, mask=edges)

def f_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(gray, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)

def f_pixelate(img, block=18):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def f_glitch(img):
    out = img.copy()
    out[..., 2] = np.roll(img[..., 2], 10, axis=1)     # red shifted right
    out[..., 0] = np.roll(img[..., 0], -10, axis=1)    # blue shifted left
    h = img.shape[0]
    for _ in range(6):                                 # random shifted bands
        y = np.random.randint(0, h - 20)
        bh = np.random.randint(5, 20)
        out[y:y + bh] = np.roll(out[y:y + bh], np.random.randint(-40, 40), axis=1)
    return out

def f_invert(img):
    return 255 - img

def f_red_channel(img):
    out = np.zeros_like(img)
    out[..., 2] = img[..., 2]
    return out

def f_edge(img):
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)
    out = np.zeros_like(img)
    out[edges > 0] = (0, 255, 255)                     # yellow lines on black
    return out

def f_blur(img):
    return cv2.GaussianBlur(img, (31, 31), 0)

def f_thermal(img):
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

def f_rainbow(img):
    """Moving diagonal rainbow stripes (a hint of the picture shows through)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    hue = ((xx + yy) * 0.45 + time.time() * 90) % 180
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (150 + 0.3 * gray).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def f_stencil(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast and create pure black and white
    _, result = cv2.threshold(
    gray,
    100,
    255,
    cv2.THRESH_BINARY
)

    # Convert back to BGR for the camera display
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


FILTERS = [
    ("DUAL-TONE", f_dual_tone), ("CARTOON", f_cartoon), ("SKETCH", f_sketch),
    ("PIXELATE", f_pixelate), ("GLITCH", f_glitch), ("INVERT", f_invert),
    ("RED-CHANNEL", f_red_channel), ("EDGE", f_edge), ("BLUR", f_blur),
    ("THERMAL", f_thermal), ("RAINBOW-WAVE", f_rainbow), ("STENCIL", f_stencil),
]

# Landmark indexes
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, PINKY_TIP = 0, 4, 8, 9, 20
FINGERTIPS = [4, 8, 12, 16, 20]

MODES = ["RECTANGLE 2D", "3D (all 5 fingers)"]


class ScreenFilter:
    """A shape between two hands. The camera picture INSIDE the shape gets a
    filter; everything outside stays normal.

      RECTANGLE 2D ........ 4 corners = thumb tip + index tip of each hand
      3D (all 5 fingers) .. outline around all 10 fingertips (both hands)
    """

    def __init__(self, start_filter=0):
        self.filter_idx = start_filter
        self.mode_idx = 1                 # start in 3D, like the video
        self.cooldown = 0.8               # seconds between filter changes
        self.smooth_pts = None            # 10 smoothed fingertip points
        self.was_pinching = False
        self.last_change = 0.0

    # ---- controls -------------------------------------------------------
    @property
    def filter_name(self):
        return FILTERS[self.filter_idx][0]

    @property
    def mode_name(self):
        return MODES[self.mode_idx]

    def next_filter(self):
        self.filter_idx = (self.filter_idx + 1) % len(FILTERS)

    def prev_filter(self):
        self.filter_idx = (self.filter_idx - 1) % len(FILTERS)

    def toggle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(MODES)

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _xy(hand, idx, w, h):
        return np.array([hand[idx].x * w, hand[idx].y * h], np.float32)

    def _is_pinky_pinch(self, hand, w, h):
        """Thumb tip touching pinky tip, measured relative to palm size."""
        palm = np.linalg.norm(self._xy(hand, WRIST, w, h) - self._xy(hand, MIDDLE_MCP, w, h))
        d = np.linalg.norm(self._xy(hand, THUMB_TIP, w, h) - self._xy(hand, PINKY_TIP, w, h))
        return palm > 1 and d / palm < 0.35

    def _fingertips(self, hands, w, h):
        """10 points: the 5 fingertips of the left hand, then of the right."""
        ordered = sorted(hands, key=lambda hd: hd[WRIST].x)     # left hand first
        return np.array([self._xy(hd, i, w, h) for hd in ordered for i in FINGERTIPS],
                        np.float32)

    def _shape(self):
        """Polygon (int32) for the current mode, from the smoothed fingertips."""
        p = self.smooth_pts
        if self.mode_idx == 0:                                  # RECTANGLE 2D
            left = sorted([p[0], p[1]], key=lambda q: q[1])     # thumb, index (top first)
            right = sorted([p[5], p[6]], key=lambda q: q[1])
            quad = np.float32([left[0], right[0], right[1], left[1]])
            return quad.astype(np.int32)
        return cv2.convexHull(p.astype(np.int32))[:, 0, :]      # 3D: all 10 tips

    def _apply(self, frame, poly):
        h, w = frame.shape[:2]
        filtered = FILTERS[self.filter_idx][1](frame)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        out = frame.copy()
        out[mask > 0] = filtered[mask > 0]                      # filter IN PLACE
        cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # ---- main entry point: call once per frame --------------------------
    def update(self, frame, hands):
        """Handle the thumb-pinky touch and draw the shape when 2 hands are seen.
        Returns the frame (a new array when the shape is drawn)."""
        h, w = frame.shape[:2]

        # thumb-pinky touch -> next filter (edge-triggered + delay)
        pinching = any(self._is_pinky_pinch(hd, w, h) for hd in hands)
        if pinching and not self.was_pinching \
                and time.time() - self.last_change > self.cooldown:
            self.next_filter()
            self.last_change = time.time()
        self.was_pinching = pinching

        if len(hands) == 2:
            if not pinching:                       # shape holds still while pinching
                pts = self._fingertips(hands, w, h)
                self.smooth_pts = pts if self.smooth_pts is None \
                    else 0.5 * self.smooth_pts + 0.5 * pts
            if self.smooth_pts is not None:
                frame = self._apply(frame, self._shape())
        else:
            self.smooth_pts = None

        return frame
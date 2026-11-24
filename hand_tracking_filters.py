"""
Hand tracking + floating filter window (MediaPipe Tasks + OpenCV)

Hold up BOTH hands: the shape between them shows the camera with a filter.

Controls
  Touch thumb to pinky (either hand) ... next filter (shape holds still)
  f or SPACE ........................... next filter
  m .................................... toggle mode (WINDOW / MASK)
  d .................................... show/hide hand skeleton
  q or ESC ............................. quit
"""
import time
import cv2
import numpy as np
import mediapipe as mp

# ------------------------------------------------------------------ setup
model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ------------------------------------------------------------ hand drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = {4, 8, 12, 16, 20}
LINE_COLOR = (255, 255, 255)
DOT_COLOR = (255, 0, 0)          # blue (BGR)
DOT_RING_COLOR = (255, 255, 255)


def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], LINE_COLOR, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 7 if i in FINGERTIPS else 5
        cv2.circle(frame, (x, y), r, DOT_RING_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r - 2, DOT_COLOR, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- filters
def f_none(img):
    return img

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
    out[..., 2] = np.roll(img[..., 2], 10, axis=1)
    out[..., 0] = np.roll(img[..., 0], -10, axis=1)
    h = img.shape[0]
    for _ in range(6):
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
    out[edges > 0] = (0, 255, 255)
    return out

def f_blur(img):
    return cv2.GaussianBlur(img, (31, 31), 0)

def f_thermal(img):
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

def f_rainbow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    xs = np.linspace(0, 180, w, dtype=np.float32)[None, :]
    shift = (xs + time.time() * 60) % 180
    hsv[..., 0] = ((hsv[..., 0].astype(np.float32) + shift) % 180).astype(np.uint8)
    hsv[..., 1] = np.maximum(hsv[..., 1], 150)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

FILTERS = [
    ("NONE", f_none), ("DUAL-TONE", f_dual_tone), ("CARTOON", f_cartoon),
    ("SKETCH", f_sketch), ("PIXELATE", f_pixelate), ("GLITCH", f_glitch),
    ("INVERT", f_invert), ("RED-CHANNEL", f_red_channel), ("EDGE", f_edge),
    ("BLUR", f_blur), ("THERMAL", f_thermal), ("RAINBOW-WAVE", f_rainbow),
]

# ------------------------------------------------------------ hand -> shape
WRIST, THUMB_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_TIP = 0, 4, 9, 12, 20


def xy(hand, idx, w, h):
    return np.array([hand[idx].x * w, hand[idx].y * h], np.float32)


def is_pinky_pinch(hand, w, h):
    """Thumb tip touching pinky tip, measured relative to palm size."""
    palm = np.linalg.norm(xy(hand, WRIST, w, h) - xy(hand, MIDDLE_MCP, w, h))
    d = np.linalg.norm(xy(hand, THUMB_TIP, w, h) - xy(hand, PINKY_TIP, w, h))
    return palm > 1 and d / palm < 0.35


def build_quad(hands, w, h):
    """Corners TL, TR, BR, BL from two hands (wrist + middle fingertip each)."""
    pairs = []
    for hand in hands:
        pts = [xy(hand, WRIST, w, h), xy(hand, MIDDLE_TIP, w, h)]
        pts.sort(key=lambda p: p[1])                    # higher point first
        pairs.append(pts)
    pairs.sort(key=lambda pr: pr[0][0] + pr[1][0])      # left hand first
    (tl, bl), (tr, br) = pairs
    return np.float32([tl, tr, br, bl])


def apply_effect(frame, quad, filter_fn, mode_window):
    """Draw the filtered 'screen' inside the quad."""
    h, w = frame.shape[:2]
    poly = quad.astype(np.int32)
    filtered = filter_fn(frame)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)       # fillPoly also handles twisted shapes

    if mode_window:    # squeeze the whole filtered frame into the shape
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, quad)
        content = cv2.warpPerspective(filtered, M, (w, h))
    else:              # reveal the filter under the shape
        content = filtered

    out = frame.copy()
    out[mask > 0] = content[mask > 0]
    cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    filter_idx, mode_window, show_skeleton = 1, True, True
    smooth_quad, was_pinching, last_pinch = None, False, 0.0

    with HandLandmarker.create_from_options(options) as detector:
        camera = cv2.VideoCapture(0)
        start = time.time()

        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)                    # mirror
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int((time.time() - start) * 1000)
            results = detector.detect_for_video(mp_image, timestamp)
            hands = results.hand_landmarks or []

            # thumb-pinky touch -> next filter (edge-triggered + cooldown)
            pinching = any(is_pinky_pinch(hd, w, h) for hd in hands)
            if pinching and not was_pinching and time.time() - last_pinch > 0.8:
                filter_idx = (filter_idx + 1) % len(FILTERS)
                last_pinch = time.time()
            was_pinching = pinching

            if len(hands) == 2:
                if not pinching:                          # hold shape while pinching
                    quad = build_quad(hands, w, h)
                    smooth_quad = quad if smooth_quad is None \
                        else 0.5 * smooth_quad + 0.5 * quad
                if smooth_quad is not None:
                    frame = apply_effect(frame, smooth_quad,
                                         FILTERS[filter_idx][1], mode_window)
            else:
                smooth_quad = None

            if show_skeleton:
                for hd in hands:
                    draw_hand(frame, hd)

            cv2.putText(frame, f"MODE: {'WINDOW' if mode_window else 'MASK'}  [m]",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FILTER: {FILTERS[filter_idx][0]}  [touch thumb-pinky / f]",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Python Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("f"), 32):
                filter_idx = (filter_idx + 1) % len(FILTERS)
            elif key == ord("m"):
                mode_window = not mode_window
            elif key == ord("d"):
                show_skeleton = not show_skeleton

        camera.release()
        cv2.destroyAllWindows()
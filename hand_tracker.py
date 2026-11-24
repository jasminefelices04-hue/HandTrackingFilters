import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandTracker:

    # Pairs of landmark indexes joined by lines (standard 21-point skeleton)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),            # index finger
        (5, 9), (9, 10), (10, 11), (11, 12),       # middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),     # ring finger
        (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
        (0, 17),                                   # palm edge
    ]
    FINGERTIPS = {4, 8, 12, 16, 20}

    # Look & feel (BGR colors)
    LINE_COLOR = (255, 255, 255)        # white lines
    LINE_THICKNESS = 2
    DOT_COLOR = (255, 0, 0)             # blue dot centers
    DOT_RING_COLOR = (255, 255, 255)    # white ring around each dot

    def __init__(self, model_path, num_hands=2, confidence=0.7):

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )

        self.detector = HandLandmarker.create_from_options(options)
        self._last_timestamp = -1

    def detect_hands(self, frame, timestamp):
        """Return a list of hands; each hand is a list of 21 landmarks
        (empty list when no hand is visible)."""

        # MediaPipe needs timestamps that strictly increase
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1
        self._last_timestamp = timestamp

        # Convert BGR to RGB, then wrap for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.detector.detect_for_video(mp_image, timestamp)

        return results.hand_landmarks or []

    def draw_hand(self, frame, hand_landmarks):
        """Draw white lines and ringed dots for one hand, in place."""

        height, width, _ = frame.shape

        points = [
            (int(lm.x * width), int(lm.y * height))
            for lm in hand_landmarks
        ]

        # Lines first, so the dots sit on top of them
        for a, b in self.HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b],
                     self.LINE_COLOR, self.LINE_THICKNESS, cv2.LINE_AA)

        # Dots: white ring + colored center (fingertips a little bigger)
        for i, (x, y) in enumerate(points):
            radius = 7 if i in self.FINGERTIPS else 5
            cv2.circle(frame, (x, y), radius, self.DOT_RING_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius - 2, self.DOT_COLOR, -1, cv2.LINE_AA)

    def close(self):
        self.detector.close()
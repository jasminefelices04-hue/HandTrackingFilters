
import cv2
import time

from effects import VisualEffects
from hand_tracker import HandTracker
from gestures import GestureRecognizer
from screen_filter import ScreenFilter


WINDOW = "RETROLENS"

# Load hand tracking model
model_path = "hand_landmarker.task"

# Create hand tracker
tracker = HandTracker(model_path, num_hands=2)

# Create gesture recognizer
recognizer = GestureRecognizer()

# Create visual effects
effects = VisualEffects()

# Create the filter shape (shown between two hands)
screen = ScreenFilter()

# Window + slider for the filter-change delay
cv2.namedWindow(WINDOW)
cv2.createTrackbar(
    "Filter delay (x0.1s)",
    WINDOW,
    8,
    30,
    lambda v: None
)

# Open webcam
camera = cv2.VideoCapture(0)

start = time.time()
both_fists_before = False
last_mode_change = 0.0

try:

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Generate timestamp
        timestamp = int((time.time() - start) * 1000)

        # Detect hands
        hands = tracker.detect_hands(
            frame,
            timestamp
        )

        # Recognize the gesture of every hand
        gestures = [recognizer.recognize(hand) for hand in hands]

        # Both hands made into fists -> switch mode
        both_fists = (
            len(hands) == 2
            and all(g == "FIST" for g in gestures)
        )

        if (
            both_fists
            and not both_fists_before
            and time.time() - last_mode_change > 1.0
        ):
            screen.toggle_mode()
            last_mode_change = time.time()

        both_fists_before = both_fists

        # Slider -> filter-change delay
        screen.cooldown = max(
            0.1,
            cv2.getTrackbarPos(
                "Filter delay (x0.1s)",
                WINDOW
            ) / 10
        )

        # Draw filtered shape between two hands
        frame = screen.update(frame, hands)

        # Draw every hand without displaying gesture names
        for hand_landmarks in hands:

            # Draw hand skeleton
            tracker.draw_hand(
                frame,
                hand_landmarks
            )

            # Apply glowing fingertip effect
            frame = effects.glowing_fingertips(
                frame,
                hand_landmarks
            )

        # Show current mode
        cv2.putText(
            frame,
            f"MODE: {screen.mode_name} [C / 2 fists]",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Show current filter
        cv2.putText(
            frame,
            f"FILTER: {screen.filter_name} [N / P]",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Display camera
        cv2.imshow(WINDOW, frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            screen.toggle_mode()
        elif key in (ord("n"), ord("f"), 32):
            screen.next_filter()
        elif key == ord("p"):
            screen.prev_filter()

finally:

    camera.release()
    tracker.close()
    cv2.destroyAllWindows()

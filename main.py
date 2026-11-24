
import cv2
import time

from effects import VisualEffects
from hand_tracker import HandTracker
from gestures import GestureRecognizer


# Load hand tracking model
model_path = "hand_landmarker.task"

# Create hand tracker
tracker = HandTracker(model_path, num_hands=2)

# Create gesture recognizer
recognizer = GestureRecognizer()

# Create visual effects
effects = VisualEffects()

# Open webcam
camera = cv2.VideoCapture(0)

start = time.time()

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

        # Process detected hands
        if hands:

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

                # Recognize gesture without displaying its name
                gesture = recognizer.recognize(
                    hand_landmarks
                )

        # Display camera
        cv2.imshow(
            "Python Hand Tracking",
            frame
        )

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    camera.release()
    tracker.close()
    cv2.destroyAllWindows()

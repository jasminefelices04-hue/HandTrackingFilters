import math


class GestureRecognizer:

    def _dist(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _extended(self, landmarks, tip, pip):
        """A finger is extended if its tip is farther from the wrist than
        its middle joint. Unlike comparing y values, this still works when
        the hand is tilted, sideways or upside down."""
        wrist = landmarks[0]
        return self._dist(landmarks[tip], wrist) > self._dist(landmarks[pip], wrist)

    def recognize(self, landmarks):

        # Check whether each finger is extended
        index_finger = self._extended(landmarks, 8, 6)
        middle_finger = self._extended(landmarks, 12, 10)
        ring_finger = self._extended(landmarks, 16, 14)
        pinky_finger = self._extended(landmarks, 20, 18)

        # 1. Open Palm
        if (
            index_finger
            and middle_finger
            and ring_finger
            and pinky_finger
        ):
            return "OPEN PALM"

        # 2. Fist
        elif (
            not index_finger
            and not middle_finger
            and not ring_finger
            and not pinky_finger
        ):
            return "FIST"

        # 3. Peace Sign
        elif (
            index_finger
            and middle_finger
            and not ring_finger
            and not pinky_finger
        ):
            return "PEACE SIGN"

        # 4. Pointing (only the index finger up)
        elif (
            index_finger
            and not middle_finger
            and not ring_finger
            and not pinky_finger
        ):
            return "POINTING"

        else:
            return "UNKNOWN"
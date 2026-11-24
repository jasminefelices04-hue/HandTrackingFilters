import cv2
import numpy as np


class VisualEffects:

    def __init__(self):

        # Fingertip landmark indexes
        self.FINGERTIPS = [4, 8, 12, 16, 20]

    def glowing_fingertips(self, frame, landmarks):

        height, width, _ = frame.shape

        # Draw the glow on a black canvas, so only the glow gets blurred
        # (not the camera picture underneath)
        glow = np.zeros_like(frame)

        points = []

        for index in self.FINGERTIPS:

            # Get fingertip position
            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

            # Outer glow
            cv2.circle(glow, (x, y), 25, (255, 100, 0), -1, cv2.LINE_AA)

            # Inner glow
            cv2.circle(glow, (x, y), 12, (255, 200, 100), -1, cv2.LINE_AA)

        # Blur the canvas so the circles melt into a soft light
        glow = cv2.GaussianBlur(glow, (0, 0), 14)

        # Add the light on top of the camera picture
        frame = cv2.add(frame, glow)

        # Crisp bright center, drawn after the blur so it stays sharp
        for (x, y) in points:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)

        return frame
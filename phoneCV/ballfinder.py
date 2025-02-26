import cv2
import numpy as np

def detect_red_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range (tweak these values if needed)
    lower_red1 = np.array([0, 120, 70])   # Lower range of red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70]) # Upper range of red
    upper_red2 = np.array([180, 255, 255])

    # Threshold image to detect red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours of red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_positions = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2  # Get center of object
            detected_positions.append((cx, cy, w, h))

            # Draw rectangle (for debugging)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, detected_positions

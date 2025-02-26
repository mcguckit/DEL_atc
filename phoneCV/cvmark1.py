import cv2
import numpy as np
import os
import shutil

# Function to clear output folders before rerunning
def clear_folder(folder):
    """Deletes all files in a folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Deletes the entire folder
    os.makedirs(folder, exist_ok=True)  # Recreate the empty folder

# Function to detect red objects in a frame
def detect_red_objects(image):
    """Detect red objects (balls) in a given frame using HSV filtering."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red1 = np.array([0, 120, 70])   # Lower red (hue 0-10)
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70]) # Upper red (hue 170-180)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Apply morphological operations to reduce noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return mask

def process_frames(frame_folder, output_folder):
    """Process all frames in a folder and detect red balls."""
    clear_folder(output_folder)
    frame_files = sorted(os.listdir(frame_folder))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error loading frame: {frame_file}")
            continue

        # Detect red objects
        mask = detect_red_objects(frame)

        # Find contours of detected objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_positions = []

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2  # Center of detected object
                detected_positions.append((cx, cy))

                # Draw bounding box and center point
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Save the processed frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)

        # Print detections for debugging
        print(f"{frame_file}: {detected_positions}")

    print(f"Processed frames saved in {output_folder}")

# Usage
current_dir = os.path.dirname(os.path.abspath(__file__))
frames_folder = os.path.join(current_dir, "frames_synced")  # Folder containing frames
frame_folder = os.path.join(frames_folder, "frames_saanvi_one_synced")  # Folder with frames to process
output_folder = os.path.join(current_dir, "processed_frames")  # Folder to save processed frames

process_frames(frame_folder, output_folder)

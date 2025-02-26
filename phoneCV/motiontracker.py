import cv2
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frame_folder = os.path.join(current_dir, "frames_synced", "frames_saanvi_one_synced")

# Create Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

frame_files = sorted(os.listdir(frame_folder))  # Sorted list of frames

for frame_file in frame_files:
    frame_path = os.path.join(frame_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error loading frame: {frame_file}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Show results
    cv2.imshow("Motion Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

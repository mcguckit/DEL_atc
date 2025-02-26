import cv2
import os
import screeninfo

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frame_folder = os.path.join(current_dir, "frames_synced", "frames_saanvi_one_synced")

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Load frames
frame_files = sorted(os.listdir(frame_folder))
first_frame = os.path.join(frame_folder, frame_files[0])
frame = cv2.imread(first_frame)

if frame is None:
    print(f"Error loading first frame: {first_frame}")
    exit()

# Get screen width & height
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Resize the frame if it's too large
if frame.shape[1] > screen_width or frame.shape[0] > screen_height:
    scale_x = screen_width / frame.shape[1]
    scale_y = screen_height / frame.shape[0]
    scale = min(scale_x, scale_y) * 0.9  # Scale down slightly to fit
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)    

# Convert to grayscale and apply motion detection
fgmask = fgbg.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

# Find contours of moving objects
contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea) if contours else None

if largest_contour is None:
    print("❌ No motion detected in the first frame.")
    exit()

# Get bounding box from the largest moving object
x, y, w, h = cv2.boundingRect(largest_contour)
bbox = (x, y, w, h)

# Initialize CSRT Tracker with the detected bounding box
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, bbox)

# Process frames
for frame_file in frame_files[1:]:  # Skip first frame
    frame_path = os.path.join(frame_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error loading frame: {frame_file}")
        continue

    # Apply motion detection
    fgmask = fgbg.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Update CSRT tracker
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw tracked object

    # Show results
    cv2.imshow("Motion Mask", fgmask)
    cv2.imshow("CSRT Auto-Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

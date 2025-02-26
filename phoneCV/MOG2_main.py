import cv2
import os
import shutil
import screeninfo

def clear_folder(folder):
    """Deletes all files in a folder before processing."""
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Deletes the entire folder
    os.makedirs(folder, exist_ok=True)  # Recreate the empty folder

def process_mog2_tracking(input_folder, output_folder):
    """Applies MOG2 background subtraction and saves processed frames with contours."""
    # Initialize background subtractor (MOG2)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # Ensure output directory is clean
    clear_folder(output_folder)

    # Load frames
    frame_files = sorted(os.listdir(input_folder))

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error loading frame: {frame_file}")
            continue

        # Get screen width & height
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height

        # Resize frame if it's too large
        if frame.shape[1] > screen_width or frame.shape[0] > screen_height:
            scale_x = screen_width / frame.shape[1]
            scale_y = screen_height / frame.shape[0]
            scale = min(scale_x, scale_y) * 0.9  # Scale down slightly to fit
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Convert to grayscale and apply motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (biggest moving object)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 100:  # Ignore small objects
                # Bounding Box
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Bounding Ellipse (Optional)
                if len(largest_contour) >= 5:  # Fit ellipse needs at least 5 points
                    ellipse = cv2.fitEllipse(largest_contour)
                    cv2.ellipse(frame, ellipse, (255, 0, 0), 2)

        # Save processed frame to output folder
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)

        # Show results
        cv2.imshow("Motion Mask", fgmask)
        cv2.imshow("MOG2 Tracking", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"✅ Processed frames saved in {output_folder}")

# ========= MAIN EXECUTION =========
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output locations for both datasets
    datasets = {
        "saanvi": {
            "input": os.path.join(current_dir, "frames_synced", "frames_saanvi_one_synced"),
            "output": os.path.join(current_dir, "mog2_processed", "frames_saanvi_one_processed")
        },
        "nick": {
            "input": os.path.join(current_dir, "frames_synced", "frames_nick_one_synced"),
            "output": os.path.join(current_dir, "mog2_processed", "frames_nick_one_processed")
        }
    }

    # Run processing for each dataset
    for name, paths in datasets.items():
        print(f"\n🚀 Processing {name} dataset...")
        process_mog2_tracking(paths["input"], paths["output"])

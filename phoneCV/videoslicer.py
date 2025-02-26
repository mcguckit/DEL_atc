import cv2
import os

# This script extracts frames from a video file and saves them as images.
# Each image is named with its frame number and timestamp.
def extract_frames(video_path, output_folder):
    if not os.path.exists(video_path):
        print(f"Error: No such video as {video_path} exists.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        print("Warning: FPS detection failed, assuming 30 FPS.")
        fps = 30  # Assume 30 FPS if unknown

    os.makedirs(output_folder, exist_ok=True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps  # Time in seconds
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}_{timestamp:.3f}.png")
        
        # if frame_idx % 10 == 0:  # Save every 10th frame
        cv2.imwrite(frame_filename, frame)
        
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}...")

        frame_idx += 1

    cap.release()
    print(f"Frames saved in {output_folder}")


# Frame Extraction

# Define input and output folders
input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_saanvi_one")

if not os.path.exists(input_folder):
    print(f"Error: The input folder {input_folder} does not exist.")
    exit()
# Extract frames from the video
video = os.path.join(input_folder, "flight1_saanviphone.MOV")
extract_frames(video, output_folder)


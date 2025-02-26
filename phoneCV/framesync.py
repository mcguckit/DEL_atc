import cv2
import os
import shutil

# This script synchronizes frames from two different sources (Nick1 and Saanvi1) based on a known offset.
# It also downsamples the frames to reduce the number of images for further processing.

# Nick1 spin-up frame: frame_000795_26.489.png (796th frame, 26.489s)
# Saanvi1 spin-up frame: frame_000780_26.023.png (781st frame, 26.023s)

# Config: Define frame offsets & downsample rate
START_FRAME_N1 = 796  # Nick1's first flight frame
START_FRAME_S1 = 781  # Saanvi1's first flight frame
DOWNSAMPLE_RATE = 10  # Extract every 5th frame (adjustable)

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frame_folder_n1 = os.path.join(current_dir, "frames_nick_one")
frame_folder_s1 = os.path.join(current_dir, "frames_saanvi_one")
output_folder = os.path.join(current_dir, "frames_synced")
os.makedirs(output_folder, exist_ok=True)

# Output directories for synced frames
synced_n1 = os.path.join(output_folder, "frames_nick_one_synced")
synced_s1 = os.path.join(output_folder, "frames_saanvi_one_synced")
os.makedirs(synced_n1, exist_ok=True)
os.makedirs(synced_s1, exist_ok=True)

# Function to clear output folders before rerunning
def clear_folder(folder):
    """Deletes all files in a folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Deletes the entire folder
    os.makedirs(folder, exist_ok=True)  # Recreate the empty folder

# Clear output directories
clear_folder(synced_n1)
clear_folder(synced_s1)

def sync_and_downsample_frames(input_folder, output_folder, offset, downsample_rate):
    """Syncs and downsamples frames from a folder."""
    frame_files = sorted(os.listdir(input_folder))  # Sort to ensure order

    for i, frame_file in enumerate(frame_files):
        if i < offset:  
            continue  # Skip the first N frames based on offset

        if (i - offset) % downsample_rate != 0:
            continue  # Skip frames to downsample

        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error loading frame: {frame_file}")
            continue

        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)  # Save synced & downsampled frame

        print(f"Saved synced frame: {output_path}")

# Process Nick1 & Saanvi1 frames
sync_and_downsample_frames(frame_folder_n1, synced_n1, START_FRAME_N1, DOWNSAMPLE_RATE)
sync_and_downsample_frames(frame_folder_s1, synced_s1, START_FRAME_S1, DOWNSAMPLE_RATE)

print(f"Synced & downsampled frames saved in {output_folder}")
# DEL_atc
First steps in tracking drones for automatic control of swarms in high-density GPS denied environments

Current process:

1. get the following videos from the USRC drive and put them in \videos:
    a. flight1_nickphone.mp4
    b. flight2_nickphone.mp4
    c. flight1_saanviphone.MOV
    d. flight2_saanviphone.MOV

2. run videoslicer.py
    This guy pretty simply slices out each individual frame and gives them an index and a timestamp for a name.

3. run framesync.py
    I manually went into the 3k+ frames from the videos and found where they lined up (first frame with prop spin).
    Then this file takes every 10 frames from that point to what I decided was a good time to stop (landing).
    It puts them in \frames_synced

4. run MOG2_main.py
    It uses the MOG2 alg to separate out the pixels that are changing (moving) over time.
    Then it uses the cv2 findContours to make a bounding box around the biggest moving object.
    So yeah this would NOT work for a swarm AT ALL. It literally just finds the biggest moving blob lol.


* Theres a bunch of other files but they're like me messing around and iterating with chatgpt cuz I don't trust myself to manage my work in one file sometimes. U can probably igonre most of them. motiontracker.py is fun to look at but anything that mentions csrt does not work rn.
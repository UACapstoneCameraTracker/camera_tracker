import cv2
from camera_tracker.utils import get_frame_generator

fg = get_frame_generator(mock=True)

for frame in fg:
    cv2.imshow('camera', frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

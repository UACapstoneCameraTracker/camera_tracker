import cv2
from pathlib import Path
from camera_tracker.utils import get_frame_generator

source = get_frame_generator()
fifo_path = '/home/pi/fifo_img.jpg'

for frame in source:
    cv2.imshow('app', frame)
    print(frame.shape)

    if not Path(fifo_path).exists():
        os.mkfifo(fifo_path)
    
    if frame is not None:
        cv2.imwrite(fifo_path, frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

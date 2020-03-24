import cv2
import time
import numpy as np
# from app import settings
from contextlib import suppress

with open('/home/pi/fifo_img.jpg', 'rb') as f:
    while True:
        data = f.read()
        with suppress(Exception):
            if len(data) != 0:
                print(len(data))


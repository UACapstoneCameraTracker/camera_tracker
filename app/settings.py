from setting_profiles.distance_5 import *

# debug
DISPLAY = False

# communication
IMG_FIFO_PATH = '/home/pi/fifo_img.jpg'
CMD_FIFO_PATH = '/home/pi/fifo_cmd'

CAMERA_MOVING_THRESHOLD = IMG_SIZE[0] * IMG_SIZE[1] / 2
VALID_LOC_FRAME_CNT = 3
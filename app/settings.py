IMG_SIZE = (640, 360)

# tracker
# CSRT, KCF, MEDIANFLOW, MOSSE
TRACKER_NAME = 'KCF'
IOU_THRESHOLD = 0.4
MAX_TRACKER_HEALTH = 10

# detector
BBOX_AREA_MIN_TH = 150
BBOX_AREA_MAX_TH = IMG_SIZE[0] * IMG_SIZE[1] / 10
PIXEL_DIFFERENCE_TH = 10
STRUCTURING_KERNEL_SHAPE = (5, 5)

# camera moving
CAMERA_MOVING_TH = IMG_SIZE[0] * IMG_SIZE[1] * (3 / 4)

# debug
DISPLAY = True

# communication
IMG_FIFO_PATH = '/home/pi/fifo_img.jpg'
LOC_FIFO_PATH = '/home/pi/fifo_loc.txt'

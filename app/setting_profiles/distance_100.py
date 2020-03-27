IMG_SIZE = (640, 360)

# tracker
# CSRT, KCF, MEDIANFLOW, MOSSE
TRACKER_NAME = 'KCF'
IOU_THRESHOLD = 0.4
MAX_TRACKER_HEALTH = 5
TIME_BEFORE_RECENTRE = 60

# detector
BBOX_AREA_MIN_TH = 150
BBOX_AREA_MAX_TH = IMG_SIZE[0] * IMG_SIZE[1] / 10
PIXEL_DIFFERENCE_TH = 10
STRUCTURING_KERNEL_SHAPE = (5, 5)

# camera moving
DEAD_ZONE_X = IMG_SIZE[0] / 4
DEAD_ZONE_Y = IMG_SIZE[1] / 4
import cv2
import time
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from camera_tracker.tracking_system import TrackingSystem

import settings


def setup_tracking_system():
    pre_tracker_pipe = [
        pc.ResizeTransformer(out_size=settings.IMG_SIZE)
    ]

    pre_detector_pipe = [
        pc.ResizeTransformer(out_size=settings.IMG_SIZE),
        pc.GrayscaleTransformer(),
        pc.BlurTransformer()
    ]

    detector = predictors.PixelDifferenceDetector(pixel_difference_threshold=settings.PIXEL_DIFFERENCE_TH,
                                                  structuring_kernel_shape=settings.STRUCTURING_KERNEL_SHAPE,
                                                  bbox_area_min=settings.BBOX_AREA_MIN_TH)
    tracker = predictors.CvTracker(tracker_name=settings.TRACKER_NAME,
                                   tracker_health=settings.MAX_TRACKER_HEALTH)

    tracking_sys = TrackingSystem(tracker=tracker,
                                  detector=detector,
                                  pre_tracker_pipe=pre_tracker_pipe,
                                  pre_detector_pipe=pre_detector_pipe,
                                  video_source=utils.get_frame_generator(),
                                  iou_threshold=settings.IOU_THRESHOLD,
                                  display=settings.DISPLAY)
    return tracking_sys


if __name__ == '__main__':
    tracking_sys = setup_tracking_system()
    tracking_sys.start()

    while True:
        time.sleep(3)
        frame = tracking_sys.get_video_frame()
        loc = tracking_sys.get_location()

        print('frame received' if frame is not None else 'no frame')
        print('loc received' if loc is not None else 'no location')


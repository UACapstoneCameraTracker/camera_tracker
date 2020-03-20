import os
import cv2
import time
import threading
from pathlib import Path
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from camera_tracker.tracking_system import TrackingSystem
import settings

from motor_control import gimbal


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
                                                  bbox_area_min=settings.BBOX_AREA_MIN_TH,
                                                  bbox_area_max=settings.BBOX_AREA_MAX_TH)
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


def server_communication():
    while True:
        time.sleep(0.03)
        frame = tracking_sys.get_video_frame()
        print('frame received' if frame is not None else 'no frame')
        if frame is not None:
            with open(settings.IMG_FIFO_PATH, 'wb') as fifo:
                fifo.flush()
                frame = frame.astype('uint8')
                fifo.write(frame.tobytes())


def motor_communication():
    while True:
        with tracking_sys.loc_cv:
            while not tracking_sys.loc_cv.wait():
                pass
            loc = tracking_sys.get_location()
            gimbal.move_to(loc, settings.IMG_SIZE)


if __name__ == '__main__':
    if not Path(settings.IMG_FIFO_PATH).exists():
        os.mkfifo(settings.IMG_FIFO_PATH)

    tracking_sys = setup_tracking_system()
    tracking_sys.start()

    # start server communication thread
    server_comm_thread = threading.Thread(
        target=server_communication, name='server_comm')
    server_comm_thread.start()

    motor_communication()

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
    camera_moving_detector = predictors.CameraMovingDetector(
        predictors.PixelDifferenceDetector(pixel_difference_threshold=settings.PIXEL_DIFFERENCE_TH,
                                                  structuring_kernel_shape=settings.STRUCTURING_KERNEL_SHAPE,
                                                  bbox_area_min=settings.BBOX_AREA_MIN_TH,
                                                  bbox_area_max=settings.BBOX_AREA_MAX_TH),
        settings.CAMERA_MOVING_THRESHOLD
    )

    tracking_sys = TrackingSystem(tracker=tracker,
                                  detector=detector,
                                  camera_moving_detector=camera_moving_detector,
                                  pre_tracker_pipe=pre_tracker_pipe,
                                  pre_detector_pipe=pre_detector_pipe,
                                  video_source=utils.get_frame_generator(),
                                  iou_threshold=settings.IOU_THRESHOLD,
                                  valid_loc_frame_cnt=settings.VALID_LOC_FRAME_CNT,
                                  display=settings.DISPLAY)
    return tracking_sys


def server_communication():
    while True:
        frame = tracking_sys.get_video_frame()
        if frame is not None:
            with open(settings.IMG_FIFO_PATH, 'wb') as fifo:
                fifo.flush()
                success, img = cv2.imencode('.jpg', frame)
                if success:
                    fifo.write(bytearray(img))


def server_command():
    while True:
        with open(settings.CMD_FIFO_PATH, 'r') as fifo:
            cmd = fifo.readlines()
            cmd = [c.strip() for c in cmd]
            if not cmd:
                continue
            if cmd[0] == 'manual':
                if cmd[1] == 'start':
                    tracking_sys.pause()
                elif cmd[1] == 'stop':
                    tracking_sys.resume()
            elif cmd[0] == 'select target':
                bbox = tuple([int(n) for n in cmd[1].split(',')])
                tracking_sys.set_target(bbox)
            else:
                print('unknown command:')
                print(cmd)


def motor_communication():
    gimbal.init_gimbal(settings.IMG_SIZE)
    while True:
        with tracking_sys.loc_cv:
            while not tracking_sys.loc_cv.wait():
                pass
            loc = tracking_sys.get_location()
            if (settings.IMG_SIZE[0] / 2 - settings.DEAD_ZONE_X) < loc[0] < (settings.IMG_SIZE[0] / 2 + settings.DEAD_ZONE_X) and \
                    (settings.IMG_SIZE[1] / 2 - settings.DEAD_ZONE_Y) < loc[1] < (settings.IMG_SIZE[1] / 2 + settings.DEAD_ZONE_Y):
                print(f'object ({int(loc[0])}, {int(loc[1])}) in dead zone')
                continue
            print('location received, stopping..')
            tracking_sys.pause()
            gimbal.move_to(loc)
            print('tracking system starting...')
            tracking_sys.resume()


if __name__ == '__main__':
    if not Path(settings.IMG_FIFO_PATH).exists():
        os.mkfifo(settings.IMG_FIFO_PATH)

    if not Path(settings.CMD_FIFO_PATH).exists():
        os.mkfifo(settings.CMD_FIFO_PATH)

    tracking_sys = setup_tracking_system()

    motor_thread = threading.Thread(target=motor_communication, name='motor')
    motor_thread.start()

    tracking_sys.start()

    # start server communication thread
    server_comm_thread = threading.Thread(
        target=server_communication, name='server_comm')
    server_comm_thread.start()

    server_cmd_thread = threading.Thread(
        target=server_command, name='server_cmd')
    server_cmd_thread.start()

    while True:
        time.sleep(1)

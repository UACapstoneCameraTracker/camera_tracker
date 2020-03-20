"""
This version lets the detector to run all the time,
so adjustments to the tracker can be made.
"""
import time
import threading
import cv2
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from contextlib import suppress


class TrackingSystem:
    """
    The tracking system.

    A TrackingSystem object has two outputs:
    1. location: the location of tracked object. When the location
                 is available, receiving thread(s) will be notified
                 by a condition variable.
    2. frame: the current frame received. Receiving thread(s) can 
              get the current frame by calling get_video_frame method.   
    """

    def __init__(self, *args, **kwargs):
        self.tracker = kwargs['tracker']
        self.detector = kwargs['detector']
        self.pre_tracker_pipe = kwargs['pre_tracker_pipe']
        self.pre_detector_pipe = kwargs['pre_detector_pipe']
        self.video_source = kwargs['video_source']
        self.iou_threshold = kwargs['iou_threshold']
        self.display = kwargs['display']
        self.thread = None
        self.run_lock = threading.Lock()
        self.running = False

        self.curr_frame = None
        self.frame_lock = threading.RLock()

        self.location = None
        self.loc_lock = threading.RLock()
        self.loc_cv = threading.Condition(self.loc_lock)

        self.tracking = False
        self.detected = False

    def reset_state_vars(self):
        """
        Reset state variables. This function assumes
        no other thread is running so it's not thread-safe
        """
        self.curr_frame = None
        self.location = None
        self.tracking = False
        self.detected = False

    def start(self):
        self.thread = threading.Thread(
            target=self.run_sys, name='TrackingSystem')

        with self.run_lock:
            self.running = True

        self.thread.start()
        print('thread started')

    def stop(self):
        with self.run_lock:
            self.running = False
        self.thread.join()

        self.reset_state_vars()
        print('thread stopped')

    def get_location(self):
        with self.loc_lock:
            loc = self.location
        return loc

    def get_video_frame(self):
        with self.frame_lock:
            if self.curr_frame is not None:
                frame = self.curr_frame.copy()
            else:
                frame = None
        return frame

    def run_sys(self):
        for frame_orig in self.video_source:
            with self.run_lock:
                if not self.running:
                    break

            with self.frame_lock:
                self.curr_frame = frame_orig
            frame = frame_orig.copy()
            frame = utils.run_pipeline(self.pre_detector_pipe, frame)
            self.detected, detect_bbox = self.detector.predict(frame)

            if self.tracking:
                frame = frame_orig.copy()
                frame = utils.run_pipeline(self.pre_tracker_pipe, frame)
                self.tracking, track_bbox = self.tracker.predict(frame)
                if self.detected:
                    # correct tracking if possible
                    iou = utils.bbox_intersection_over_union(
                        detect_bbox, track_bbox)
                    if iou < self.iou_threshold:
                        self.tracker.decrease_health()
                        if self.tracker.get_health() == 0:
                            self.tracking = False
                # else keep tracking
            else:
                # tracker not tracking right now
                if self.detected:
                    # detected, so initialize tracker
                    self.tracker.init_tracker(frame, detect_bbox)
                    self.tracking = True
                    track_bbox = detect_bbox
                # else continue loop

            with self.loc_lock:
                if self.tracking:
                    self.location = (track_bbox[0] + track_bbox[2] / 2,
                                     track_bbox[1] + track_bbox[3] / 2)
                    self.loc_cv.notify_all()
                else:
                    self.location = None

            tracker_stat = self.tracker.get_stat()

            if self.display:
                frame_display = frame_orig.copy()
                frame_display = utils.run_pipeline(
                    self.pre_tracker_pipe, frame_display)
                if self.tracking:
                    p1 = (int(track_bbox[0]), int(track_bbox[1]))
                    p2 = (int(track_bbox[0] + track_bbox[2]),
                          int(track_bbox[1] + track_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1)
                if self.detected:
                    p1 = (int(detect_bbox[0]), int(detect_bbox[1]))
                    p2 = (int(detect_bbox[0] + detect_bbox[2]),
                          int(detect_bbox[1] + detect_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

                cv2.putText(frame_display, 'Tracker FPS : {:.2f}'.format(tracker_stat['fps']), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.putText(frame_display, 'tracker', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(frame_display, 'detector', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                cv2.imshow('app', frame_display)
                with suppress(Exception):
                    cv2.imshow('delta', self.detector.img_delta)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

                print(f'tracking: {self.tracking}; detected: {self.detected}')
                print(
                    f"time taken on detecting: {self.detector.get_stat()['frame_process_time']}")
                print(
                    f"time taken on tracking: {self.tracker.get_stat()['frame_process_time']}")

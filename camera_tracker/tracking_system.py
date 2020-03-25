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
        self.pause_lock = threading.RLock()
        self.paused = False

        self.curr_frame = None
        self.frame_lock = threading.RLock()

        self.location = None
        self.loc_lock = threading.RLock()
        self.loc_cv = threading.Condition(self.loc_lock)

        self.tracking = False
        self.detected = False

        self.track_bbox = None

        # stats
        self.fps = 0

    def reset_state_vars(self):
        """
        Reset state variables. This function assumes
        no other thread is running so it's not thread-safe
        """
        self.curr_frame = None
        self.location = None
        self.tracking = False
        self.detected = False
        self.detector.prev_img = None

    def start(self):
        self.thread = threading.Thread(
            target=self.run_sys, name='TrackingSystem')

        with self.run_lock:
            self.running = True

        self.thread.start()

        print('threads started')

    def stop(self):
        with self.run_lock:
            self.running = False
        self.thread.join()

        self.reset_state_vars()
        print('threads stopped')

    def pause(self):
        with self.pause_lock:
            self.paused = True
        self.reset_state_vars()
        self.detector.prev_img = None

    def resume(self):
        with self.pause_lock:
            self.paused = False

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
        t0 = time.time()
        for frame_orig in self.video_source:
            with self.run_lock:
                if not self.running:
                    break

            with self.frame_lock:
                self.curr_frame = frame_orig

            with self.pause_lock:
                if self.paused:
                    time.sleep(0.01)
                    continue

            frame = frame_orig.copy()
            frame = utils.run_pipeline(self.pre_detector_pipe, frame)
            self.detected, detect_bbox = self.detector.predict(frame)

            if self.tracking:
                frame = frame_orig.copy()
                frame = utils.run_pipeline(self.pre_tracker_pipe, frame)
                self.tracking, self.track_bbox = self.tracker.predict(frame)
                if self.detected and self.tracking:
                    # correct tracking if possible
                    iou = utils.bbox_intersection_over_union(
                        detect_bbox, self.track_bbox)
                    if iou < self.iou_threshold:
                        self.tracker.decrease_health()
                        if self.tracker.get_health() == 0:
                            self.tracking = False
                    
                    # only update location info when both tracking and detected
                    with self.loc_lock:
                        self.location = (self.track_bbox[0] + self.track_bbox[2] / 2,
                                        self.track_bbox[1] + self.track_bbox[3] / 2)
                        self.loc_cv.notify_all()

                else:
                    self.location = None
                # else keep tracking
            else:
                # tracker not tracking right now
                if self.detected:
                    # detected, so initialize tracker
                    self.tracker.init_tracker(frame, detect_bbox)
                    self.tracking = True
                    self.track_bbox = detect_bbox
                # else continue loop

            

            t_frame = time.time() - t0
            self.fps = 1 / t_frame
            tracker_stat = self.tracker.get_stat()

            if self.display:
                frame_display = frame_orig.copy()
                frame_display = utils.run_pipeline(
                    self.pre_tracker_pipe, frame_display)
                if self.tracking:
                    p1 = (int(self.track_bbox[0]), int(self.track_bbox[1]))
                    p2 = (int(self.track_bbox[0] + self.track_bbox[2]),
                          int(self.track_bbox[1] + self.track_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1)
                if self.detected:
                    p1 = (int(detect_bbox[0]), int(detect_bbox[1]))
                    p2 = (int(detect_bbox[0] + detect_bbox[2]),
                          int(detect_bbox[1] + detect_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

                cv2.putText(frame_display, 'FPS : {:.2f}'.format(self.fps), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)
                cv2.putText(frame_display, 'tracker', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_display, 'detector', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow('app', frame_display)
                with suppress(Exception):
                    cv2.imshow('delta', self.detector.img_delta)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

                # print(f'tracking: {self.tracking}; detected: {self.detected}')
                # print(
                #     f"time taken on detecting: {self.detector.get_stat()['frame_process_time']}")
                # print(
                #     f"time taken on tracking: {self.tracker.get_stat()['frame_process_time']}")

            t0 = time.time()

    def set_target(bbox):
        self.pause()
        self.reset_state_vars()

        self.tracker.init_tracker(frame, bbox)
        self.tracking = True
        self.track_bbox = bbox

        self.resume()

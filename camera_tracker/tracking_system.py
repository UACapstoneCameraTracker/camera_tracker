"""
This version lets the detector to run all the time,
so adjustments to the tracker can be made.
"""
import threading
import cv2
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from contextlib import suppress


class TrackingSystem:
    def __init__(self, *args, **kwargs):
        self.tracker = kwargs['tracker']
        self.detector = kwargs['detector']
        self.pre_tracker_pipe = kwargs['pre_tracker_pipe']
        self.pre_detector_pipe = kwargs['pre_detector_pipe']
        self.video_source = kwargs['video_source']
        self.iou_threshold = kwargs['iou_threshold']
        self.display = kwargs['display']
        self.thread = None

        self.curr_frame = None
        self.location = None
        self.frame_lock = threading.Lock()
        self.loc_lock = threading.Lock()

    def start(self):
        self.thread = threading.Thread(
            target=self.run_tracking, name='TrackingSystem')
        self.thread.start()
        print('thread started')

    def get_location(self):
        with self.loc_lock:
            loc = self.location
        return loc

    def get_video_frame(self):
        with self.frame_lock:
            frame = self.curr_frame.copy()
        return frame

    def run_tracking(self):
        tracking = False
        detected = False
        detect_bbox = None
        track_bbox = None

        for frame_orig in self.video_source:
            with self.frame_lock:
                self.curr_frame = frame_orig
            frame = frame_orig.copy()
            frame = utils.run_pipeline(self.pre_detector_pipe, frame)
            detected, detect_bbox = self.detector.predict(frame)

            if tracking:
                frame = frame_orig.copy()
                frame = utils.run_pipeline(self.pre_tracker_pipe, frame)
                tracking, track_bbox = self.tracker.predict(frame)
                if detected:
                    # correct tracking if possible
                    iou = utils.bbox_intersection_over_union(
                        detect_bbox, track_bbox)
                    print(f'iou: {iou}')
                    if iou < self.iou_threshold:
                        self.tracker.decrease_health()
                        if self.tracker.get_health() == 0:
                            tracking = False
                # else keep tracking
            else:
                # tracker not tracking right now
                if detected:
                    # detected, so initialize tracker
                    self.tracker.init_tracker(frame, detect_bbox)
                    tracking = True
                    track_bbox = detect_bbox
                # else continue loop

            with self.loc_lock:
                if tracking:
                    self.location = (track_bbox[0] + track_bbox[2] / 2,
                                    track_bbox[1] + track_bbox[3] / 2)
                else:
                    self.location = None

            tracker_stat = self.tracker.get_stat()
            print('tracking:', tracking, 'detected:',
                  detected, 'fps', tracker_stat['fps'])

            if self.display:
                frame_display = frame_orig.copy()
                frame_display = utils.run_pipeline(
                    self.pre_tracker_pipe, frame_display)
                if tracking:
                    p1 = (int(track_bbox[0]), int(track_bbox[1]))
                    p2 = (int(track_bbox[0] + track_bbox[2]),
                          int(track_bbox[1] + track_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1)
                if detected:
                    p1 = (int(detect_bbox[0]), int(detect_bbox[1]))
                    p2 = (int(detect_bbox[0] + detect_bbox[2]),
                          int(detect_bbox[1] + detect_bbox[3]))
                    cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

                cv2.putText(frame_display, "Tracker FPS : {:.2f}".format(tracker_stat['fps']), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                cv2.imshow('app', frame_display)
                with suppress(Exception):
                    cv2.imshow('delta', self.detector.img_delta)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

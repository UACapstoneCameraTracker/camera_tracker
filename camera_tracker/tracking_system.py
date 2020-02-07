"""
This version lets the detector to run all the time,
so adjustments to the tracker can be made.
"""

import cv2
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from contextlib import suppress


IMG_SIZE = (960, 540)
TRACKER_NAME = 'KCF'
IOU_THRESHOLD = 0.4


class TrackingSystem:
    def __init__(self, *args, **kwargs):
        self.tracker = kwargs['tracker']
        self.detector = kwargs['detector']
        self.pre_tracker_pipe = kwargs['pre_tracker_pipe']
        self.pre_detector_pipe = kwargs['pre_detector_pipe']
        self.video_source = kwargs['video_source']

    def run(self, display=False):
        tracking = False
        detected = False
        detect_bbox = None
        track_bbox = None

        for frame_orig in self.video_source:

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
                    if iou < IOU_THRESHOLD:
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

            print('tracking:', tracking, 'detected:', detected)

            if display:
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

                cv2.imshow('app', frame_display)
                with suppress(Exception):
                    cv2.imshow('delta', self.detector.img_delta)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break


if __name__ == '__main__':
    pre_tracker_pipe = [
        pc.ResizeTransformer(out_size=IMG_SIZE)
    ]

    pre_detector_pipe = [
        pc.ResizeTransformer(out_size=IMG_SIZE),
        pc.GrayscaleTransformer(),
        pc.BlurTransformer()
    ]

    detector = predictors.PixelDifferenceDetector()
    tracker = predictors.CvTracker(TRACKER_NAME)

    tracking_sys = TrackingSystem(
        tracker=tracker,
        detector=detector,
        pre_tracker_pipe=pre_tracker_pipe,
        pre_detector_pipe=pre_detector_pipe,
        video_source=utils.get_frame_generator(),
    )

    tracking_sys.run(display=True)

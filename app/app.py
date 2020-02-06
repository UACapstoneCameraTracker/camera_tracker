import cv2
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from contextlib import suppress


IMG_SIZE = (960, 540)
TRACKER_NAME = 'MOSSE'


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

def run(display=False):
    tracking = False
    detected = False
    cap = utils.get_stream()

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            raise RuntimeError('frame not received')

        if not tracking:
            frame = frame_orig.copy()
            frame = utils.run_pipeline(pre_detector_pipe, frame)
            detected, detect_bbox = detector.predict(frame)

        if tracking or detected:
            # detected, start tracking
            frame = frame_orig.copy()
            frame = utils.run_pipeline(pre_tracker_pipe, frame)
            if not tracking:
                tracker.init_tracker(frame, detect_bbox)
            tracking, track_bbox = tracker.predict(frame)
        
        print('tracking:', tracking, 'detected:', detected)

        if display:
            frame_display = frame.copy()
            if tracking:
                p1 = (int(track_bbox[0]), int(track_bbox[1]))
                p2 = (int(track_bbox[0] + track_bbox[2]), int(track_bbox[1] + track_bbox[3]))
                cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1)
            if detected:
                p1 = (int(detect_bbox[0]), int(detect_bbox[1]))
                p2 = (int(detect_bbox[0] + detect_bbox[2]), int(detect_bbox[1] + detect_bbox[3]))
                cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

            cv2.imshow('app', frame_display)
            with suppress(Exception):
                cv2.imshow('delta', detector.img_delta)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break


if __name__ == '__main__':
    run(display=True)

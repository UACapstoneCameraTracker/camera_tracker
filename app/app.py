import cv2
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils


IMG_SIZE = (960, 540)
TRACKER_NAME = 'MOSSE'


pre_tracker_pipe = [
    pc.ResizeTransformer(out_size=IMG_SIZE)
]

pre_detector_pipe = [
    pc.GrayscaleTransformer(),
    pc.BlurTransformer()
]

detector = predictors.PixelDifferenceDetector()
tracker = predictors.CvTracker(TRACKER_NAME)

def run(display=False):
    tracking = False
    cap = utils.get_stream()

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('frame not received')

        frame = utils.run_pipeline(pre_tracker_pipe, frame)
        frame_color = frame.copy()

        if not tracking:
            frame = utils.run_pipeline(pre_detector_pipe, frame)

            detected, bbox = detector.predict(frame)

            if display:
                if detected:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                

            # movement detected!
            if detected:
                print('movement detected')
                tracker.init_tracker(frame_color, bbox)
                tracking, bbox = tracker.predict(frame_color)
        
        if tracking:
            tracking, bbox = tracker.predict(frame_color)

        if display:
            if tracking:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame_color, p1, p2, (255, 0, 0), 2, 1)

            # display the image
            cv2.imshow('detecter', frame)
            cv2.imshow('tracker', frame_color)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        tracker_stat = tracker.get_stat()
        if tracker_stat['frame_count'] % 10 == 0:
            print(tracker_stat)





if __name__ == '__main__':
    run(display=True)

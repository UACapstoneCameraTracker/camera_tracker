import cv2
from camera_tracker.predictors import PixelDifferenceDetector
from camera_tracker.utils import run_pipeline
from camera_tracker.pipeline_components import (
    ResizeTransformer,
    BlurTransformer,
    GrayscaleTransformer
)

out_size = (960, 540)

def make_pipeline():
    return [
        ResizeTransformer(out_size),
        GrayscaleTransformer(),
        BlurTransformer(),
    ]

cap = cv2.VideoCapture(0)
retry = 5
while retry > 0:
    ret, frame = cap.read()
    if frame is not None:
        break
    print('retrying...')
    retry -= 1

pipe = make_pipeline()
detector = PixelDifferenceDetector()

while True:
    ret, frame = cap.read()

    frame = run_pipeline(pipe, frame)

    ok, bbox = detector.predict(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # display the image
    cv2.imshow("detector", frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
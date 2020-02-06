import camera_tracker.pipeline_components as pc
import camera_tracker.utils as utils

IMG_SIZE = (960, 540)
TRACKER_NAME = 'MOSSE'

preprocessing = [
    pc.ResizeTransformer(IMG_SIZE),
    pc.GrayscaleTransformer(),
]

predictors = {
    'detector': pc.Detector(),
    'tracker': pc.Tracker(TRACKER_NAME)
}

if __name__ == '__main__':
    cap = utils.get_stream()
    ret, frame = cap.read()

    img = utils.run_pipeline(preprocessing)

    

    # low level interaction

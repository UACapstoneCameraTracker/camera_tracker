import cv2
from abc import ABC, abstractmethod
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils

IMG_SIZE = (960, 540)
TRACKER_NAME = 'MOSSE'


pre_tracker_pipe = [
    pc.ResizeTransformer(out_size=IMG_SIZE),
]

pre_detector_pipe = [
    pc.ResizeTransformer(out_size=IMG_SIZE),
    pc.GrayscaleTransformer(),
    pc.BlurTransformer()
]

detector = predictors.PixelDifferenceDetector()
tracker = predictors.CvTracker(TRACKER_NAME)

bounding_box = None


class State(ABC):
    @abstractmethod
    def run(self, frame):
        pass

    @abstractmethod
    def next(self):
        pass


class Environment(ABC):
    @abstractmethod
    def run():
        pass


class DetectState(State):
    def __init__(self):
        self.detected = False

    def __repr__(self):
        return 'DetectState(detected={})'.format(self.detected)

    def run(self, env):
        global bounding_box
        frame = env.copy()
        frame = utils.run_pipeline(pre_detector_pipe, frame)
        self.detected, bounding_box = detector.predict(frame)

    def next(self):
        if self.detected:
            return App.track
        return App.detect


class TrackState(State):
    def __init__(self):
        self.tracking = False
        self.initialized = False

    def __repr__(self):
        return 'TrackState(initialized={}, tracking={})'.format(self.initialized, self.tracking)

    def run(self, env):
        global bounding_box
        frame = env.copy()
        frame = utils.run_pipeline(pre_tracker_pipe, frame)
        if not self.initialized:
            tracker.init_tracker(frame, bounding_box)
            self.initialized = True
            self.tracking = True
        else:
            self.tracking, bounding_box = tracker.predict(frame)

    def next(self):
        if self.tracking:
            return App.track
        return App.detect


class StateMachine:
    def __init__(self, init_state, env):
        self.curr_state = init_state
        self.env = env

    def run(self):
        while True:
            env_data = self.env.run()
            self.curr_state.run(env_data)
            self.curr_state = self.curr_state.next()
            print(self.curr_state)


class AppEnv(Environment):
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap

    def run(self):
        ok, frame = self.cap.read()
        return frame if ok else None


class App(StateMachine):
    def __init__(self, env):
        super().__init__(init_state=App.detect, env=env)


App.detect = DetectState()
App.track = TrackState()

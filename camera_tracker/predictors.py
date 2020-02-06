"""
This module provides predictors (trackers and detectors)
"""

import cv2
from typing import Dict
from .utils import (
    tracker_factory,
    BoundingBox,
    Image
)


class CvTracker(BasePredictionComponent):
    """
    A wrapper to OpenCV tracker.
    """

    def __init__(self, tracker_name: str):
        super().__init__()
        self.tracker_name = tracker_name
        self.tracker = tracker_factory(self.tracker_name)
        self.tracker_inited = False

        self.fps = 0
        self.frame_cnt = 0
        self.fail_cnt = 0

    def init_tracker(self, initial_frame: Image, initial_bbox: BoundingBox) -> None:
        self.tracker.init(initial_frame, initial_bbox)
        self.tracker_inited = True

    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        timer = cv2.getTickCount()
        tracker_status, bbox = self.tracker.update(img)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        self.frame_cnt += 1
        self.fps += (fps - self.fps) / self.frame_cnt

        if not tracker_status:
            self.fail_cnt += 1

        return tracker_status, bbox

    def get_stat(self) -> Dict[str, int]:
        return {
            'fps': self.fps,
            'frame_count': self.frame_cnt,
            'failed_count': self.fail_cnt
        }


class PixelDifferenceDetector(BasePredictionComponent):
    """
    Detect movement by comparing two consecutive frames pixel by pixel.
    """
    def __init__(self, threshold=30, structuring_kernel_shape=(10, 10)):
        super().__init__()

        self.threshold = threshold
        self.structuring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, structuring_kernel_shape)
        self.prev_img = None

    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        if len(img.shape) != 2:
            raise RuntimeError('PixelDifferenceDetector only supports grayscale image')

        if self.prev_img is None:
            self.prev_img = img
            return False, None
        
        img_delta = cv2.absdiff(self.prev_img, img)
        _, img_delta = cv2.threshold(img_delta, self.threshold, 255, cv2.THRESH_BINARY)

        img_delta = cv2.morphologyEx(img_delta, cv2.MORPH_OPEN, kernel)

        _, contours, _ = cv2.findContours(img_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if countours:
            moving_object_boxes = sorted([cv2.boundingRect(cntr) for cntr in countours], key=lambda i: i[2]*i[3])
            return True, moving_object_boxes[-1]
        else:
            return False, None


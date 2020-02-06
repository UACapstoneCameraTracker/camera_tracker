"""
This module provides predictors (trackers and detectors)
"""

import cv2
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

from .utils import (
    tracker_factory,
    BoundingBox,
    Image
)


class BasePredictionComponent(ABC):
    @abstractmethod
    def predict(self, img: Image) -> Any:
        pass


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
        if not self.tracker_inited:
            raise RuntimeError('tracker not initialized!')

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

    def __init__(self, threshold=10, structuring_kernel_shape=(20, 20)):
        super().__init__()

        self.threshold = threshold
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, structuring_kernel_shape)
        self.prev_img = None

    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        if len(img.shape) != 2:
            raise RuntimeError(
                'PixelDifferenceDetector only supports grayscale image')

        if self.prev_img is None:
            self.prev_img = img
            return False, None

        img_delta = cv2.absdiff(self.prev_img, img)
        _, img_delta = cv2.threshold(
            img_delta, self.threshold, 255, cv2.THRESH_BINARY)

        img_delta = cv2.morphologyEx(img_delta, cv2.MORPH_OPEN, self.kernel)
        img_delta = cv2.dilate(img_delta, self.kernel, iterations=2)

        self.img_delta = img_delta

        _, contours, _ = cv2.findContours(
            img_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            moving_object_boxes = sorted(
                [cv2.boundingRect(cntr) for cntr in contours], key=lambda i: i[2]*i[3])
            
            biggest_box = moving_object_boxes[0]
            if biggest_box[2] * biggest_box[3] >= img.shape[0] * img.shape[1] / 2:
                ret = (False, None)
            else:
                ret = (True, biggest_box)
        else:
            ret = (False, None)
        
        self.prev_img = img.copy()
        return ret

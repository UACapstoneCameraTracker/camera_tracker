"""
This module provides predictors (trackers and detectors)
"""

import cv2
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

from .utils import (
    tracker_factory,
    BoundingBox,
    Image,
    bbox_area
)

BBOX_AREA_MIN_TH = 150
MAX_TRACKER_HEALTH = 30


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
        self.tracker_inited = False
        self.tracker = None

        # stats
        self.fps = 0
        self.frame_cnt = 0
        self.fail_cnt = 0
        self.tracker_health = MAX_TRACKER_HEALTH

    def init_tracker(self, initial_frame: Image, initial_bbox: BoundingBox) -> None:
        self.tracker = tracker_factory(self.tracker_name)
        self.tracker.init(initial_frame, initial_bbox)
        self.tracker_inited = True
        self.tracker_health = MAX_TRACKER_HEALTH

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

    def get_health(self):
        return self.tracker_health
    
    def decrease_health(self):
        self.tracker_health -= 1



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

        # filter out background noises
        img_delta = cv2.morphologyEx(img_delta, cv2.MORPH_OPEN, self.kernel)
        # enlarge foreground objects
        img_delta = cv2.morphologyEx(img_delta, cv2.MORPH_CLOSE, self.kernel)

        self.img_delta = img_delta

        contours, _ = cv2.findContours(
            img_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            moving_object_boxes = sorted(
                [cv2.boundingRect(cntr) for cntr in contours], key=lambda i: i[2]*i[3])

            biggest_box = moving_object_boxes[0]
            ret = (self.validate_bbox(biggest_box, img_delta), biggest_box)
        else:
            ret = (False, None)

        self.prev_img = img.copy()
        return ret
    
    def validate_bbox(self, bbox: BoundingBox, frame: Image):
        area = bbox_area(bbox)
        img_area = frame.shape[0] * frame.shape[1]
        return bbox[2] > 1 and bbox[3] > 1 and (BBOX_AREA_MIN_TH < area < (img_area / 2))

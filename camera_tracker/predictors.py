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
    bbox_area,
    run_pipeline
)

from .pipeline_components import (
    ThresholdTransformer,
    OpeningTransformer,
    ClosingTransformer,
)


class BasePredictionComponent(ABC):
    @abstractmethod
    def predict(self, img: Image) -> Any:
        pass


class CvTracker(BasePredictionComponent):
    """
    A wrapper to OpenCV tracker.
    """

    def __init__(self, tracker_name: str, tracker_health: int):
        super().__init__()
        self.tracker_name = tracker_name
        self.tracker_inited = False
        self.tracker = None

        # stats
        self.fps = 0
        self.tot_frame_cnt = 0
        self.this_success_frame_cnt = 0
        self.fail_cnt = 0
        self.max_tracker_health = tracker_health
        self.tracker_health = self.max_tracker_health

    def init_tracker(self, initial_frame: Image, initial_bbox: BoundingBox) -> None:
        self.tracker = tracker_factory(self.tracker_name)
        self.tracker.init(initial_frame, initial_bbox)
        self.tracker_inited = True
        self.tracker_health = self.max_tracker_health
        self.fps = 0
        self.this_success_frame_cnt = 0

    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        if not self.tracker_inited:
            raise RuntimeError('tracker not initialized!')

        timer = cv2.getTickCount()
        tracker_status, bbox = self.tracker.update(img)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        self.tot_frame_cnt += 1
        self.this_success_frame_cnt += 1
        self.fps += (fps - self.fps) / self.this_success_frame_cnt

        if not tracker_status:
            self.fail_cnt += 1

        return tracker_status, bbox

    def get_stat(self) -> Dict[str, int]:
        return {
            'fps': self.fps,
            'frame_count': self.tot_frame_cnt,
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

    def __init__(self, pixel_difference_threshold: int,
                 structuring_kernel_shape: Tuple[int, int],
                 bbox_area_min: float,
                 bbox_area_max: float):
        super().__init__()

        self.threshold = pixel_difference_threshold
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, structuring_kernel_shape)
        self.bbox_area_min = bbox_area_min
        self.bbox_area_max = bbox_area_max
        self.prev_img = None

        self.pipe = [
            ThresholdTransformer(self.threshold),
            OpeningTransformer(self.kernel),
            ClosingTransformer(self.kernel)
        ]

    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        if len(img.shape) != 2:
            raise RuntimeError(
                'PixelDifferenceDetector only supports grayscale image')

        if self.prev_img is None:
            self.prev_img = img
            return False, None

        img_delta = cv2.absdiff(self.prev_img, img)

        img_delta = run_pipeline(self.pipe, img_delta)

        self.img_delta = img_delta

        contours, _ = cv2.findContours(
            img_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = [cv2.boundingRect(cntr) for cntr in contours]
            contours_filtered = list(filter(self.validate_bbox, contours))
            if contours_filtered:
                biggest_box = max(contours_filtered, key=lambda i: i[2]*i[3])
                ret = (self.validate_bbox(biggest_box), biggest_box)
            else:
                ret = (False, None)
        else:
            ret = (False, None)

        self.prev_img = img.copy()
        return ret

    def validate_bbox(self, bbox: BoundingBox) -> bool:
        area = bbox_area(bbox)
        return bbox[2] > 1 and bbox[3] > 1 and (self.bbox_area_min < area < self.bbox_area_max)

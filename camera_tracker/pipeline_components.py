"""
This module provides essential pipelines for image processing and tracking.
"""
from typing import Tuple, Any
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .utils import (
    tracker_factory
)

BoundingBox = Tuple[int, int, int, int]
Image = np.array

class BaseTransformComponent(ABC):
    @abstractmethod
    def transform(self, img: Image) -> Image:
        pass

class BasePredictionComponent(ABC):
    @abstractmethod
    def predict(self, img: Image) -> Any:
        pass


class ResizeTransformer(BaseTransformComponent):
    def __init__(self, out_size: Tuple[int, int]):
        super().__init__()
        self.out_size = out_size

    def transform(self, img: Image) -> Image:
        out = cv2.resize(img, self.out_size)
        return out


class GrayscaleTransformer(BaseTransformComponent):
    def __init__(self):
        super().__init__()
    
    def transform(self, img: Image) -> Image:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out


class Tracker(BasePredictionComponent):
    def __init__(self, tracker_name: str):
        super().__init__()
        self.tracker_name = tracker_name
        self.tracker = tracker_factory(self.tracker_name)
        self.tracker_inited = False

        self.fps = 0
        self.frame_cnt = 0
    
    def init_tracker(self, initial_frame: Image, initial_bbox) -> None:
        self.tracker.init(initial_frame, initial_bbox)
        self.tracker_inited = True
    
    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:
        timer = cv2.getTickCount()
        tracker_status, bbox = self.tracker.update(img)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        tot_fps += fps
        frame_cnt += 1


class Detector(BasePredictionComponent):
    def __init__(self):
        super().__init__()
    
    def predict(self, img: Image) -> Tuple[bool, BoundingBox]:



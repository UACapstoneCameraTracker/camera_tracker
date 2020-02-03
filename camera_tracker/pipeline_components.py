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

class BaseTransformComponent(ABC):
    @abstractmethod
    def transform(self, img: np.array) -> np.array:
        pass


class BasePredictionComponent(ABC):
    @abstractmethod
    def predict(self, img: np.array) -> Any:
        pass
        

class ResizeTransformer(BaseTransformComponent):
    def __init__(self, out_size: Tuple[int, int]):
        super().__init__()
        self.out_size = out_size

    def transform(self, img: np.array) -> np.array:
        out = cv2.resize(img, self.out_size)
        return out


class GrayscaleTransformer(BaseTransformComponent):
    def __init__(self):
        super().__init__()
    
    def transform(self, img: np.array) -> np.array:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out


class Tracker(BasePredictionComponent):
    def __init__(self, tracker_name: str):
        super().__init__()
        self.tracker_name = tracker_name
        self.tracker = tracker_factory(self.tracker_name)
        self.tracker_inited = False
    
    def init_tracker(self, initial_frame, initial_bbox):
        raise NotImplementedError()

    
    def predict(self, img: np.array) -> np.array:
        return super().predict(img)


class Detector(BasePredictionComponent):
    raise NotImplementedError()


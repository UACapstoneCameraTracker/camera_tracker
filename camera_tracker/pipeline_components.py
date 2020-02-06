"""
This module provides essential pipelines for image processing.
"""


import cv2
from abc import ABC, abstractmethod

from .utils import (
    tracker_factory,
    BoundingBox,
    Image
)


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


class BlurTransformer(BaseTransformComponent):
    """
    Apply Gaussian Blur to input image.
    """
    def __init__(self):
        super().__init__()

    def transform(self, img: Image) -> Image:
        out = cv2.GaussianBlur(img, ksize=(21, 21), sigmaX=0)
        return out


class GrayscaleTransformer(BaseTransformComponent):
    def __init__(self):
        super().__init__()

    def transform(self, img: Image) -> Image:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out

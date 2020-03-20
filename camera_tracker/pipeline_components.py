"""
This module provides essential pipelines for image processing.
"""


import cv2
from typing import Tuple
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

    def __init__(self, ksize=(21, 21)):
        super().__init__()
        self.ksize = ksize

    def transform(self, img: Image) -> Image:
        out = cv2.GaussianBlur(img, ksize=self.ksize, sigmaX=0)
        return out


class GrayscaleTransformer(BaseTransformComponent):
    def __init__(self):
        super().__init__()

    def transform(self, img: Image) -> Image:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out


class ThresholdTransformer(BaseTransformComponent):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def transform(self, img: Image):
        _, img = cv2.threshold(
            img, self.threshold, 255, cv2.THRESH_BINARY)
        return img


class OpeningTransformer(BaseTransformComponent):
    def __init__(self, kernel):
        super().__init__()
        self._kernel = kernel

    def transform(self, img: Image) -> Image:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, self._kernel)
        return img


class ClosingTransformer(BaseTransformComponent):
    def __init__(self, kernel):
        super().__init__()
        self._kernel = kernel

    def transform(self, img: Image) -> Image:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self._kernel)
        return img


class DilatingTransformer(BaseTransformComponent):
    def __init__(self, kernel, iteration=1):
        super().__init__()
        self._kernel = kernel
        self._iteration = iteration

    def transform(self, img: Image) -> Image:
        img = cv2.dilate(img, self._kernel, iterations=self._iteration)
        return img

import unittest
import cv2
from camera_tracker.utils import run_pipeline
from camera_tracker.pipeline_components import (
    ResizeTransformer,
    BlurTransformer,
    GrayscaleTransformer
)

img = cv2.imread('pipeline_test_img.jpg')
out_size = (960, 540)

def make_pipeline():
    return [
        ResizeTransformer(out_size),
        GrayscaleTransformer(),
        BlurTransformer(),
    ]

class PipelineTest(unittest.TestCase):
    def test_output_shape(self):
        pipe = make_pipeline()

        out = run_pipeline(pipe, img)

        self.assertEqual(out_size[::-1], out.shape)


if __name__ == '__main__':
    unittest.main()
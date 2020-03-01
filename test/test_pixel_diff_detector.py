import unittest
import cv2
from camera_tracker.predictors import PixelDifferenceDetector

img1 = cv2.imread('tracking_img1.png')
img2 = cv2.imread('tracking_img2.png')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

class DetectorTest(unittest.TestCase):
    def test_not_gray_scale(self):
        with self.assertRaises(Exception) as context:
            detector = PixelDifferenceDetector(pixel_difference_threshold=10,
                                                  structuring_kernel_shape=(20, 20),
                                                  bbox_area_min=150)
            detector.predict(img1)
    
    def test_detect(self):
        detector = PixelDifferenceDetector(pixel_difference_threshold=10,
                                                  structuring_kernel_shape=(20, 20),
                                                  bbox_area_min=150)
        ret = detector.predict(gray1)
        self.assertFalse(ret[0])

        ret = detector.predict(gray2)
        self.assertTrue(ret[0])

        print(ret[1])


if __name__ == '__main__':
    unittest.main()
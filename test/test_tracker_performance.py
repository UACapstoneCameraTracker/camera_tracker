import unittest
import time
import camera_tracker.pipeline_components as pc
import camera_tracker.predictors as predictors
import camera_tracker.utils as utils
from camera_tracker.tracking_system import TrackingSystem
from app import settings

def setup_tracking_system():
    pre_tracker_pipe = [
        pc.ResizeTransformer(out_size=settings.IMG_SIZE)
    ]

    pre_detector_pipe = [
        pc.ResizeTransformer(out_size=settings.IMG_SIZE),
        pc.GrayscaleTransformer(),
        pc.BlurTransformer()
    ]

    detector = predictors.PixelDifferenceDetector(pixel_difference_threshold=settings.PIXEL_DIFFERENCE_TH,
                                                  structuring_kernel_shape=settings.STRUCTURING_KERNEL_SHAPE,
                                                  bbox_area_min=settings.BBOX_AREA_MIN_TH,
                                                  bbox_area_max=settings.BBOX_AREA_MAX_TH)
    tracker = predictors.CvTracker(tracker_name=settings.TRACKER_NAME,
                                   tracker_health=settings.MAX_TRACKER_HEALTH)

    tracking_sys = TrackingSystem(tracker=tracker,
                                  detector=detector,
                                  pre_tracker_pipe=pre_tracker_pipe,
                                  pre_detector_pipe=pre_detector_pipe,
                                  video_source=utils.get_frame_generator(mock=True),
                                  iou_threshold=settings.IOU_THRESHOLD,
                                  display=True)
    return tracking_sys


class TrackingSystemPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.sys = setup_tracking_system()
        self.sys.start()
    
    def tearDown(self):
        self.sys.stop()

    def test_tracking_system_fps(self):
        time.sleep(10)
        stats = self.sys.tracker.get_stat()
        fail_ratio =  stats['failed_count'] / stats['frame_count']
        print('frame count', stats['frame_count'])
        print('failed count', stats['failed_count'])
        print('fail_ratio', fail_ratio)
        print('fps', stats['fps'])

if __name__ == '__main__':
    unittest.main()
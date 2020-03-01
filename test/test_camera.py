import unittest
import time
from statistics import mean
import numpy as np
from camera_tracker.utils import get_frame_generator

source = get_frame_generator()

class CameraTest(unittest.TestCase):
    
    def test_read_frame(self):
        frame = next(source)
        self.assertNotEqual(len(frame.shape), 0)
        
    def test_frame_channel(self):
        frame = next(source)
        self.assertEqual(frame.shape[2], 3)
    
    def test_frame_size(self):
        frame = next(source)
        self.assertEqual(frame.shape[0:2], (480, 640))
    
    def test_fps(self):
        gen = get_frame_generator()
        # skip the first frame to remove setup time
        _ = next(gen)
        T = []
        last_t = time.time()
        for _ in range(120):
            _ = next(gen)
            t = time.time()
            T.append(t - last_t)
            last_t = time.time()
        
        fps = 1 / mean(T)
        print('fps: ', fps)
        self.assertGreaterEqual(fps, 25)

if __name__ == '__main__':
    unittest.main()
import cv2
import numpy as np
from typing import Tuple, Any
from pathlib import Path

# custom types
BoundingBox = Tuple[int, int, int, int]
Image = np.array


def tracker_factory(tracker_name):
    tracker_table = {
        'KCF': cv2.TrackerKCF_create,
        'MIL': cv2.TrackerMIL_create,
        'BOOSTING': cv2.TrackerBoosting_create,
        'GOTURN': cv2.TrackerGOTURN_create,
        'MOSSE': cv2.TrackerMOSSE_create,
        'CSRT': cv2.TrackerCSRT_create,
        'MEDIANFLOW': cv2.TrackerMedianFlow_create
    }
    return tracker_table[tracker_name]()


def get_stream(mock=False):
    if mock:
        video_path = str(Path(__file__).parents[2] / 'videos/helicopter1.MOV')
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    return cap

def get_frame_generator(mock=False):
    """
    this function returns a generator that yields the current frame
    """
    cap = get_stream(mock)
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        yield frame


def run_pipeline(pipe, img):
    for p in pipe:
        img = p.transform(img)
    return img


def run_predictor(pipe, img):
    raise NotImplementedError()


def bbox_area(bbox: BoundingBox):
    return bbox[2] * bbox[3]


def bbox_intersection_over_union(bbox_a, bbox_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox_a[0], bbox_b[0])
    yA = max(bbox_a[1], bbox_b[1])
    xB = min(bbox_a[0]+bbox_a[2], bbox_b[0]+bbox_b[2])
    yB = min(bbox_a[1]+bbox_a[3], bbox_b[1]+bbox_b[3])
    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    a_area = bbox_a[2] * bbox_a[3]
    b_area = bbox_b[2] * bbox_b[3]
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(a_area + b_area - inter_area)
    # return the intersection over union value
    if iou < 0:
        iou = 0
    return iou

import numpy as np
import cv2

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
        cap = cv2.VideoCapture('../videos/helicopter1.MOV')
    else:
        cap = cv2.VideoCapture(0)
    
    return cap


def run_pipeline(pipe, img):
    for p in pipe:
        img = p.transform(img)
    return img


def run_predictor(pipe, img):
    raise NotImplementedError()
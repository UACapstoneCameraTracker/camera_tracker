from __future__ import print_function

import cv2
import sys

FRAME_NAME = 'tracker'
WIDTH = 960
HEIGHT = 540


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


def retrieve_frame(cap, width, height):
    ret, frame = cap.read()
    if not ret:
        return None

    resized_img = cv2.resize(frame, (width, height))
    return resized_img


if len(sys.argv) != 3:
    print('tracker.py <tracker name> <video path>')
    exit(1)



def run_tracking(tracker, cap, initial_bbox, display=False, add_text=True):
    if not cap.isOpened():
        raise AttributeError('Cannot open stream {}'.format(video_path))
        

    frame = retrieve_frame(cap, WIDTH, HEIGHT)
    if frame is None:
        raise RuntimeError('cannot read video file')

    # initialize my tracker
    tracker.init(frame, initial_bbox)

    tot_fps = 0
    frame_cnt = 0
    lost_target_frame_cnt = 0
    while True:
        frame = retrieve_frame(cap, WIDTH, HEIGHT)
        if frame is None:
            break

        # calculate the fps
        timer = cv2.getTickCount()
        tracker_status, bbox = mytracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        tot_fps += fps
        frame_cnt += 1

        if not tracker_status:
            lost_target_frame_cnt += 1

        if display:
            if tracker_status:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            elif add_text:
                cv2.putText(frame, "Tracking failure detected", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if add_text:
                cv2.putText(frame, tracker_name + " Tracker", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.putText(frame, "FPS : " + str(int(fps)), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.putText(frame, "bbox : " + str(bbox), (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            # display the image
            cv2.imshow(FRAME_NAME, frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    return {
        'average fps': tot_fps / frame_cnt,
        'lost count ratio': lost_target_frame_cnt / frame_cnt
    }


if __name__ == '__main__':
    tracker_name = sys.argv[1].upper()
    video_path = sys.argv[2]

    # create tracker
    mytracker = tracker_factory(tracker_name)

    # create video stream
    try:
        video_path = int(video_path)
    except:
        pass
    cap = cv2.VideoCapture(video_path)

    retry = 5
    while retry > 0:
        frame = retrieve_frame(cap, WIDTH, HEIGHT)
        if frame is not None:
            break
        retry -= 1

    bbox = cv2.selectROI(frame, False)

    ret = run_tracking(mytracker, cap, bbox, display=True)
    print('average fps {:.2f}'.format(ret['average fps']))
    print('lost frame: {:.2%}'.format(ret['lost count ratio']))

    cap.release()
    cv2.destroyAllWindows()
import detect
import cv2
import time
import os
from ipdb import set_trace

def camera_detect():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'

    start = time.time()
    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            loc = detect.get_all_predict(frame)
            if type(loc) != int:
                img = detect.print_rectangle(frame, loc)
            cv2.imshow("myCamera_demo", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        else:
            break

    total_time = time.time() - start
    FPS = round(cnt/total_time, 2)
    print("the average FPS is {}".format(FPS))

def video_detect_save(videoPath):
    cap = cv2.VideoCapture(videoPath)
    assert os.path.exists(videoPath), "the denoted video file does not exist,please check it again."
    assert cap.isOpened(), "faile to open the denoted vido."

    while cap.isOpened():
        pass



if __name__ == '__main__':
    #batch_size, confidence, nms_thresh, num_classes, classes, CUDA, model, inp_dim = detect.init()
    camera_detect()

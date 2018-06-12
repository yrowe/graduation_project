import detect
import cv2
import time
import os
from ipdb import set_trace

def camera_detect(save_path = 'camera_deomo.avi'):
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print((frame_width,frame_height))

    out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M','J','P','G'),15, (frame_width,frame_height))

    start = time.time()
    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            loc = detect.get_all_predict(frame)
            if type(loc) != int:
                img = detect.print_rectangle(frame, loc)

            out.write(img)
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

def video_detect_save(videoPath, save_path = 'outp.avi'):
    cap = cv2.VideoCapture(videoPath)
    assert os.path.exists(videoPath), "the denoted video file does not exist,please check it again."
    assert cap.isOpened(), "faile to open the denoted vido."

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            loc = detect.get_all_predict(frame)
            if type(loc) != int:
                img = detect.print_rectangle(frame, loc)
            cv2.imshow("processed img", img)
            out.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
'''
def video_real_time_demo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    assert os.path.exists(videoPath), "the denoted video file does not exist, please check it again"
    assert cap.isOpened(), "failed to open the denoted video."

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)
    #our machine ability's upper bound. My GPU is GTX 1066. you can refer this value to adjust your own machine.
    upper_bound = 15
    
    if fps < 15:
        #simply read image frame by frame, then process them.
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                loc = detect.get_all_predict(frame)
                if type(loc) != int:
                    img = detect.print_rectangle(frame, loc)
                cv2.imshow("real time video detection", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

    else:
        #example we got a video of 60FPS, but we can process upon to 15FPS
        #so we got a 4 times larger task.
        #in order to solve this. we choose process 1 of 4 of this video.
        #so we need a count calculator now. 
        interval = int(fps/upper_bound)
        cnt = 0
        #what will happed if video fps is between [15, 30]
        while cap.isOpened():
            ret, frame = cap.read()
            if cnt % interval != 0:
                cnt += 1
                continue

            cnt = 0
            if ret:
                loc = detect.get_all_predict(frame)
                if type(loc) != int:
                    


    #I supposed that yolov3 has a stable FPS of 15.




'''
if __name__ == '__main__':
    #batch_size, confidence, nms_thresh, num_classes, classes, CUDA, model, inp_dim = detect.init()
    camera_detect()

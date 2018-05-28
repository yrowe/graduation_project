import body_net
import cv2
import torch
import time

def camera_detect():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'

    extractor, classifier = body_net.vgg16_decompose()
    rpn = body_net.RegionProposalNetwork()
    head = body_net.VGG16RoIHead(classifier)
        
    submod = body_net.FasterRCNNVGG16(extractor, rpn, head)
    net = body_net.FasterRCNNTrainer(submod).cuda()

    print("loading model...")
    net.load_state_dict(torch.load('fasterRCNN.pth'))
    print("successfully load faster rcnn.")

    start = time.time()
    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            locs = net.get_all_locs(frame)
            img = body_net.print_rectangle(frame, locs)
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


if __name__ == '__main__':
    camera_detect()
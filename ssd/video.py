import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import VOC_CLASSES as labels
from ssd import build_ssd
import time
# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 model

net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')

cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot open camera'
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print((frame_width,frame_height))

save_path = 'ssd_demo.avi'
out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M','J','P','G'), 23, (frame_width,frame_height))

start = time.time()
cnt = 0
while cap.isOpened():
    cnt += 1
    ret, frame = cap.read()
    if ret:
        x = cv2.resize(frame, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        xx = xx.cuda()
        y = net(xx)
        color = (0, 0, 255)
        detections = y.data
        scale = torch.Tensor(frame.shape[1::-1]).repeat(2)
        i = 15
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            #print(pt[0], pt[1], pt[2], pt[3])
            #coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            #print(*coords)
            cv2.rectangle(frame,(pt[0],pt[1]),(pt[2],pt[3]),color,2)
            j+=1

        out.write(frame)
        #cv2.imshow("myCamera_demo", frame)
        '''
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        continue
        '''
        if cnt == 200:
            break

    else:
        break

total_time = time.time() - start
FPS = round(cnt/total_time, 2)
print("the average FPS is {}".format(FPS))
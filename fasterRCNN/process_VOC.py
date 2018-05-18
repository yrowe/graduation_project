import os
import torch as t
from tools.config import opt
from lib.VGG16.faster_rcnn_vgg16 import FasterRCNNVGG16
from lib.cascade.trainer import FasterRCNNTrainer
from tools.util import  read_image
from tools.vis_tool import vis_bbox, predict
from tools import array_tool as at
from ipdb import set_trace
import cv2


faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('/home/wrc/fasterrcnn_0.712.pth')

f = open("testall.txt")

cnt = 0
imgPath = f.readline()
while imgPath:
    cnt += 1
    print(cnt)
    labelPath = imgPath.replace("\n","")
    imgPath = labelPath
    labelPath = labelPath.replace("JPEGImages","labels")
    labelPath = labelPath.replace(".jpg",".txt")
    savePath = labelPath.replace("labels", "rcnnpredict")
    if os.path.getsize(labelPath) == 0:
        imgPath = f.readline()
        continue

    img = read_image(imgPath)
    img = t.from_numpy(img)[None]

    #set_trace()
    _bboxes, _labels, _scores = predict(img, model=trainer.faster_rcnn, specific_label=['person'])

    loc = []

    length = len(_bboxes[0])

    for i in range(length):
        loc.append([_bboxes[0][i][1], _bboxes[0][i][3], _bboxes[0][i][0], _bboxes[0][i][2]])

    outfile = open(savePath, "w")
    for i in range(len(loc)):
        outlines = '0'
        
        for j in range(4):
            outlines = outlines + ' ' + str(loc[i][j])
        outlines = outlines + '\n'
        #set_trace()
        outfile.write(outlines)
    outfile.close()

    imgPath = f.readline()
    
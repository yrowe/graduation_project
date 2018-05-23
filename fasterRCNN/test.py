import os
import torch as t
from tools.config import opt
from lib.VGG16.faster_rcnn_vgg16 import FasterRCNNVGG16
from lib.cascade.trainer import FasterRCNNTrainer
from tools.util import  read_image
from tools.vis_tool import vis_bbox, predict
from tools import array_tool as at

img = read_image('tst.jpg')
set_trace()
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('/home/wrc/fasterrcnn_0.712.pth')

_bboxes, _labels, _scores = predict(img, model=trainer.faster_rcnn, specific_label=['person'])
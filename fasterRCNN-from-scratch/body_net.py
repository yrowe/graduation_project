import torch
import torch.nn as nn
import torchvision.models
from ipdb import set_trace
import numpy as np
import cv2


class FasterRCNNTrainer(nn.Module):
    def __init__(self, fasterRCNN):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = fasterRCNN

    def forward(self, x):
        #extract feature network, reuse of vgg16.
        x = self.faster_rcnn.extractor(x)
        #now we got [1, 3, 600, 800]
        

        return x


class FasterRCNNVGG16(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNNVGG16, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super(RegionProposalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.score = nn.Conv2d(512, 18, kernel_size=(1,1), stride=(1,1))
        self.loc = nn.Conv2d(512, 36, kernel_size=(1,1), stride=(1,1))

class VGG16RoIHead(nn.Module):
    def __init__(self, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(in_features=4096, out_features=84, bias=True)
        self.score = nn.Linear(in_features=4096, out_features=21, bias= True)


def vgg16_decompose():
    #we don't neet parameters
    model = torchvision.models.vgg16(pretrained=False)  
    # no need for the last maxpooling layer
    extractor = list(model.features)[:-1]
    #no need for the last fulling conncected layer
    classifier = list(model.classifier)[:-1]
    #since we will implement the customed dropout method,
    #we simply discard the original dropout of vgg16
    del classifier[5]
    del classifier[2]
    
    return nn.Sequential(*extractor), nn.Sequential(*classifier)

def preprocess(img, min_size=600, max_size=1000):
    #img is get by func cv2.imread. So its shape is [H, W, C] and its channel format is BGR.
    
    H, W, C = img.shape
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    
    #note that cv2.resize inputs should be (W, H)!
    #set_trace()
    img = cv2.resize(img, (int(scale*W), int(scale*H)))
    
    #since we reuse VGG16 which is trained in ImageNet dataset. We need to normalize it in order to
    #get the total average value equals 0.
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(1, 1, 3)
    img = (img - mean).astype(np.float32, copy=True)
    #return shape should be [C, H ,W]
    img = np.transpose(img, (2,0,1))
    return img

extractor, classifier = vgg16_decompose()
rpn = RegionProposalNetwork()
head = VGG16RoIHead(classifier)
    
tst = FasterRCNNVGG16(extractor, rpn, head)
net = FasterRCNNTrainer(tst).cuda()

net.load_state_dict(torch.load('fasterRCNN.pth'))

img = cv2.imread("1.jpg")
img = preprocess(img)

input_x = torch.from_numpy(img)
input_x.unsqueeze_(0)
#set_trace()
input_x = input_x.cuda()

outp = net(input_x)
"""
outp.shape
torch.Size([1, 512, 14, 14])
"""

#already finish load pretrained model
#!TODO forward process
set_trace()
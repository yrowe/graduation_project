import torch
import torch.nn as nn
import torchvision.models
from ipdb import set_trace
import numpy as np
import cv2
from torch.nn import functional as F


class FasterRCNNTrainer(nn.Module):
    def __init__(self, fasterRCNN):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = fasterRCNN
        self.anchor_base = np.array([[ -37.254833,  -82.50967 ,   53.254833,   98.50967 ],
                                     [ -82.50967 , -173.01933 ,   98.50967 ,  189.01933 ],
                                     [-173.01933 , -354.03867 ,  189.01933 ,  370.03867 ],
                                     [ -56.      ,  -56.      ,   72.      ,   72.      ],
                                     [-120.      , -120.      ,  136.      ,  136.      ],
                                     [-248.      , -248.      ,  264.      ,  264.      ],
                                     [ -82.50967 ,  -37.254833,   98.50967 ,   53.254833],
                                     [-173.01933 ,  -82.50967 ,  189.01933 ,   98.50967 ],
                                     [-354.03867 , -173.01933 ,  370.03867 ,  189.01933 ]],
                                     dtype=np.float32)
        self.feat_stride = 16

    def forward(self, x):
        #extract feature network, reuse of vgg16.
        x = self.faster_rcnn.extractor(x)
        #now we got feature map
        h = x.shape[2]
        w = x.shape[3]

        anchor = generate_anchors(self.anchor_base, 
                   self.feat_stride, h, w)

        n_anchor = self.anchor_base.shape[0]
        
        #one more 3*3 conv to extractor features.
        layer1 = F.relu(self.rpn.conv1(x))

        #now we need to forward into 2 paths.
        #location path:
        rpn_locs = self.rpn.loc(layer1)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        
        #score path:
        rpn_scores = self.rpn.score(layer1)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_fg_scores = rpn_scores.view(1, h, w, n_anchor, 2)[:, :, :, :, 1].contiguous().view(1, -1)
        rois = self.proposal_layer(
                rpn_locs[0].cpu().data.numpy(),
                rpn_fg_scores[0].cpu().data.numpy(),
                anchor, img_size, scale=scale)

        rois_indices = np.zeros(len(rois, ), dtype=np.int32)
        return x

    def proposal_layer(self, loc, score, anchor, img_size, scale):
        nms_thresh = 0.7
        pre_nms = 6000
        post_nms = 300
        min_size = 16

        roi = loc2bbox(anchor, loc)

        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[0])
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[1])

        min_size = min_size*scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        keep = np.where((hs >= min_size)&(ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.argsort()[::-1]
        order = order[:pre_nms]
        roi = roi[order, :]


        keep = keep[:post_nms]
        roi = roi[keep]

        return roi




def generate_anchors(anchor_base, feat_stride, height, width):
    xx = np.arange(0, width*feat_stride, feat_stride)
    yy = np.arange(0, height*feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(xx, yy)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis = 1)

    A = anchor_base.shape[0]
    K = shift.shape[0]

    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1,0,2))
    anchor = anchor.reshape((K*A, 4)).astype(np.float32)

    return anchor


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

with torch.no_grad():
    outp = net(input_x)
"""
outp.shape
torch.Size([1, 512, 14, 14])
"""

#already finish load pretrained model
#!TODO forward process
set_trace()
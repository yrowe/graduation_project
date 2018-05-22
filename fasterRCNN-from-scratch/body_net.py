import torch
import torch.nn as nn
import torchvision.models
from ipdb import set_trace


class FasterRCNNTrainer(nn.Module):
    def __init__(self, fasterRCNN):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = fasterRCNN


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

extractor, classifier = vgg16_decompose()
rpn = RegionProposalNetwork()
head = VGG16RoIHead(classifier)
    
tst = FasterRCNNVGG16(extractor, rpn, head)
net = FasterRCNNTrainer(tst)

net.load_state_dict(torch.load('fasterRCNN.pth'))
#already finish load pretrained model
#!TODO forward process
set_trace()
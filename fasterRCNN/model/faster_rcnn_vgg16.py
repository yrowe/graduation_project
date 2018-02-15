import torch
import torch.nn as nn
from torchvision.models import vgg16

from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
from model.roi import VGG16RoIHead


class FasterRCNNVGG16(FasterRCNN):
	'''this a specific FasterRCNN model based on VGG16 recongnization network.
	and it inherited class FasterRCNN.   see more details about class FasterRCNN in './model/faster_rcnn.py'

	Args:
		n_fg_class(int):  The number of the foreground class,that is to say number of classes excluded background. 
			So the total class number should be (n_fg_class + 1).
		ratios(list of floats): They are different ratios of heights and widths of anchors.
		anchor_scales(list of numbers): They are different sizes of anchors.
		
		ratios and anchor_scales create the different anchors (3 for each, and 9 for total, descripted by paper)
	'''

	feat_stride = 16    # downsample 16x for output of conv5 in vgg16
	# ?variable in this,is a global variable?
	
	def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
		extractor, classifier = vgg16_decompose()

		rpn = RegionProposalNetwork()   #all in default. and the diction variable refers to what?

		head = VGG16RoIHead()           #@TODO  the interface

		super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)




def vgg16_decompose():
	'''
	this function decompose the whole vgg16 net into two part: 'features' and 'classifier'.
		'features' part is the first 30 layer of original vgg16,which includes 13 conv layers,13 relu layers, 4 pooling layers.
		'classifier' part is the following 3 fully connected layer of original vgg16 net.
	
	the detailed VGG structure are as follows:
	VGG(
	  (features): Sequential(
	    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU(inplace)
	    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU(inplace)
	    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU(inplace)
	    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU(inplace)
	    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU(inplace)
	    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU(inplace)
	    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU(inplace)
	    (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU(inplace)
	    (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU(inplace)
	    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU(inplace)
	    (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU(inplace)
	    (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU(inplace)
	    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU(inplace)
	    (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	  )
	  (classifier): Sequential(
	    (0): Linear(in_features=25088, out_features=4096)
	    (1): ReLU(inplace)
	    (2): Dropout(p=0.5)
	    (3): Linear(in_features=4096, out_features=4096)
	    (4): ReLU(inplace)
	    (5): Dropout(p=0.5)
	    (6): Linear(in_features=4096, out_features=1000)
	  )
	)
	'''
	model = vgg16(pretrained=False)
	model.load_state_dict(torch.load('checkpoints/vgg16-caffe.pth'))   #firstly fix to use caffe pretrained model. latter will add more optional choices.

	#processing features part.
	features = list(model.features)[:-1]  #discard the last maxpooling layer of vgg16 features part.
	#in the purpose of saving GPU memory.fixed the first 10 layers(that is to say the first 4 convolutional layers).
	#these layers would not back propogate grads.
	for layer in features[:10]:
		for p in layer.parameters():
			p.requires_grad = False      


	classifier = list(model.classifier)[:-1] #discard the last linear layer,which are used to classify task.

	#since we expect to use customed dropout layer in RoI head. we fisrt delete the original implement of dropout.
	del classifier[2]
	del classifier[5]

	return nn.Sequential(*features), nn.Sequential(*classifier)
	
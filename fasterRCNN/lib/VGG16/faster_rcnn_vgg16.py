import torch
import torch.nn as nn
from torchvision.models import vgg16

from lib.VGG16.faster_rcnn import FasterRCNN
from lib.RPN.region_proposal_network import RegionProposalNetwork
from lib.RoI.roi_module import RoIPooling2D
from tools import array_tool as at 
from tools.config import opt 


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

		rpn = RegionProposalNetwork(
					512, 512,
					ratios=ratios,
					anchor_scales=anchor_scales,
					feat_stride=self.feat_stride)   #all in default. and the diction variable refers to what?

		head = VGG16RoIHead(
					n_class=n_fg_class+1,
					roi_size=7,
					spatial_scale=(1./self.feat_stride),
					classifier=classifier)           #@TODO  the interface

		super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)


class VGG16RoIHead(nn.Module):
	''' RoI pooling head part of FasterRCNN with VGG16 version
	This outputs class-wise locations and classification based on feature
	maps in the given RoIs
	
	Args:
		n_class(int): the number of foreground and background class..
		roi_size(int):  height and width of the feature maps after RoI-pooling.
		spatial_scale(float): scale of the roi is resized.
		classifier(nn.Module): four layer totally, two linear and two relu. 
	'''

	def __init__(self, n_class, roi_size, spatial_scale, classifier):
		super(VGG16RoIHead, self).__init__()
		self.classifier = classifier
		self.cls_loc = nn.Linear(4096, n_class*4)
		self.score = nn.Linear(4096, n_class)

		normal_init(self.cls_loc, 0, 0.001)
		normal_init(self.score, 0, 0.01)

		self.n_class = n_class
		self.roi_size = roi_size
		self.spatial_scale = spatial_scale
		self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

	def forward(self, x, rois, roi_indices):
		'''
		To be simplified, since we fixed the batchsize to 1,
		There is no need the param roi_indices

		Args:
			x(Variable): feature map, 4D variable(B,C,H,W).
			rois(ndarray): a bounding box array containing coordinates of
				proposal boxes after :meth`ProposalTargetCreator`.
				A typical number is (128, 4), where 128 represents the 
				proposal boxes of a single image, 4 represents the coordinates
				of each boxes respectively.
			roi_indices(torch.Tensor): since this algorithm should support
				batch_size greater than 1.So rois actually should be a concatenate
				of various images, and roi_indices should index which original image
				they belong to. this implement fixes batch_size to 1.This params
				would be delete latter.   @!TODO
		'''
		#in case roi_indices and rois are ndarray.
		roi_indices = at.totensor(roi_indices).float()
		rois = at.totensor(rois).float()

		indices_and_rois = torch.cat([roi_indices[:, None],rois],dim=1)
		#(B, ymin, xmin, ymax, xmax) -> (B, xmin, ymin, xmax, ymax)
		xy_indices_and_rois = indices_and_rois[:, [0,2,1,4,3]]
		indices_and_rois = torch.autograd.Variable(xy_indices_and_rois.contiguous())

		pool = self.roi(x, indices_and_rois)   #128*512*7*7    after RoI pooling layer
		pool = pool.view(pool.size(0), -1)     #128 * 25088  inorder to share weight.
		
		fc7 = self.classifier(pool)
		roi_cls_locs = self.cls_loc(fc7)
		roi_scores = self.score(fc7)

		return roi_cls_locs, roi_scores

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
	model.load_state_dict(torch.load(opt.caffe_pretrain_path))   #firstly fix to use caffe pretrained model. latter will add more optional choices.

	#processing features part.
	features = list(model.features)[:-1]  #discard the last maxpooling layer of vgg16 features part.
	#in the purpose of saving GPU memory.fixed the first 10 layers(that is to say the first 4 convolutional layers).
	#these layers would not back propogate grads.
	for layer in features[:10]:
		for p in layer.parameters():
			p.requires_grad = False      


	classifier = list(model.classifier)[:-1] #discard the last linear layer,which are used to classify task.

	#since we expect to use customed dropout layer in RoI head. we fisrt delete the original implement of dropout.
	del classifier[5]
	del classifier[2]

	return nn.Sequential(*features), nn.Sequential(*classifier)
	

def normal_init(layer, mean, stddev):
	'''
	weight initialzer: truncated normal or random normal.

	default is random normal.

	Args:
		layer(torch.nn.modules.conv.Conv2d):   three different conv layers in rpn.
		mean(float): the mean value of Gaussian distribution
		stddev(float): the standard deviation value of Gaussian distribution.
	
	this operation is inplace, so there is no returns.

	'''

	layer.weight.data.normal_(mean, stddev)
	layer.bias.data.zero_()
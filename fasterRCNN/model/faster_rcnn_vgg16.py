import torch
import torch.nn as nn
import torchvision.models.vgg16 as vgg16

from model.faster_rcnn import FasterRCNN


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
	
	def __init__(self, n_fg_class, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
		pass



def vgg16_decompose():
	'''
	this function decompose the whole vgg16 net into two part: 'features' and 'classifier'.
		'features' part is the first 30 layer of original vgg16,which includes 13 conv layers,13 relu layers, 4 pooling layers.
		'classifier' part is the following 3 fully connected layer of original vgg16 net.
	'''
	model = vgg16(pretrained=False)
	model.load_state_dict(torch.load('checkpoints/vgg16-caffe.pth'))   #firstly fix to use caffe pretrained model. latter will add more optional choices.


	
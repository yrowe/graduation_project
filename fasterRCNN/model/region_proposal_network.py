import torch.nn as nn
import numpy as np

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
	'''
	RegionProposalNetwork(aka rpn) is constructed by adding two additional conv layer: 
	one that encodes each conv map position into a short feature vector.
	a second that,at each position, outputs an objectness score 
	and regressed bounds for k region proposals relatives to various scales and aspect ratios at that location.
	
	Args:
		in_channels(int): The input channel size of the 3*3 conv layer of rpn.
		
		mid_channels(int): The output channel size of the 3*3 conv layer of rpn, 
						also used for the input of latter 1*1 conv layer.	
		
		ratios(list of float): They are various aspect ratios of width to height of the anchors.
		
		anchor_scales(list of numbers): They are various areas of anchors.
		
		feat_stride(int): Stride size after extracting features from an image.   (downsample size?)
		
		initialW(callable): Initial weight value. In default, this function uses Gaussian distribution scaled by
						0.1 to initialize weight.
		
		proposal_creator_params(dict): Key valued parameters for proposalCreator.
	'''

	def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], 
				 anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict()):
		super(RegionProposalNetwork, self).__init__()
		self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)   #@TODO to generate anchor
		self.feat_stride = feat_stride
		self.proposal_layer = ProposalCreator(self, **proposal_creator_params)   # @TODO to create some proposals.
		n_anchor = self.anchor_base.shape[0]
		self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)   #the first 3*3 conv.
		self.score = nn.Conv2d(mid_channels, n_anchor*2, kernel_size=(1, 1), stride=(1, 1), padding=0)    # one of the following two path.
		self.loc = nn.Conv2d(mid_channels, n_anchor*4, kernel_size=(1, 1), stride=(1, 1), padding=0)   # the second path.

		normal_init(self.conv1, 0, 0.01)
		normal_init(self.score, 0, 0.01)
		normal_init(self.loc, 0, 0.01)         #or just torch.nn.init.normal        see also in pytorch document.

	def forward(self, base_feat, img_size, scale=1.):
		'''
		forward propagate region proposal network.

		There are some notations as follows:
		  	'N', number, represents the batch size.
		  	'C', channel, represents channel size of input features.
		  	'H', height, represents height of input features. 
		  	'W', width, represents width of input features.
		  	'A', all, represents the total number of anchors assigned to each pixel.

		Args:
			base_feat(torch.autograd.Variable): Features extracted from images by vgg16-extractor defined in the file 'faster_rcnn_vgg16'.
											Shape: (N, C, H, W)
			img_size(float): A tuple consist of height and width, which defines the image size after scaling.
			
			scale(float): The amount of scaling done to the input images after reading them from files. ???


		Returns:
			(torch.autograd.Variable, torch.autograd.Variable, array, array, array):
			this process returns a tuple containing the following 5 elements:
				
				rpn_locs(torch.autograd.Variable): Predict bounding box location after regression. Shape: (N, H, W, A, 4).
				
				rpn_scores(torch.autograd.Variable): Predict foreground scores(propability) for anchors. Shape: (N, H, W, A, 2)
				
				rois(array): region of interests, the rpn would produce some rois to RoI head.the rois is a batch of coordinates.
							descripting the proposal boxes. So the Shape would be (R',4). 4 for 4 cooridinates value of one proposal.
							and :math `R' = \\sum_{i=1}^N R_i`, where R_i is the `i`th  predicted bounding box.
				
				rois_indices(array): this array contains the corresponding indices to rois. Shape: (R',)
				
				anchor(array): Coordinates of enumerated shifted anchors. Shape: (H, W, A, 4)

		'''
		n, _, hh, ww = x.shape
		#Generate proposals from bbox deltas and shifted anchors.
		anchor = _generate_anchors_all(
				np.array(self.anchor_base),
				self.feat_stride, hh, ww)






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
	m.bias.data.zero_()


def _generate_anchors_all(anchor_base, feat_stride, height, width):
	'''
	shifted_x, shifted_y represents the x-coordinates and y-coordinates respectively in the feature map, 
	while feat_stride is defined previously, see details in `class RegionProposalNetwork` Args.

	This function is to generate proposals from bbox deltas and shifted anchors.

	As we have A base anchors(1, A, 4)   (typically, A equals 9)
	to each cell K shifts(K, 1, 4).
	
	we got anchors(K, A, 4), and then reshape them to (K*A, 4)


	returns (K*A, 4)
	'''
	xx = np.arange(0, width*feat_stride, feat_stride)
	yy = np.arange(0, height*feat_stride, feat_stride)
	shitf_x, shift_y = np.meshgrid(xx, yy)            #the shapes of shift_x and shift_y are both (yy, xx)
	shift = np.stack((shift_y.ravel(), shift_x.ravel(),
					  shift_y.ravel(), shift_x.ravel()), axis=1)

	A =anchor_base.shape[0]    #typically is 9
	K = shift.shape[0]

	anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
	anchor = anchor.reshape((K * A, 4)).astype(np.float32)
	return anchor


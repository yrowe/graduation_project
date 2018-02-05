import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils.config import config
from .proposal_layer import ProposalLayer        # @TODO and is relative import necessary?
from .anchor_target_layer import AnchorTargetLayer   # @TODO
from ..utils.net_utils import smooth_l1_loss     # @TODO

class RPN(nn.Module):
	""" region proposal network """
	def __init__(self,din):
		super(RPN, self).__init__()

		self.din = din   #get depth of input feature map

		#some hyper parameters
		# ? feat stride refers to what?
		self.anchor_scales = cfg.ANCHOR_SCALES
		self.anchor_ratios = cfg.ANCHOR_RATIOS 
		self.feat_stride = cfg.FEAT_STRIDE[0]

		#firstly a convolutional layer
		self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias = True)   # ?how to recognize this params refer to what? when they are not concretely denoted 

		#define background/foreground classification score layer.  2*k, k = 3*3 = 9
		self.nc_score_out = len(self.anchor_ratios)*len(self.anchor_scales)*2
		self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

		#define anchor box offset prediction layer
		#output four coordinate point, so the num is 4*k,  k = 9
		self.nc_bbox_out = len(self.anchor_ratios)*len(self.anchor_scales)*4
		self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

		#define proposal layer
		self.RPN_proposal = ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

		#define anchor target layer
		self.RPN_anchor_target = AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

		self.rpn_loss_cls = 0
		self.rpn_loss_box = 0

		@staticmethod
		def reshape(x, d):
			#according to the paper,after 1*1 conv layer and softmax layer,
			#there is a reshape operation respectively
			input_shape = x.size()
			x = x.view(
				input_shape[0],
				int(d),
				int(float(input_shape[1] * input_shape[2])/ float(d)),
				input_shape[3]
			)
			return x

		def forward(self, base_feat, im_info, gt_boxes, num_boxes):
			#after the base feature extractor network,we get base features
			batch_size = base_feat.size(0)

			#firstly a conv layer, and a activation layer
			



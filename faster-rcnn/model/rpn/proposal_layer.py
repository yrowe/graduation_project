# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as numpy

from ..utils.config import cfg 
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from ..nms.nms_wrapper import nms

class _ProposalLayer(nn.Module):
	'''
	Outputs object dection proposals by applying estimated bounding-box
	transformations to a set of regular boxes(called "anchors").
	'''

	def __init__(self, feat_stride, scale, ratios):
		super(_ProposalLayer, self).__init__()

		self._feat_stride = feat_stride
		self._anchors = torch.from_numpy(generate_anchors(scales = np.array(scales),
						ratios=np.array(ratios))).float()
		self._num_anchors = self._anchors.size(0)

		#rois blob: holds R regions of interest, each is a 5-tuple
		#(n, x1, y1, x2, y2) specifying an image batch index n and a
		# rectangle (x1, y1, x2, y2)
		#top[0].reshape(1, 5)

	def forward(self, input):
		#Algorithm:
		#
		# for each (H, W) location i
		#  generate A anchor boxes centered on cell i
		#  apply predicted bbox delta at cell i to each of the anchors
		# clip predicted boxes to image.
		# remove predicted boxes with either height or width < threshold
		# sort all (proposal, score) pairs by score from highest to lowest
		# take top pre_nms
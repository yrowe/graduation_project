# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as numpy

from model.utils.config import cfg
from .generate_anchors import generate_anchors   # @TODO
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_overlaps_batch

class AnchorTargetLayer(nn.Module):
	"""
	 Assign anchors to ground-truth targets. Produce anchor classification
	 labels and bounding-box regression targets.
	"""

	def __init__(self, feat_stride, scales, ratios):
		super(AnchorTargetLayer, self).__init__()

		self._feat_stride = feat_stride
		self._scales = scales
		anchor_scales = scales
		self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
		self._num_anchors = self._anchors.size(0)

		#allow boxes to sit over the edge by a small amount
		self._allowed_border = 0   #default is 0


	def forward(self, input):
		#Algorithm:
		#
		#for each(H, W) location i
		#   generate 9 anchor boxes centered on cell i
		#   apply predicted bbox deltas at cell i to each of the 9 anchors
		#filter out-of-image anchors

		rpn_cls_score = input[0]
		gt_boxes = input[1]
		im_info = input[2]
		num_boxes = input[3]

		# map of shape (..., H, W)
		height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

		batch_size = gt_boxes.size(0)

		feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
		shift_x = np.arange(0, feat_width) * self._feat_stride
		shift_y = np.arange(0, feat_height) * self._feat_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)

		# ? ravel()?
		shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
								shift_x.ravel(), shift_y.ravel())).transpose())
		shitfs = shitfs.contiguous().type_as(rpn_cls_score).float()
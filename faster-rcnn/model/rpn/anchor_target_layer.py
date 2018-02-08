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

		A = self._num_anchors
		K = shifts.size(0)

		#? type_as move two variable to a same gpu ?
		self._anchors = self._anchors.type_as(gt_boxes)  #move to specific gpu
		all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
		all_anchors = all_anchors.view(K*A, 4)

		total_anchors = int(K*A)

		keep = ((all_anchors[:, 0] >= -self._allowed_border)&
				(all_anchors[:, 1] >= -self._allowed_border)&
				(all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border)&
				(all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))

		inds_inside = torch.nonzero(keep).view(-1)  # nonzero? view(-1)?

		#keep only inside anchors
		anchors = all_anchors[inds_side, :]

		#label : 1 is postive, 0 is negative , -1 is don't care
		labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill(-1)  # ? new ?
		bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zeros_()
		bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zeros_()

		# @TODO function bbox_overlaps_batch
		overlaps = bbox_overlaps_batch(anchors, gt_boxes)

		# what does the second param of torch.max do
		max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
		gt_max_overlaps, _ = torch.max(overlaps, 1)

		#two hyper param to be added to config.py 
		if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
			labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

		gt_max_overlaps[gt_max_overlaps==0] = 1e-5
		# ? attribute eq ?
		keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)

		if torch.sum(keep) > 0:
			labels[keep>0] = 1

		#foreground label: above threshold IoU
		labels[max_overlaps >= cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 1

		if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
			labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

		#why num_fg is produced by mul operation
		num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

		sum_fg = torch.sum((labels==1).int(), 1)
		sum_bg = torch.sum((labels==0).int(), 1)

		for i in range(batch_size):
			#subsample positive labels if we have too many
			if sum_fg[i] > num_fg:
				fg_inds = torch.nonzero(labels[i] == 1).view(-1)
				rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
				disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
				labels[i][disable_inds] = -1

			num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]

			#subsample negative labels if we have too many
			if sum_bg[i] > num_bg:
				bg_inds = torch.nonzero(labels[i]==0).view(-1)

				rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
				disable_inds = bg_inds[rand_num[: bg_inds.size(0)-num_bg]]
				labels[i][disable_inds] = -1

		offset = torch.arange(0, batch_size)*gt_boxes.size(1)

		argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
		# function _compute_targets_batch is defined below in this file
		bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

		#use a single value instead of 4 values for easy index
		bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

		if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
			num_examples = torch.sum(labels[i] >= 0)
			positive_weights = 1.0 / num_examples
			negative_weights = 1.0 / num_examples

		else:
			#what does assert do
			assert((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0)&
				   (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

		bbox_outside_weights[labels == 1] = positive_weights
		bbox_outside_weights[labels == 0] = negative_weights

		# function _unmap is defined latter in this file
		labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
		bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
		bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
		bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

		outputs = []   # ? list() ?

		labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
		labels = labels.view(batch_size, 1, A*height, width)
		outputs.append(labels)

		bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()
		outputs.append(bbox_targets)

		anchors_count = bbox_inside_weights.size(1)
		# expand ?
		bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
		bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A).permute(0, 3, 1, 2).contiguous()

		outpus.append(bbox_inside_weights)

		bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
		bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A).permute(0, 3, 1, 2).contiguous()
		outputs.append(bbox_outside_weights)
		 
		return outputs


	def backward(self, top, propagate_down, bottom):
		'''
		this layer does not propagate gradients.
		'''
		pass

	def reshape(self, bottom, top):
		'''
		reshaping happens during the call to forward
		'''
		pass

def _unmap(data, count, inds, batch_size, fill=0):
	'''
	Unmap a subset of item(data) back to the original set
	of items(of size count)
	'''
	if data.dim() == 2:
		ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
		ret[:, inds] = data

	else:
		ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)

	return ret


def _compute_targets_batch(ex_rois, gt_rois):
	'''
	Compute bounding-box regression targets for an image.
	'''

	return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
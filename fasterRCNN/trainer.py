from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import time

from utils import array_tool as at
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
						['rpn_loc_loss',
						 'rpn_cls_loss',
						 'roi_loc_loss',
						 'roi_cls_loss',
						 'total_loss'])

class FasterRCNNTrainer(nn.Module):
	'''
	It provides the cascade of FasterRCNN. and returns losses.
	The losses include:
	  `rpn_loc_loss`: The localization loss of RPN.
	  `rpn_cls_loss`: The classification loss of RPN.
	  `roi_loc_loss`: The localization loss of RoI head.
	  `roi_cls_loss`: The classification loss of RoI head.
	  `total_loss`: The sum of 4 loss above.

	  Args:
	  	faster_rcnn(model.FasterRCNN):
	  		A faster RCNN model that is going to be trained.

	'''

	def __init__(self, faster_rcnn):
		super(FasterRCNNTrainer, self).__init__()
		self.faster_rcnn = faster_rcnn
		self.rpn_sigma = opt.rpn_sigma
		self.roi_sigma = opt.roi_sigma

		#target creator create gt_bbox gt_label etc when training targets.
		self.anchor_target_creator = AnchorTargetCreator()
		self.proposal_target_creator = ProposalTargetCreator()

		self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
		self.loc_normalize_std = faster_rcnn.loc_normalize_std
		self.optimizer = self.faster_rcnn.get_optimizer()

		#self.vis = Visualizer(env=opt.env)

		#indicators for training status
		self.rpn_cm = ConfusionMeter(2)
		self.roi_cm = ConfusionMeter(21)

		self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

	def forward(self, imgs, bboxes, labels, scale):
		'''
		Forward Faster RCNN and calculate losses.

		Here are several notation used.

		`N`: the batchsize of input.  We fixed it to 1.
		`R`: the number of bounding boxes per image.

		Args:
			imgs(torch.autograd.Variable): A single image in Variable type.
			bboxes(torch.autograd.Variable): A batch of bounding boxes.
				shape `(N, R, 4)`
			labels(torch.autograd.Variable):A batch of labels.
			scale(float): Amount of scaling applied to the raw image during preprocessing.

		returns:
			namedtuple of 5 losses.
		'''

		_, _, H, W = imgs.shape
		img_size = (H, W)

		features = self.faster_rcnn.extractor(imgs)

		rpn_locs, rpn_scores, rois, roi_indices, anchor = \
			self.faster_rcnn.rpn(features, img_size, scale)

		#fix batch size to 1. simplify them -- delete the first dimension
		bbox = bboxes[0]
		label = labels[0]
		rpn_score = rpn_scores[0]
		rpn_loc = rpn_locs[0]
		roi = rois 

		# Sample RoIs and forward

		sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
				roi,
				at.tonumpy(bbox),
				at.tonumpy(label),
				self.loc_normalize_mean,
				self.loc_normalize_std)

		#sample_roi_index refers to a same image. so the index are all 0.
		sample_roi_index = torch.zeros(len(sample_roi))
		roi_cls_loc, roi_score = self.faster_rcnn.head(
			features,
			sample_roi,
			sample_roi_index)

		#RPN losses
		gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
									at.tonumpy(bbox),
									anchor,
									img_size
									)

		gt_rpn_label = at.tovariable(gt_rpn_label).long()
		gt_rpn_loc = at.tovariable(gt_rpn_loc)

		rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)   #!TODO private func `rpn_loc_loss`

		rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
		_gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
		_rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label)>-1]
		self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())


		#ROI loss
		n_sample = roi_cls_loc.shape[0]
		roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
		roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
		gt_roi_label = at.tovariable(gt_roi_label).long()
		gt_roi_loc = at.tovariable(gt_roi_loc)

		roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)

		roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

		self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

		losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
		losses = losses + [sum(losses)]       #? sum

		return LossTuple(*losses)

	def reset_meters(self):
		for key, meter in self.meters.items():
			meter.reset()
		self.roi_cm.reset()
		self.rpn_cm.reset()

	def train_step(self, imgs, bboxes, labels, scale):
		self.optimizer.zero_grad()
		losses = self.forward(imgs, bboxes, labels, scale)
		losses.total_loss.backward()
		self.optimizer.step()
		self.update_meters(losses)
		return losses

	def save(self, save_optimizer=False, save_path=None, **kwargs):
		save_dict = dict()
		save_dict['model'] = self.faster_rcnn.state_dict()
		save_dict['config'] = opt._state_dict()
		save_dict['other_info'] = kwargs

		if save_optimizer:
			save_dict['optimizer'] = self.optimizer.state_dict()

		if save_path is None:
			timestr = time.strftime('%m%d%H%M')
			save_path = 'checkpoints/fasterrcnn_%s'%timestr
			for k_, v_ in kwargs.items():
				save_path += '_%s'%v_

		torch.save(save_dict, save_path)
		return save_path

	


def _smooth_l1_loss(x, t, in_weight, sigma):

	#  ? SmoothLossLayer
	sigma2 = sigma**2
	diff = in_weight*(x-t)
	abs_diff = diff.abs()
	flag = (abs_diff.data < (1./sigma2)).float()
	flag = Variable(flag)
	y = (flag*(sigma2/2.)*(diff**2)+(1-flag)*(abs_diff-0.5/sigma2))
	return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
	in_weight = torch.zeros(gt_loc.shape).cuda()
	#Localization loss is calculated only for positive rois.

	in_weight[(gt_label>0).view(-1,1).expand_as(in_weight).cuda()] = 1    #expand_as ?  transform to the same GPU.
	loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)

	loc_loss /= (gt_label>=0).sum()

	return loc_loss 
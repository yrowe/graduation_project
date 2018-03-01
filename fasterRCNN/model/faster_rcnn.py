import torch
import torch.nn as nn
from torch.nn import functional as F 
import cupy as cp 
import numpy as np

from model.utils.bbox_tools import loc2bbox
from utils.config import opt 
from utils import array_tool as at
from model.utils.nms.non_maximum_suppression import non_maximum_suppression
from data.dataset import preprocess 



class FasterRCNN(nn.Module):
	'''
	this class is a cascade for the model faster_rcnn.
	Args:
		extractor: features which had been extracted by the featured part of CNN net(like VGG16).
					see more detail in file (@TODO).
		rpn: Region proposal network. 
				see more details in file(@TODO).
		head: the head part of RoI pooling layer.
				see more details in file(@TODO).

	'''
	def __init__(self, extractor, rpn, head,
					loc_normalize_mean=(0., 0., 0., 0.),
					loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
		super(FasterRCNN, self).__init__()
		self.extractor = extractor
		self.rpn = rpn
		self.head = head

		self.loc_normalize_mean = loc_normalize_mean
		self.loc_normalize_std = loc_normalize_std

		self.nms_thresh = 0.3
		self.score_thresh = 0.05

	@property
	def n_class(self):
		#Total number of classes including the background.
		return self.head.n_class

	def get_optimizer(self):
		'''
		return optimizer
		'''

		lr = opt.lr
		params = []
		for key, value in dict(self.named_parameters()).items():
			if value.requires_grad:
				if 'bias' in key:
					params += [{'params':[value],'lr':lr*2,'weight_decay':0}]
				else:
					params += [{'params':[value],'lr':lr,'weight_decay':opt.weight_decay}]

				self.optimizer = torch.optim.SGD(params, momentum=0.9)
				return self.optimizer

	def forward(self, x, scale=1.):
		'''
		Forward Faster R-CNN

		Scaling parameter `scale`  is used by RPN to determine the threshold to
		select small objects, which are going to be rejected irrespective of their
		confidence scores.

		some notations:
			N: number of batch size
			R': total number of RoIs produced across batches.
				Given R_i proposed RoIs from `i`th image.
				R' = \sum_{i=1}^N R_i
			L: number of classes excluding the background.

		Args:
			x(torch.autograd.Variable): 4D image Variable
			scale(float): Amount of scaling applied to the raw image during preprocessing.

		returns:
			Variable, Variable, array, array:
			Returns tuple of four values listed below.

			roi_cls_locs: Offsets and scalings for the proposed RoIs.
							shape: (R', (L+1)*4)
			roi_scores: Class predictions for the proposed RoIs.
							shape: (R', L+1)
			rois: RoIs proposed by RPN. Its shape is
							shape: (R', 4)
			roi_indices: Batch indices of RoIs. 
							shape: (R', )    !TODO : since we only complete batchsize=1 ,delete this var

		'''
		img_size = x.shape[2:]
		h = self.extractor(x)

		rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)

		roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
		return roi_cls_locs, roi_scores, rois, roi_indices



	def predict(self, imgs, sizes=None, visualize=False):
		'''
		Detect objects from images.

		This method predicts objects for each image.

		Args:
			imgs(iterable of numpy.ndarray): Array holding images.
				All images are in CHW and RGB format.
				and range of their values is `[0, 255]`

		returns:
			tuple of lists
			This method returns a tuple of three lists.
			(bboxes, labels, scores)

		'''

		self.eval()

		if visualize:
			self.nms_thresh = 0.3
			self.score_thresh = 0.7
			prepared_imgs = list()
			sizes = list()
			for img in imgs:
				size = img.shape[1:]
				img = preprocess(at.tonumpy(img))
				prepared_imgs.append(img)
				sizes.append(size)
		else:
			prepared_imgs = imgs


		bboxes = list()
		labels = list()
		scores = list()

		for img, size in zip(prepared_imgs, sizes):
			img = torch.autograd.Variable(at.totensor(img).float()[None], volatile=True)
			scale = img.shape[3]/size[1]
			roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)  #forward method
			roi_score = roi_scores.data
			roi_cls_loc = roi_cls_loc.data
			roi = at.totensor(rois)/scale

			 #convert predictions to bounding boxes in image coordinates.
			 #Bounding boxes are scaled to the scale of the input images.

			mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
			std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

			roi_cls_loc = (roi_cls_loc*std + mean)
			roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

			roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
			cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)), at.tonumpy(roi_cls_loc).reshape((-1, 4)))
			cls_bbox = at.totensor(cls_bbox)

			cls_bbox = cls_bbox.view(-1, self.n_class*4)
			#clip bounding box

			cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
			cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

			prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))

			raw_cls_bbox = at.tonumpy(cls_bbox)
			raw_prob = at.tonumpy(prob)

			bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
			bboxes.append(bbox)
			labels.append(label)
			scores.append(score)

		self.nms_thresh = 0.3
		self.score_thresh = 0.05

		self.train()
		return bboxes, labels, scores  


	def _suppress(self, raw_cls_bbox, raw_prob):
		bbox = list()
		label = list()
		score = list()

		# skip cls_id = 0, since it is background.
		for l in range(1, self.n_class):
			cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
			prob_l = raw_prob[:, l]
			mask = prob_l > self.score_thresh
			cls_bbox_l = cls_bbox_l[mask]
			prob_l = prob_l[mask]
			keep = non_maximum_suppression(cp.array(cls_bbox_l), self.nms_thresh, prob_l)
			keep = cp.asnumpy(keep)

			bbox.append(cls_bbox_l[keep])
			label.append((l-1)*np.ones((len(keep), )))
			score.append(prob_l[keep])

		bbox = np.concatenate(bbox, axis=0).astype(np.float32)
		label = np.concatenate(label, axis=0).astype(np.int32)
		score = np.concatenate(score, axis=0).astype(np.float32)

		return bbox, label, score
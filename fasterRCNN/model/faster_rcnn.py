import torch  as t
import numpy as np 
import cupy as cp 

from utils import array_tool as at   # @TODO
from model.utils.bbox_tools import loc2bbox  # @TODO
from model.utils.nms import non_maximum_suppression

from torch import nn 
from data.dataset import preprocess
from torch.nn import functional as F 
from utils.config import opt


class FasterRCNN(nn.Module):
	'''
	Base class for faster R-CNN

	this is a base class for Faster R-CNN links supporting
	object detection API. The following three stages constitute
	Faster R-CNN.

	1.Feature extraction: Image are taken and their feature maps are calculated.

	2.Region proposal networks: Given the feature maps calculated in the previous stage, produce set of RoIs around objects.

	3.Localization and Classification Heads: Using feature maps that belong
	to the proposed RoIs, classify the categories of the objects in the RoIs
	and improve localizations.

	Each stage is carried out by one of the callable class.
	torch.nn.Modeule
	objects : feature, rpn, head.

	There are two functions: method 'predict' and method '__call__'
	to conduct object detection.

	method 'predict' takes images and returns bounding boxes that are
	converted to image coordinates. This will be useful for a scenario
	when Faster R-CNN is treated as a black box function, for instance.

	method '__call__' is provided for a scenraio when intermediate outputs
	are needed, for instance ,for training and debugging.

 	Links that support object detection API have method predict with
 	the same interface.Please refer to methed predict for further details.

 	Args:
 		extractor (nn.Module): A module that takes a 
 			BCHW(batchsize, channel, height, width) image
 		    array and return feature maps.

 		rpn (nn.Module): A module that has the same interface
 			as 'model.region_proposal_network.RegionProposalNetwork'.

 		head(nn.Module): A module that taks a BCHW variable,
			RoIs and batch indices for RoIs. This returns
			class dependent localization parameters and class scores.
		
		loc_normalize_mean (tuple of four floats) : Mean values of 
			localization estimates.

		loc_normalize_std (tuple of four floats) : Standard deviation
			of localization estimates.

	'''


	def __init__(self, extractor, rpn, head,
				loc_normalize_mean=(0., 0., 0., 0.),
				loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
		super(FasterRCNN, self).__init__()
		self.extractor = extractor
		self.rpn = rpn
		self.head = head

		#mean and std
		self.loc_normalize_mean = loc_normalize_mean
		self.loc_normalize_std = loc_normalize_std
		self.use_preset('evaluate')

	@property
	def n_class(self):
		#Total number of classes including the background
		return self.head.n_class

	def forward(self, x, scale=1.):
		'''Forward Faster R-CNN.
		Scaling parameter : scale is used by RPN to determine
		the threshold to select small objects, which are going to
		be rejected irrespective of their confidence scores.

		Here are notations used.

		math: 'N' is the number of batch_size.
		math: 'R' is the total_number of RoIs produced across batches.
			Given: math R_i proposed RoIs from the math: 'i' the images,

		math: L is the number of classes excluding the background.

		Args:
			x(autograd.Variable): 4D image varible.
			scale(float): Amount of scaling applied to the raw image
				during preprocessing.

		returns:
			Variable, Variable, array, array:
			Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.


        '''

    	img_size = x.shape[2: ]

    	h = self.extractor(x)
    	rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)

    	roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
    	return roi_cls_locs, roi_scores, rois, roi_indices


	def use_preset(self, preset):
		'''Use the given preset during prediction.

		This method changes values of self.nms_thresh and 
		self.score_thresh.These values are a threshold value
		used for non_maximum suppression and a threshold value
		to discard low confidence proposals in method predict
		respectively.

		If the attributes need to be changed to something other than
		the values provided in the presets, please modify them by
		directly accessing the public attributes.

		Args:
			preset ({'visualize', 'evaluate'}): A string to
			determine the preset to use.

		'''

		if preset == 'visualize':
			self.nms_thresh = 0.3
			self.scores_thresh = 0.7
		elif preset == 'evaluate':
			self.nms_thresh = 0.3
			self.score_thresh = 0.05

		else:
			raise ValueError('preset must be visualize or evaluate')

	def _suppress(self, raw_cls_bbox, raw_prob):
		bbox = list()
		label = list()
		score = list()

		for l in range(1, self.n_class):
			cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
			prob_l = raw_prob[:, l]
			mask = prob_l > self.score_thresh
			cls_bbox_l = cls_bbox_l[mask]
			prob_l = prob_l[mask]
			keep = non_maximum_suppression(cp.array(cls_bbox_l), self.nms_thresh, prob_l)
			keep = cp.asnumpy(keep)
			bbox.append(cls_bbox_l[keep])

			label.append((l - 1)*np.ones((len(keep),)))
			score.append(prob_l[keep])

		bbox = np.concatenate(bbox, axis=0).astype(np.float32)
		label = np.concatenate(label, axis=0).astype(np.int32)
		score = np.concatenate(score, axis=0).astype(np.float32)
		return bbox, label, score

	def predict(self, img, sizes=None, visualize=False):
		'''Detect objects from images.

		This method predicts objects for each image.

		Args:
			imgs (iterable of numpy.ndarray):Arrays holding
			images. All images are in CHW and RGB format
			and the range of their value is [0, 255]

		returns:
			tuples of lists:
			This method returns a tuple of three lists.
			(bboxes, labels, scores)
			* **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.


    	'''
    	self.eval()
    	if visualize:
    		self.use_preset(visualize)
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
    		
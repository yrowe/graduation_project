import numpy as np 
import cupy as cp 

from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression

class ProposalCreator:
	''' Proposal regions are generated by calling this object in the precess of rpn.

	the method `__call__` of this object outputs object detection proposals 
	by applying estimated bounding box offsets to a set of anchors.

	This class takes parameters to control number of bounding boxes to
	pass to NMS and keep after NMS.

	If the parameters are negative, it uses all the bounding boxes supplied
	or keep all the bounding boxes returned by NMS.

	Args:
		nms_thresh(float): Threshold value used when calling NMS.
		n_train_pre_nms(int): Number of top scored bounding boxes to keep after passing to NMS in train mode.
		n_train_post_nms(int): Number of top scored bounding boxes to keep after passing to NMS in train mode.
		n_test_pre_nms(int): Number of top scored bounding boxes to keep before passing to NMS in test mode.
		n_test_post_nms(int): Number of top scored bounding boxes to keep after passing to NMS in test mode.
		force_cpu_nms(bool): If this is True,always use NMS in CPU mode.If it is False,the NMS mode is selected based on the type of inputs.
		min_size(int): A parameter to determine the threshold on discarding bounding boxes based on their sizes

	'''

	def __init__(self, 
				parent_model,
				nms_thresh=0.7,
				n_train_pre_nms=12000,
				n_train_post_nms=2000,
				n_test_pre_nms=6000,
				n_test_post_nms=300,
				min_size=16):
		self.parent_model = parent_model
		self.nms_thresh = nms_thresh
		self.n_train_pre_nms = n_train_pre_nms
		self.n_train_post_nms = n_train_post_nms
		self.n_test_pre_nms = n_test_pre_nms
		self.n_test_post_nms = n_test_post_nms
		self.min_size = min_size

	def __call__(self, loc, score, anchor, img_size, scale=1.):
		'''
		input should be ndarray
		Proposal RoIs.

		Inputs: `loc, score, anchor ` refer to the same anchor when indexed by the same index.

		`R` is the total number of anchors.This is equal to product of the height and the width
		of an image the number of anchor bases per pixel.

		Type of the output is same as inputs.

		Args:
			loc(array): Predicted offsets and scaling to anchors.
				Its shape is `(R, 4)`
			score(array): Predicted foreground probability for anchors.
				Its shape is `(R,)`
			anchor(array): Coordinates of anchorsx.Its shape is `(R, 4)`
			img_size(tuple of ints): `(height, width)`,which contains image after scaling.
			scale(float): The scaling factor used to scale an image after reading it from a file.


		Returns:
			array:
			An array of coordinates of proposal boxes.
			Its shape if `(S, 4)`,while `S` is less than `self.n_test_post_nms` in test time
			and less than `self.n_train_post_nms` in train time.
			`S` depends on the size of the predicted bounding boxes and the number of
			bounding boxes discarded by NMS.

		'''

		if self.parent_model.training
			n_pre_nms = self.n_train_pre_nms
			n_post_nms = self.n_train_post_nms

		else:
			n_pre_nms = self.n_test_pre_nms
			n_post_nms = self.n_test_post_nms


		roi = loc2bbox(anchor, loc)

		#do the truncation, lower limit:0, upper limit: img_size 
		roi[:, [0,2]] = np.clip(roi[:, [0,2]], 0, img_size[0])   # y coordinates
		roi[:, [1,3]] = np.clip(roi[:, [1,3]], 0, img_size[1])	 # x coordinates

		#Remove predicted boxes with either height or width less than threshold.
		#while threshold is the product of a preset value :self.min_size(16 in default)
		#and the value `scale`(the scaling size compared to the original image)
		
		min_size = self.min_size * scale
		hs = roi[:, 2] - roi[:, 0]   # y_{max} - y_{min}
		ws = roi[:, 3] - roi[:, 1]   # x_{max} - x_{min}

		keep = np.where((hs>=min_size)&(ws>=min_size))[0]
		roi = roi[keep, :]
		score = score[keep]

		# sort all proposal-score pairs by score from highest to lowest.
		# and keep top pre_nms_topN(6000 in default)

		order = score.argsort()[::-1]
		order = order[:n_pre_nms]
		roi = roi[order,:]

		# Apply nms(eg. threshold = 0.7)
		# Take after_nms_topN(eg. 300)

		keep = non_maximum_suppression(
			cp.ascontiguousarray(cp.asarray(roi)),
			thresh=self.nms_thresh)

		keep = keep[:,n_post_nms]
		roi = roi[keep]

		return roi


class ProposalTargetCreator(object):
	'''
	Assign ground truth bounding boxes to given RoIs.

	`__call__` of this class generates training targets
	for each object proposal.
	This is used to train Faster RCNN.


	Args:
		n_sample(int): The number of sampled regions.
		pos_ratio(float): Fraction of regions that is labeled as a foreground.
		pos_iou_thresh(float): IoU threshold for a RoI to be
			considered as a forground.
		neg_iou_thresh_upper(float): the upper threshold for a RoI to be 
			considered as a background.
		neg_iou_thresh_lower(float): the lower threshold for a RoI to be 
			considered as a background.

			So this RoI that are used as a negative sample. their IoU should in
				[`neg_iou_thresh_lower`, `neg_iou_thresh_upper`]

	'''

	def __init__(self, n_sample=128,pos_ratio=0.25,pos_iou_thresh=0.5,
					neg_iou_thresh_upper=0.5, neg_iou_thresh_lower=0.):
		self.n_sample = n_sample
		self.pos_ratio = pos_ratio
		self.pos_iou_thresh = pos_iou_thresh
		self.neg_iou_thresh_upper = neg_iou_thresh_upper
		self.neg_iou_thresh_lower = neg_iou_thresh_lower

	def __call__(self, roi, bbox, label, 
				 loc_normalize_mean=(0.,0.,0.,0.),
				 loc_normalize_std=(0.1,0.1,0.2,0.2)):
		'''
		Assign the ground truth bounding boxes to anchors.

		The function samples total of `self.n_sample` RoIs
		from the combination of `roi` and `bbox`.

		The RoIs are assigned with the ground truth class labels
		as well as bounding box offsets and scales to match the ground
		truth bounding boxes. As many as `pos_ratio * self.n_sameple`RoIs
		are sampled as foregrounds.

		Offsets and scales of bounding boxes are calculated using
		`model.utils.bbox_tools.bbox2loc`
		Also, types of input arrays and output arrays are the same.

		some notations:
		math:`S` is the total number of sampled RoIs, which equals `self.n_sample`.
		math:`L` is the number of object classes possibly including the background.

		Args:
			roi(array): Region of Interests(RoIs) from which we sample.
				Its shape is `(R, 4)`
			bbox(array): The coordinates of ground truth bounding boxes.
				Its shape is `(R', 4)`
			label(array): Ground truth bounding box labels.Its shape is `(R', )
				range is `[0, L-1]`,where `L` is the number of foreground classes.
			loc_normalize_mean(tuple of four floats): Mean values to 
				normalize cooridinates of bounding boxes.
			loc_normalize_std(tuple of four floats): Standard deviation of the coordinates
				of bounding boxes.

		returns:
			(array, array, array):

			sample_roi: Regions of interests that are sampled.
				Its shape is `(S, 4)`
			gt_roi_loc: Offsets and scales to match the sampled RoIs to
				the ground truth bounding boxes.
				Its shape is `(S, 4)`
			gt_roi_label: Labels assigned to sampled RoIs.Its shape is
				`(S, )`.Its range is `[0, L]`.the label value 0 refers background.

		'''

		n_bbox, _ = bbox.shape

		roi = np.concatenate((roi, bbox), axis=1)

		pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
		iou = bbox_iou(roi, bbox)
		gt_assignment = iou.argmax(axis=1)
		max_iou = iou.max(axis=1)

		# give bias 1 for each foreground label. so label = 0 represents foreground.
		gt_roi_label = label[gt_assignment] + 1

		#select foreground RoIs as those with >= pos_iou_thresh IoU.
		pos_index = np.where(max_iou >= self.pos_ious_thresh)[0]
		pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
		if pos_index.size > 0:
			pos_index = np.random.choice(
				pos_index, size=pos_roi_per_image, repalce=False)

		#select background RoIs as those IoU in
		# [neg_iou_thresh_lower, neg_iou_thresh_upper).
		neg_index = np.where((max_iou < self.neg_iou_thresh_upper)&
							 (max_iou >= self.neg_iou_thresh_lower))[0]
		neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
		neg_roi_per_this_image = int(min(neg_roi_per_this_image, 
										neg_index.size))
		if neg_index.size > 0:
			neg_index = np.random.choice(
					neg_index, size=neg_roi_per_this_image, replace=False)


		#The indices that we're selecting (both positive and negative).
		# ? make sure `keep_index` == 128 always true ? prove?
 		keep_index = np.append(pos_index, neg_index)
 		gt_roi_label = gt_roi_label[keep_index]
 		gt_roi_label[pos_roi_per_this_image:] = 0  # negative label->0
 		sample_roi = roi[keep_index]

 		#Compute offsets and scales to match sampled RoIs to the GTs.
 		# !TODO bbox2loc bbox_tools.py
 		gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
 		gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
 					)/np.array(loc_normalize_std, np.float32))
 		return sample_roi, gt_roi_loc, gt_roi_label

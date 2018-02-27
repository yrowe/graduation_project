from collection import defaultdict
import itertools
import numpy as np 

from model.utils.bbox_tools import bbox_iou

def eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
		iou_thresh=0.5,use_07_metric=False):
	'''
	Calculate average precision based on evaluation code of PASCAL VOC.

	This function evaluates predicted bounding boxes obtained from a 
	dataset which has N images by using average precision for each class.

	The code is based on the evaluation code use in PASCAL VOC Challenge.

	Args:
		pred_bboxes(iterable of numpy.ndarray): An iterable of `N` sets
			of bounding boxes.
			
			each element of this obj has the shape `(R, 4)`,where R is 
			the number of bboxes,which is various among different imgs.
			the second dim represents the coordinates ,the order is
			`y_{min},x_{min}, y_{max},x_{max}`

		pred_labels(iterable of numpy.ndarray): An iterable of labels.
			Similar to `pred_bboxes` its index correspond to an index
			for the base dataset. Its length is N.
	
		pred_scores(iterable of numpy.ndarray): An iterable of confidence.
			scores for predicted bounding boxes. Similar to `pred_bboxes`
			its index correspond to an index for the base dataset.

		gt_bboxes(iterable of numpy.ndarray)

		gt_labels(iterable of numpy.ndarray)

		iou_thresh(float)

		use_07_metric(bool)

	Returns:
		dict:

		ap(numpy.ndarray): An array of average precisions.
			The `l`th value corresponds to the average precision
			for class `l`. If class `l` does not exist in
			either `pred_labels` or `gt_labels`,the corresponding
			value is set to numpy.nan

		map(float): The average Precisions over classes.

	'''

	prec, rec = calc_detection_voc_prec_rec(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, iou_thresh=iou_thresh)



def calc_detection_voc_prec_rec(pred_bboxes, pred_labels, pred_scores,
								gt_bboxes, gt_labels, iou_thresh=0.5):
	pred_bboxes = iter(pred_bboxes)
	pred_labels = iter(pred_labels)
	pred_scores = iter(pred_scores)

	gt_bboxes = iter(gt_bboxes)
	gt_labels = iter(gt_labels)

	gt_difficults = iter(None)

	n_pos = defaultdict(int)
	score = defaultdict(list)
	match = defaultdict(list)

	iter_zip = zip(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults)

	for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in iter_zip:
		
		gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

		for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
			pred_mask_l = pred_label == 1
			pred_bbox_l = pred_bbox[pred_mask_l]
			pred_score_l = pred_score[pred_mask_l]

			order = pred_score_l.argsort()[::-1]
			pred_bbox_l = pred_bbox_l[order]
			pred_score_l = pred_score_l[order]

			gt_mask_l = gt_label == 1
			gt_bbox_l = gt_bbox[gt_mask_l]
			gt_difficult_l = gt_difficult[gt_mask_l]

			n_pos[l] += np.logical_not(gt_difficult_l).sum()
			score[l].extend(pred_score_l)

			if len(pred_bbox_l) == 0:
				continue
			if len(gt_bbox_l) == 0:
				match[l].extend((0, )*pred_bbox_l.shape[0])
				continue

			# VOC evaluation follows integer typed bounding boxes.
			pred_bbox_l = pred_bbox_l.copy()
			pred_bbox_l[:, 2:] += 1
			gt_bbox_l = gt_bbox_l.copy()
			gt_bbox[:, 2:] += 1

			iou = bbox_iou(pred_bbox_l, gt_bbox_l)
			gt_index = iou.argmax(axis = 1)
			#set -1 if there is no matching ground truth
			gt_index[iou.max(axis=1) < iou_thresh] = -1
			del iou

			selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
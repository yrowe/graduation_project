from collections import defaultdict
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

	ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

	return {'ap': ap, 'map':np.nanmean(ap)}



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
			for gt_idx in gt_index:
				if gt_idx >= 0:
					if gt_difficult_l[gt_idx]:
						match[1].append(-1)
					else:
						if not selec[gt_idx]:
							match[1].append(1)
						else:
							match[1].append(0)
				selec[gt_idx] = True

			else:
				match[1].append(0)

	for iter_ in (pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
		if next(iter_, None) is not None:
			raise ValueError('Length of input iterables need to be same')

	n_fg_class = max(n_pos.keys()) + 1
	prec = [None]*n_fg_class
	rec = [None]*n_fg_class

	for l in n_pos.keys():
		score_l = np.array(score[l])
		match_l = np.array(match[l], dtype=np.int8)

		order = score_l.argsort()[::-1]
		match_l = match_l[order]

		tp = np.cumsum(match_l == 1)
		fp = np.cumsum(match_l == 0)

		# If an element of fp+tp is 0,
		# the corresponding element of prec[1] is nan.

		prec[l] = tp/(fp + tp)
		#If n_pos[l] is 0, rec[l] is None
		if n_pos[l] > 0:
			rec[l] = tp / n_pos[l]

	return prec, rec 


def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


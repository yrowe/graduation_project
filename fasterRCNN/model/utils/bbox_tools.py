import numpy as np 


def loc2bbox(src_bbox, loc):
	'''Decoding bounding boxes from bounding box offsets and scales.

	Given bounding box offsets and scales, this function decodes
	the representation to coordinates in 2D image coordinates.

	Given a bounding box(src_bbox) whose center is :math`(y, x)=p_y, p_x`
	and size :math`p_h, p_w`.

	Given scales and offsets(loc) :math`t_y, t_x, t_h, t_w`

	then decoded bounding box's center and size. 
	:math:`\\hat{g}_y`, :math:`\\hat{g}_x`,
	:math:`\\hat{g}_h`, :math:`\\hat{g}_w`.

	the calculating formulas are as follows:

	`\\hat{g}_y = p_h t_y + p_y`
	`\\hat{g}_x = p_w t_x + p_x`
	`\\hat{g}_h = p_h \\exp(t_h)`
	`\\hat{g}_w = p_w \\exp(t_w)`

	the inverse operation of the method provided by paper Faster-RCNN.
	
	Args:
		src_bbox(array): A coordinates of bounding boxes.
			Its shape is :math:`(R, 4)`,These coordinates are
			:math:`p_{y_min}, p_{xmin}, p_{ymax}, p_{xmax}`
		loc(array): An array with offsets and scales.
			Its shape is :math:`(R, 4)`,
			it contains :math:`t_y, t_x, t_h, t_w`

	Returns:
		array:
		Decoded bounding box coordinates.Its shape is :math:`(R, 4)`
		in detail, they are
		:math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
			   \\hat{g}_{ymax}, \\hat{g}_{xmax}.`

	'''
	#in the case of empty bbox
	if src_bbox.shape[0] == 0:
		return np.zeros((0, 4), dtype=loc.dtype)

	# transform the input params  `p_{y_min}, p_{xmin}, p_{ymax}, p_{xmax}`
	# to `p_y, p_x, p_h, p_w`,where p_y and p_x represents 
	#the center coordinates of bbox.

	p_h = src_bbox[:, 2] - src_bbox[:, 0]
	p_w = src_bbox[:, 3] - src_bbox[:, 1]
	p_y = src_bbox[:, 0] + 0.5 * p_h
	p_x = src_bbox[:, 1] + 0.5 * p_w

	#shape `(R, )`

	# got the offsets and scales from input `loc`
	# for the purpose of simplification, we separate them.

	t_y = loc[:, 0]
	t_x = loc[:, 1]    
 	t_h = loc[:, 2]
 	t_w = loc[:, 3]

 	#shape `(R, )`

 	# then decode them to the center coordinates `ctr_y, ctr_x`
 	# and scales `h, w`

 	ctr_y = t_y * p_h + p_y
 	ctr_x = t_x * p_w + p_x
 	h = np.exp(t_h) * p_h
 	w = np.exp(t_w) * p_w

 	roi = np.zeros(loc.shape, dtype=loc.dtype)

 	roi[:, 0:1] = (ctr_y - 0.5 * h)[:, np.newaxis]   #y_min
 	roi[:, 1:2] = (ctr_x - 0.5 * w)[:, np.newaxis]   #x_min
 	roi[:, 2:3] = (ctr_y + 0.5 * h)[:, np.newaxis]   #y_max
 	roi[:, 3:4] = (ctr_x + 0.5 * w)[:, np.newaxis]   #x_max

 	return roi


def bbox2loc(src_bbox, dst_bbox):
	'''
	given src_bbox and dst_bbox, encoding its loc,

	given bounding boxes, this function computes offsets and scales
	to match the source bounding boxes to the GT bounding boxes.
	Mathematically, given a bounding box whose center is
	:math`(y, x) = p_y, p_x` and size is :math`(p_h, p_w)`
	and the ground truth bounding box whose center is
	:math`(g_y, g_x)` and size :math:(g_h, g_w), the offsets and scales
	between these two bbox are computed as follows:

	`:math:` t_y = 
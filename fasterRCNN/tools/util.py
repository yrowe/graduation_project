import numpy as np 
from PIL import Image
import random

def resize_bbox(bbox, in_size, out_size):
	'''
	Resize bounding boxes according to image scale

	Args:
		bbox(numpy.ndarray): An array whose shape is `(R, 4)`.
			`R` is the number of bounding boxes.
		in_size(tuple): A tuple of length 2. The height and
			width of the image before resized
		out_size(tuple): height and width of image after resized.

	returns:
		bbox(numpy.ndarray): shape `(R, 4)`
	'''

	bbox = bbox.copy()  # shallow copy necessary?
	y_scale = out_size[0] / in_size[0]
	x_scale = out_size[1] / in_size[1]
	bbox[:, 0] = y_scale * bbox[:, 0]
	bbox[:, 2] = y_scale * bbox[:, 2]
	bbox[:, 1] = x_scale * bbox[:, 1]
	bbox[:, 3] = x_scale * bbox[:, 3]
	return bbox

def read_image(path, dtype=np.float32):
	'''
	read an image from a file.

	This function reads an image from given file.

	Args:
		path(string): path of the image file.
		dype: type of array.

	returns:
		img(np.ndarray): A RGB image which dim order (C,H,W)

	'''

	f = Image.open(path)
	try:

		img = f.convert('RGB')
		img = np.asarray(img, dtype=dtype)
	finally:
		if hasattr(f, 'close'):
			f.close()

	# (H, W, C) --> (C, H, W)
	return img.transpose((2, 0, 1))


def random_flip(img, y_random=False, x_random=False, copy = False):
	'''
	randomly flip an image in vertical or horizontal direction.

	Args:
		img(numpy.ndarray): An array that gets flipped in random. CHW format.
		y_random(bool): Randomly flip in vertical direction.
		x_random(bool): Randomly flip in horizontal direction.
		return_param(bool): Returns information of flip.
		copy(bool): If False, a view of img will be returned.

	Returns:
		numpy.ndarray or (numpy.ndarray, dict):

		If return_param = True.
		 will return a extra variable param, which points out whether this img got
		 y_flip, or  x_flip.

	'''

	y_flip, x_flip = False, False
	if y_random:
		y_flip = random.choice([True, False])

	if x_random:
		x_flip = random.choice([True, False])

	if y_flip:
		img = img[:, ::-1, :]
	if x_flip:
		img = img[:, ::-1, :]

	if copy:
		img = img.copy()

	return img, {'y_flip': y_flip, 'x_flip': x_flip}

def flip_bbox(bbox, size, y_flip=False, x_flip=False):

	'''
	Args:
		bbox(numpy.ndarray): An array whose shape is `(R, 4)`, 
			where R is the number of bbox in this img.
			the coordinates are also after scaled.

		size(tuple): height and width of this img after scaled.

		y_flip(bool): flip bbox according if this img after vertical flip

		x_flip(bool): flip bbox according if this img after horizontal flip

	return:
		bbox(numpy.ndarray): flip bbox or not.
	'''

	H, W = size
	bbox = bbox.copy()
	if y_flip:
		y_max = H - bbox[:, 0]
		y_min = H - bbox[:, 2]
		bbox[:, 0] = y_min
		bbox[:, 2] = y_max

	if x_flip:
		x_max = W - bbox[:, 1]
		x_min = W - bbox[:, 3]
		bbox[:, 1] = x_min
		bbox[:, 3] = x_max
	return bbox 
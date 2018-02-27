import numpy as np
from skimage import transform as sktsf


from .voc_dataset import VOCBboxDataset
from . import util



def preprocess(img, min_size=600, max_size=1000):
	'''
	Preprocess an image for feature extraction

	1.resize:
		resize the original image to the requirement that:
		the longer edge is less than max_size,while the 
		shorter one is greater than min_size.
		these two principle are required simultaneously.

	2.normalization:
		use the method normalization that Imagenet Competition required.

	Args:
		img(numpy.ndarray): An image. This is in CHW and RGB format.
				The range of its value is [0, 255]

	return:
		numpy.ndarray: preprocessed img.

	'''
	C, H, W = img.shape
	scale1 = min_size/min(H, W)
	scale2 = max_size/max(H, W)
	scale = min(scale1, scale2)

	img = img/255.
	img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect')

	return normalize(img)

def normalize(img):

	'''
	return [-125, 125] BGR
	'''

	img = img[[2, 1, 0], :, :]  #RGB - BGR
	img = img*255
	mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
	img = (img - mean).astype(np.float32, copy=True)
	return img 



class Dataset:
	def __init__(self, opt):
		self.opt = opt
		self.db = VOCBboxDataset(opt.voc_data_dir)
		self.tsf = Transform(opt.min_size, opt.max_size)

	def __getitem__(self, idx):
		ori_img, bbox, label = self.db.get_example(idx)
		img, bbox, label, scale = self.tsf((ori_img, bbox, label))
		return img.copy(), bbox.copy(), label.copy(), scale 

	def __len__(self):
		return len(self.db)

class TestDataset:
	def __init__(self, opt):
		self.opt = opt
		self.db = VOCBboxDataset(opt.voc_data_dir, split='test')
	def __getitem__(self, idx):
		ori_img, bbox, label, difficult = self.db.get_example(idx)
		img = preprocess(ori_img)
		return img, ori_img.shape[1:], bbox, label

	def __len__(self):
		return len(self.db)


class Transform:
	def __init__(self, min_size=600, max_size=1000):
		self.min_size = min_size
		self.max_size = max_size

	def __call__(self, in_data):
		img, bbox, label = in_data
		_, H, W = img.shape
		img = preprocess(img, self.min_size, self.max_size)
		_, o_H, o_W = img.shape
		scale = o_H/H
		bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

		#horizontally flip
		img, params = util.random_flip(img, x_random=True)
		bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

		return img, bbox, label, scale

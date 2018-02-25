import os
import xml.etree.ElementTree as ET 
import numpy as np 

from .util import read_image

class VOCBboxDataset:
	'''
	Args:
		data_dir(string): Path to the root of the training data. defined in config
		split({'train', 'val', 'trainval', 'test'}): Select a split of the dataset.
		year({'2007', '2012'}): Use a dataset prepared for the year of dataset.
		use_difficult(bool): under delete.
		return_difficult(bool): under delete.
	'''

	def __init__(self, data_dir, split='trainval'):
		id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
		self.ids = [id_.strip() for id_ in open(id_list_file)]
		self.data_dir = data_dir
		#self.use_difficult = use_difficult
		#self.return_difficult = return_difficult
		self.label_names = VOC_BBOX_LABEL_NAMES

	def __len__(self):
		return len(self.ids)

	def get_example(self, i):
		'''
		Returns the i-th example.

		Returns a color image and bounding boxes. The image is in CHW format.
		The returned image is RGB.

		Args:
			i(int): The index of the example.

		returns:
		 	tuple of an image and bounding boxes.
		'''

		id_ = self.ids[i]
		anno = ET.parse(
				os.path.join(self.data_dir, 'Annotations',id_ + '.xml'))
		bbox = list()
		label = list()

		for obj in anno.findall('object'):
			bndbox_anno = obj.find('bndbox')
			#substract 1 to make pixel indexes 0-based.
			bbox.append([int(bndbox_anno.find(tag).text)-1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
			name = obj.find('name').text.lower().strip()
			label.append(VOC_BBOX_LABEL_NAMES.index(name))

		bbox = np.stack(bbox).astype(np.float32)
		label = np.stack(label).astype(np.int32)

		img_file = os.path.join(self.data_dir, 'JPEGImages', id_+'.jpg')
		img = read_image(img_file)

		return img, bbox, label 

	__getitem__ = get_example

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
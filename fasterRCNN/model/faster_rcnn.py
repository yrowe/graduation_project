import torch
import torch.nn as nn


class fasterRCNN(nn.Module):
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
	def __init__(self, extractor, rpn, head):
		super(fasterRCNN, self).__init__()
		self.extractor = extractor
		self.rpn = rpn
		self.head = head

	@property
	def n_class(self):
		#Total number of classes including the background.
		return self.head.n_class
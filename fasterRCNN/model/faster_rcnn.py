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
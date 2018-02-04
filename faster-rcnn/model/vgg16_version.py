import torch
import torch.nn as nn
import torch.n.functional as F
from torch.autograd import Variable
import torchvision.models as models

from utils.config import cfg

class Faster_RCNN_with_VGG16(nn.Module):
	def __init__(self,classes):
		self.model_path = cfg.PATH   #the pretrained model path
		self.classes = classes       #concrete classes of the dataset
		self.n_classes = len(classes) #number of classes
		self.din = 512               #depth of rpn's input features

		#two type of loss
		self.RCNN_loss_cls = 0
		self.RCNN_loss_bbox = 0

		#define rpn
		self.RCNN_rpn = RPN(self.din)
		self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)
		self.RCNN_roi_pool = RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
		self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE,cfg.POOLING_SIZE, 1.0/16.0)

		self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

		vgg = models.vgg16()  #torchvision.models
		state_dict = torch.load(self.model_path)
		vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

		#vgg net is mainly consist of two part.
		#features and classifier,you could instance a vgg model than print it to look its structure

		self.RCNN_base = 

		vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])


	def forward(self, im_data, im_info, gt_boxes, num_boxes):
		batch_size = im_data.size(0)   #due to the limit of GPU memory,batchsize generally equals 1.
		
		im_info = im_info.data    #Tensor has two part,Storage and data, now we get data of it
		gt_boxes = gt_boxes.data
		num_boxes = num_boxes.data

		#feature extractor. VGG's conv,relu,maxpooling layer


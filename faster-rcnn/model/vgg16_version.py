import torch
import torch.nn as nn
import torch.n.functional as F
from torch.autograd import Variable
import torchvision.models as models

from utils.config import cfg
from rpn.rpn import RPN

class Faster_RCNN_with_VGG16(nn.Module):
	def __init__(self,classes):
		super(Faster_RCNN_with_VGG16, self).__init__()    #what does this line mean?should it be necessary?
		self.model_path = cfg.PATH   #the pretrained model path
		self.classes = classes       #concrete classes of the dataset
		self.n_classes = len(classes) #number of classes
		self.din = 512               #depth of rpn's input features

		#two type of loss
		self.RCNN_loss_cls = 0
		self.RCNN_loss_bbox = 0

		#define rpn
		# @TODO RPN function 
		self.RCNN_rpn = RPN(self.din) 
		# @TODO ProposalTargetLayer function
		self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)
		# @TODO RoIPooling function
		self.RCNN_roi_pool = RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
		# @TODO RoIAlignAvg function
		self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE,cfg.POOLING_SIZE, 1.0/16.0)

		self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

		vgg = models.vgg16()  #torchvision.models
		state_dict = torch.load(self.model_path)
		vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

		#vgg net is mainly consist of two part:
		#features and classifier,you could instance a vgg model then print it to look its structure

		#RCNN_base is used to extract features,and it doesn't use the last (maxpooling?) layer of vgg16
		self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

		#also the fully connected layers of vgg is used to as the head of RCNN
		vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
		#for the purpose of saving GPU memory, we fix the layer before conv3 of VGG16
		for layer in range(10):
			for p in self.RCNN_base[layer].parameters():p.requires_grad = False

		#the classifier of VGG is also used for the head of RoI pooling
		self.RCNN_top = vgg.classifier

		self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
		self.RCNN_bbox_pred = nn.Linear(4096, 4*self.n_classes)


		#init the some weight of the net above. the function is at this script.
		_normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
		_normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
		_normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
		_normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
		_normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


	def forward(self, im_data, im_info, gt_boxes, num_boxes):
		#due to the limit of GPU memory,batchsize generally equals 1.
		batch_size = im_data.size(0)
		
		#Tensor has two part,Storage and data, now we get data of it
		im_info = im_info.data    
		gt_boxes = gt_boxes.data
		num_boxes = num_boxes.data

		#feature extractor. VGG's conv,relu,maxpooling layer to get the original feature map
		base_feat = self.RCNN_base(im_data)

		#feed base feature map to RPN to obtain rois
		rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

		#when it is training, use ground truth bounding box for refining(?)
		if self.training:
			roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
			rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

			rois_label = Variable(rois_label.view(-1).long())  #? view(?) means what?
			rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
			rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
			rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

		else:
			rois_label = None
			rois_target = None
			rois_inside_ws = None
			rois_outside_ws = None
			rpn_loss_cls = 0
			rpn_loss_bbox = 0

		rois = Variable(rois)
		#do roi pooling based on predicted rois

		# @TODO function affine_grid_gen
		if cfg.POOLING_MODE == 'crop':
			grid_xy = _affine_grid_gen(rois.view(-1, 5),base_feat.size()[2:], self.grid_size)
			grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]],3).contiguous()
			pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach()) # ? what does detach mean
			if cfg.CROP_RESIZE_WITH_MAX_POOL:
				pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
		elif cfg.POOLING_MODE == 'align':
			pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
		elif cfg.POOLING_MODE == 'pool':
			pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))


		# feed pooled features to top model
		pooled_feat = self._head_to_tail(pooled_feat)

		#compute bbox offset
		bbox_pred = self.RCNN_bbox_pred(pooled_feat)
		if self.training:
			#select the corresponding columns according to roi labels
			bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1)/ 4), 4)
			bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label_size(0), 1, 1).expand(rois_label.size(0), 1, 4))
			bbox_pred = bbox_pred_select.squeeze(1)	

		#compute object classification probability
		cls_score = self.RCNN_cls_score(pooled_feat)
		cls_prob = F.softmax(cls_score)

		#why init again
   		RCNN_loss_cls = 0
   		RCNN_loss_bbox = 0

   		if self.training:
   			#classification loss
   			RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

   			#bounding box regression L1 loss
   			# @TODO smooth_l1_loss function
   			RCNN_loss_bbox = smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

   		cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
   		bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

   		return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label



	def _head_to_tail(self, pool5):
		pool5_flat = pool5.view(pool5.size(0), -1)
		fc7 = self.RCNN_top(pool5_flat)

		return fc7


	def _normal_init(m, mean, stddev, truncated=False):
		# weight initalizer: truncated normal and random normal.
		if truncated:
			m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
		else:
			m.weight.data.normal_(mean, stddev)
			m.bias.data.zero_()
from pprint import pprint

class Config:
	voc_data_dir = '/home/wrc/PASCAL_VOC/VOCdevkit/VOC2007'
	min_size = 600   #image resize
	max_size = 1000  #image resize
	num_workers = 8

	#sigma for l1_smooth_loss
	rpn_sigma = 3.
	roi_sigma = 1.

	#param for optimizer
	weight_decay = 0.0005
	lr_decay = 0.1
	lr = 1e-3

	#visulization
	env = 'faster-rcnn'
	port = 8097
	plot_every = 40  #vis every N iter

	#preset
	data = 'voc'
	pretrained_model = 'vgg16'

	#training epoch
	epoch = 14

	use_adam = False
	use_chainer = False
	use_drop = False

	debug_file = '/tmp/debugf'

	test_num = 10000
	#model
	load_path = None

	caffe_pretrain = True  #use caffe pretrained model instead of torchvision's
	caffe_pretrain_path = '/home/wrc/pretrained_model/vgg16_caffe.pth'

	def _parse(self, kwargs):
		state_dict = self._state_dict()
		for k, v in kwargs.items():
			if k not in state_dict:
				raise ValueError('Unknown Option: "--%s"'%k)
			setattr(self, k, v)

			print('=======use customed config======')
			pprint(k, v)
			print('===============end===============')

	def _state_dict(self):
		return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()
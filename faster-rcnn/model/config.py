#This text is mainly used to save some hyper parameters

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#pretrained model path
__C.PATH = '/home/wrc/pretrained_model/vgg16_caffe.pth'

#Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# @TODO what does this para mean
__C.CROP_RESIZE_MAX_POOL = True

#
__C.POOLING_MODE = 'crop'

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False
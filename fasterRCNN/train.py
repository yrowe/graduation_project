import os
import fire
import torch
from torch
from torch.autograd import Variable
from torch.utils import data as data_

from data.dataset import Dataset, TestDataset, inverse_normalize
from utils.config import opt
from model import FasterRCNNVGG16
from utils import array_tool as at 
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

def train(**kwargs):
	opt._parse(kwargs)
	dataset = Dataset(opt)
	
	
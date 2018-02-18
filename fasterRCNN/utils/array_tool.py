import torch
import numpy as np 

def totensor(data, cuda=True):
	if isinstance(data, np.ndarray):
		tensor = torch.from_numpy(data)
	if isinstance(data, torch._TensorBase):
		tensor = data
	if isinstance(data, torch.autograd.Variable):
		tensor = data.data

	tensor = tensor.cuda()
	return tensor
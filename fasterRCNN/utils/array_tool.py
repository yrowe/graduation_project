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


def tonumpy(data):
	if isinstance(data, np.ndarray):
		return data
	if isinstance(data, torch._TensorBase):
		return data.cpu().numpy()
	if isinstance(data, torch._autograd.Variable):
		return tonumpy(data.data)

def tovariable(data):
	if isinstance(data, np.ndarray):
		return tovariable(totensor(data))
	if isinstance(data, torch._TensorBase):
		return t.autograd.Variable(data)
	if isinstance(data, torch.autograd.Variable):
		return data

def scalar(data):
	if isinstance(data, np.ndarray):
		return data.reshape(1)[0]
	if isinstance(data, t._TensorBase):
		return data.view(1)[0]

	if isinstance(data, torch.autograd.Variable):
		return data.data.view(1)[0]
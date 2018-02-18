from collections import namedtuple
from string import Template

import cupy as cp
import torch
import torch.nn as nn
from torch.autograd import Function

from model.utils.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple(['Stream',['ptr']])

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name,code,**kwargs):
	cp.cuda.runtime.free(0)
	code = Template(code).substitute(**kwargs)
	kernel_code = cupy.cuda.compile_with_cache(code)
	return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
	return (N+K-1)//K

class RoI(Function):
	def __init__(self, outh, outw, spatial_scale):
		self.forward_fn = load_kernel('roi_forward', kernel_forward)
		self.backward_fn = load_kernel('roi_backward', kernel_backward)
		self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

	def forward(self, x, rois):
		x = x.contiguous()
		rois = rois.contiguous()
		self.in_size = B, C, H, W = x.size()
		self.N = N = rois.size(0)
		output = torch.zeros(N, C, self.outh, self.outw).cuda()
		self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
		self.rois = rois
		args = [x.data_ptr(), rois.data_ptr(),
					output.data_ptr(),
					self.argmax_data.data_ptr(),
					self.spatial_scale, C, H, W,
					self.outh, self.outw,
					output.numel()]

		stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
		self.forward_fn(args=args,
						block=(CUDA_NUM_THREADS, 1, 1),
						grid=(GET_BLOCKS(output.numel()),1,1),
						stream=stream)
		return output

	def backward(self, grad_output):
		grad_output = grad_output.contiguous()
		B, C, H, W = self.in_size
		grad_input = t.zeros(self.in_size).cuda()
		stream = Stream(ptr=)
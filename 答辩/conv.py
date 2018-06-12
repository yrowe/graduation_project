import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

me = Image.open("me.jpg")
input = ToTensor(me).unsqueeze(0)

kernel = torch.ones(3,3)/9
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3,3), 1, bias= True)
conv.weight = kernel.view(1,1,3,3)

out = conv(input)
ToPILImage(out.squeeze(0))
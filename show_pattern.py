import torch.nn.functional as F
from net.memnet import MemNet
import torch
import cv2 
import numpy as np

memnet = MemNet(in_channels=1, channels=16, FeaturemapNum=104, num_resblock=4, num_memblock=6)
start_epoch = 995
memnet.load_state_dict(torch.load("saved_models/uie-memnet_%d.pth" % (start_epoch)))

pattern = memnet.patten_kernel_weight
pattern = F.sigmoid(pattern)
pattern = (torch.round(pattern) - pattern).detach() + pattern

for i in range(pattern.shape[0]):
    p = pattern[i,0,...].detach().cpu().numpy()
    p = (255*p).astype(np.uint8)
    cv2.imwrite(f'patten_{i}.png', p)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim
from net.memnet import MemNet
from utils.utils import *
from utils.FDL import *
import cv2
import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv
import random
import torch.utils.data as dataf
import torch.nn.functional as F





dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.FloatTensor)




# Initialize generator 
memnet = MemNet(in_channels=1, channels=32, FeaturemapNum=208, num_resblock=6, num_memblock=6).cuda()
memnet.load_state_dict(torch.load("./saved_models/uie-memnet_923.pth"))

memnet.eval()


path='./input1/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
i=1
for item in path_list:
    impath=path+item
    imgx= cv2.imread(path+item,0)
    imgx=cv2.resize(imgx,(1024,1024))
    #imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype(dtype)

    imgx= torch.from_numpy(imgx)
    imgx=imgx.permute(2,0,1).unsqueeze(0)
    imgx=imgx/255.0
    #plt.imshow(imgx[0,:,:,:])
    #plt.show()
    imgx = Variable(imgx).cuda()
    #print(imgx.shape)
    out=memnet(imgx)
    save_image(out, "./output/%d.jpg" % (i), nrow=5, normalize=True)
    i=i+1



def compute_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def compute_mse(img1,img2):
    mse=np.mean( (img1/255. - img2/255.) ** 2 )
    return mse



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



path1='./input1/'#要改
path2='./output/'#要改
path_list = os.listdir(path1)
path_list.sort(key=lambda x:int(x.split('.')[0]))
PSNR=[]
SSIM=[]
for item in path_list:
    impath1=path1+item
    impath2=path2+item
    imgx= cv2.imread(impath1)
    imgx=cv2.resize(imgx,(256,256))
    imgy= cv2.imread(impath2)
    imgy=cv2.resize(imgy,(256,256))
    #print(imgx.shape)
    psnr1=compute_psnr(imgx[:,:,0],imgy[:,:,0])
    psnr2=compute_psnr(imgx[:,:,1],imgy[:,:,1])
    psnr3=compute_psnr(imgx[:,:,2],imgy[:,:,2])
    
    
    ss=calculate_ssim(imgx,imgy)
    psnr=(psnr1+psnr2+psnr3)/3.0
    if psnr>30:
        print(impath2+str(psnr))

    PSNR.append(psnr)
    SSIM.append(ss)



PSNR=np.array(PSNR)    
print(PSNR.mean())


SSIM=np.array(SSIM)    
print(SSIM.mean())
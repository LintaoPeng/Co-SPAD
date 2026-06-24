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
from torch.nn.utils import clip_grad_norm_

dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
#torch.cuda.set_device(2)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids = [0, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
torch.set_default_tensor_type(torch.FloatTensor)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""

    memnet.eval()
    i=random.randrange(1,50)
    input_x = Variable(X_test[i,:,:,:]).cuda()
    label_y = Variable(y_test[i,:,:,:]).cuda()
    input_x=input_x.unsqueeze(0)
    label_y=label_y.unsqueeze(0)
    output_y = memnet(input_x)
    #print(fake_B.shape)
    imgx=label_y.data
    imgy=output_y.data
    x=imgx[:,:,:,:]
    y=imgy[:,:,:,:]
    img_sample = torch.cat((x,y), -2)
    save_image(img_sample, "images/%s/%s.png" % ('results', batches_done), nrow=5, normalize=True)#要改




training_x=[]
path='./data/train/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item,0)
    imgx=cv2.resize(imgx,(1024,1024))
    training_x.append(imgx)   

X_train = []
for features in training_x:
    X_train.append(features)

X_train = np.array(X_train).reshape(-1,1,1024,1024)
X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_train=X_train.astype(dtype)
X_train= torch.from_numpy(X_train)
#X_train=X_train/255.0
print("input shape:",X_train.shape)
y_train = X_train



test_x=[]
path='./data/test/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx = cv2.imread(path+item,0)
    imgx = cv2.resize(imgx,(1024,1024))
    test_x.append(imgx)


X_test=[]
for features in test_x:
    X_test.append(features)

X_test = np.array(X_test).reshape(-1,1,1024,1024)
X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())
X_test=X_test.astype(dtype)
X_test= torch.from_numpy(X_test)
print("output shape:",X_test.shape)
y_test=X_test




dataset = dataf.TensorDataset(X_train,y_train)
#loader = dataf.DataLoader(dataset, batch_size=4, shuffle=True,num_workers=4)
loader = dataf.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,           # 增加 CPU worker 数（根据核数调整）
    pin_memory=True,         # 启用固定内存，加速 CPU→GPU 传输
    prefetch_factor=4,       # 让每个 worker 预加载多个 batch
    persistent_workers=True  # 避免反复创建 worker
)


memnet = MemNet(in_channels=1, channels=32, FeaturemapNum=208, num_resblock=6, num_memblock=6)
#memnet = torch.nn.DataParallel(memnet, device_ids=device_ids)
# Enable multi-GPU training using DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    memnet = nn.DataParallel(memnet)
memnet = memnet.cuda()



#net.apply(weights_init_kaiming)
MSE= nn.L1Loss(size_average=False).cuda()
SSIM = pytorch_ssim.SSIM().cuda()
FDL_loss = FDL(loss_weight=1.0,alpha=2.0,patch_factor=4,ave_spectrum=True,log_matrix=True,batch_matrix=True).cuda()



LR=0.00004

optimizer = torch.optim.Adam(memnet.parameters(), lr=LR, betas=(0.5, 0.999))
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.8)



use_pretrain=True
if use_pretrain:
    # Load pretrained models
    start_epoch=126
    memnet.load_state_dict(torch.load("saved_models/uie-memnet_%d.pth" % (start_epoch)))
    print('successfully loading epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 0
    print('No pretrain model found, training will start from scratch！')


# ----------
#  Training
# ----------
f1 = open('psnr.csv','w',encoding='utf-8')#要改
csv_writer1 = csv.writer(f1)
f2 = open('SSIM.csv','w',encoding='utf-8')#要改
csv_writer2 = csv.writer(f2)

checkpoint_interval=5
epochs=start_epoch
n_epochs=1000
sample_interval=1000

# ingnored when opt.mode=='S'
psnr_max=0
psnr_list = [] 
prev_time = time.time()



for epoch in range(epochs,n_epochs):
    psnr_list = []
    for i, batch in enumerate(loader):

        # Model inputs
        #Input = Variable(batch[0]).cuda().contiguous() 
        #GT = Variable(batch[1]).cuda().contiguous()
        Input = batch[0].to(device='cuda', non_blocking=True)
        GT = batch[1].to(device='cuda', non_blocking=True)




        # ------------------
        #  Train 
        # ------------------

        optimizer.zero_grad()

        # loss
        output = memnet(Input)
        loss_l2= MSE(output, GT)/(GT.size()[2]**2) 
        loss_ssim=1-SSIM(output,GT)
        ssim_value = -(loss_ssim.item()-1)
        fdl_loss = FDL_loss(output, GT)
        loss_final=loss_ssim*10+loss_l2*10+fdl_loss*10000


        loss_final.backward()
        optimizer.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(loader) + i
        batches_left = n_epochs * len(loader) - batches_done
        out_train= torch.clamp(output, 0., 1.) 
        psnr_train = batch_PSNR(out_train,GT, 1.)
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if batches_done%100==0:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %f] [SSIM: %f][loss: %f][fdl_loss: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(loader),
                    psnr_train,
                    ssim_value,
                    loss_final.item(),
                    fdl_loss.item()*5000, 
                    time_left,
                )
            )


        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done)
            csv_writer1.writerow([str(psnr_train)])
            csv_writer2.writerow([str(ssim_value)])
        psnr_list.append(psnr_train)

    PSNR_epoch=np.array(psnr_list)
    if PSNR_epoch.mean()>psnr_max:
        torch.save(memnet.state_dict(), "saved_models/uie-memnet_%d.pth" % (epoch))
        psnr_max=PSNR_epoch.mean()
        print("")
        print('A checkpoint Saved PSNR= %f'%(psnr_max))


    scheduler.step()
#    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
#        # Save model checkpoints
#        torch.save(memnet.state_dict(), "saved_models/uie-memnet_%d.pth" % (epoch))
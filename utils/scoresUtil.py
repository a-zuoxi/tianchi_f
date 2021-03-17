import os

from torch_fidelity import calculate_metrics
import argparse

import imageio
import scipy
import tqdm
from PIL import Image

from progressbar import *

import lpips
import torch
from torchvision import transforms

use_gpu = True
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and use_gpu) else 'cpu')

input1 = '../attacks/tianchi2021_adv/m_di2_fgsm_resnet152_lossTest/images/'
# input1 = '/home/f/Downloads/Attack-ImageNet/images/'

input2 = '../data/imagenet_all/images/'


#
# adv = Image.open(input1+"0.jpg")
# adv = transforms.ToTensor()(adv)
# clean = Image.open(input2+"0.jpg")
# clean = transforms.ToTensor()(clean)
# print(adv.shape, clean.shape)
# result = (clean - adv).pow(2).sum().pow(0.5)
# print(result.item())
# # input2 = '/home/f/Downloads/tianchi2021/data/imagenet_round1_210122/images/'
# metrics_dict = calculate_metrics(input1, input2, cuda=True, isc=False, fid=True, kid=False, verbose=False)
# print("input1: ", input1, "input2: ", input2)
# fid = metrics_dict["frechet_inception_distance"]
# fid = 1- min(fid,200)/200.0
# score_fid = pow(fid,0.5)
# print("score_fid: ", score_fid)

import tqdm
import torch
img0 = torch.zeros(10,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(10,3,64,64)

files = os.listdir(input1)
print(len(files))
loss_fn = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

total_lpips= 0
widgets = ['LPIPS :', Percentage(), ' ', Bar('#'), ' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets)
for file in tqdm.tqdm(files):
    # Load images
    img0 = lpips.im2tensor(lpips.load_image(os.path.join(input1,file))) # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(os.path.join(input2,file)))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Compute distance
    with torch.no_grad():
        dist01 = loss_fn(img0,img1)
    total_lpips += dist01

lpips = total_lpips / 5000
print("lpips: ",lpips)
score_lpips = pow(1-2*(min(max(lpips,0.2), 0.7)-0.2),0.5)
if isinstance(score_lpips, torch.Tensor):
    score_lpips = score_lpips.item()
# 0.2331
# print("score_fid:",score_fid,", score_lpips:",score_lpips)
# score_sub = 100*score_fid*score_lpips
# print("score_sub without score_ars is ", score_sub)
# print(score_fid,score_lpips,score_sub)
print(score_lpips)
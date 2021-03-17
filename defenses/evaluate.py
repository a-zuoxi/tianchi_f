import glob
import os
import argparse
import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import CIFAR10

#from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)


DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() ) else 'cpu')
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

input='./data/cifar10'
#path='/media/wanghao/000F5F8400087C68/CYJ-5-29/DDN/fast_adv/attacks/DeepFool'
#test_set=CIFAR10(input, train=False, transform=test_transform, download=True)
test_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(0,30000)))
train_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(48000, 50000)))

test_loader = data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)


m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
weight_norm='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
weight_AT='./weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
weight_ALP='/media/unknown/Data/PLP/fast_adv/defenses/weights/AT+ALP/cifar10acc0.8699999809265136_50.pth'
weight_conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_mixed_Attention/cifar10acc0.8759999752044678_100.pth'
weight_attention='/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_Attention/cifar10acc0.8729999780654907_120.pth'
weight_smooth='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
weight_025smooth='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
weight_05smooth='/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'
weight_025conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
weight_05conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_cifar10_mixed_Attention/cifar10acc0.8434999763965607_130.pth'
weight_1conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1MixedAttention_mixed_attention_cifar10_ep_25_val_acc0.7080.pth'
weight_025conv_mixatten_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25Mixed+ALP_cifar10_ep_85_val_acc0.8650.pth'
model_dict = torch.load(weight_AT)
model.load_state_dict(model_dict)
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        noise = torch.randn_like(images, device='cuda') * 0#l2norm=1时，对应norm=10
        #print(torch.norm(noise))#l2norm=1时，对应norm=10
        image_shape = images + noise
        #image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=1) + images
        #logits,_ = model.forward_attention(images.detach(), image_shape.detach())
        logits = model(image_shape.detach())
        #logits=model(image_shape)
        test_accs = AverageMeter()
        test_losses = AverageMeter()
        test_accs.append((logits.argmax(1) == labels).float().mean().item())
    # print(test_accs)
    # test_losses.append(loss.item())
print('\nTest accuracy ', test_accs.avg)
import PIL
import glob
import os
import argparse
import pandas as pd
import cv2
import tqdm
from copy import deepcopy
import os
import PIL
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
import random

from PIL import Image

import argparse
import tqdm
import glob
import PIL
from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, lr_scheduler,Adam
from torch.backends import cudnn
from progressbar import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms

from sklearn.metrics import accuracy_score
#from torchvision.datasets import CIFAR10
import sys
from fast_adv.models.cifar10 import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import CIFAR10

from fast_adv.models.cifar10 import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

parser.add_argument('--data', default='data/train', help='path to dataset')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weight/cifar10_ALP/', help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.2, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=10, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int,default=0, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=100, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(Image.open(image_path))#
        #b=Image.fromarray(jpeg_compression(image_path))
        #image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample

def prepareData(input_dir,img_size=32,batch_size=args.batch_size):#img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]


    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.8, test_size=0.2)
    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data': ImageSet(val_data, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders


train_loader = prepareData(args.data)['train_data']#data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(45000)))
val_loader = prepareData(args.data)['val_data']#data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                      #list(range(45000, 50000)))

m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=args.drop)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict = torch.load('./weights/AT_cifar10_clean0.879_adv.pth')
model.load_state_dict(model_dict)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.adv == 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

attacker = DDN(steps=args.steps, device=DEVICE)

max_loss = torch.log(torch.tensor(10.)).item()  # for callback
best_acc = 0
best_epoch = 0

for epoch in range(args.epochs):
    scheduler.step()
    cudnn.benchmark = True
    model.train()
    requires_grad_(m, True)
    accs = AverageMeter()
    losses = AverageMeter()
    attack_norms = AverageMeter()
    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    i = 0
    sigma = 1
    for batch_data in pbar(train_loader):
        images, labels =batch_data['image'].to(DEVICE),batch_data['label_idx'].to(DEVICE)
        #print(labels)
        #原图loss
        logits = model(images)
        loss_ori = F.cross_entropy(logits, labels)

        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(m, False)
            adv = attacker.attack(model, images, labels)
            l2_norms = (adv - images).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - images, p=2, dim=0, maxnorm=args.max_norm) + images
            attack_norms.append(mean_norm.item())
            requires_grad_(m, True)
            model.train()
            logits = model(adv.detach())
        else:
            logits = model(images)
        i = i + 1
        loss = F.cross_entropy(logits, labels)
            #logits_adv = model(adv.detach())
            #loss_adv = F.cross_entropy(logits_adv, labels)
       

       # loss = loss_ori #+ 0.5*F.mse_loss(logits_adv,logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.append((logits.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())

        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, min(losses.last_avg, max_loss))
            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)
            if args.adv is not None and epoch >= args.adv:
                CALLBACK.scalar('L2', epoch + i / length, attack_norms.last_avg)

    print('Epoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, losses.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(m, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()
    widgets = ['val :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    with torch.no_grad():
        for batch_data in pbar(val_loader):
            images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_losses.append(loss.item())

    if CALLBACK:
        CALLBACK.scalar('Val_Loss', epoch + 1, val_losses.last_avg)
        CALLBACK.scalar('Val_Acc', epoch + 1, val_accs.last_avg)

    print('Epoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, val_losses.avg, val_accs.avg))

    save_path = 'weights/best/'
    if val_accs.avg >= best_acc:  # args.adv is None and
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())
        files2remove = glob.glob(os.path.join(save_path, 'ATT_*'))
        for _i in files2remove:
            os.remove(_i)
        strsave = "ATT_cifar10_ep_%d_val_acc%.4f.pth" % (epoch, best_acc)
        torch.save(model.cpu().state_dict(),
                   os.path.join(save_path, strsave))
        model.to(DEVICE)

    if args.adv is None and val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())

    if not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(), os.path.join(args.save_folder, args.save_name + 'acc{}_{}.pth'.format(val_accs.avg,(epoch + 1))), cpu=True)

if args.adv is None:
    model.load_state_dict(best_dict)





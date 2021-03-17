import glob
import os
import sys
import argparse

import PIL
import tqdm
from progressbar import *
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
import scipy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt

from models.wsdan.wsdan import WSDAN

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
# from tianchi2021.models.wsdan.wsdan import WSDAN
# from tianchi2021.models.cifar10.wide_resnet import wide_resnet
from utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
# from tianchi2021.attacks import DDN
# from tianchi2021.utils.messageUtil import send_email

# from fast_adv.process.jpeg_compression import JpegCompression
# from fast_adv.process.jpeg_compression import jpeg_c

"""

"""

parser = argparse.ArgumentParser(description='CIFAR10 Training data augmentation')

parser.add_argument('--data', default='../data/cifar10/gridmask', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/cifar10_wsdan_gridmask/', help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.9, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=2, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, default=1, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                    help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')
parser.add_argument('--visdom-env', '--ve', type=str, default="cifar10-wsdan-gridmask",
                    help='which env visdom is running.')

parser.add_argument('--num-attentions', '--na', default=32, type=int, help='number of attention maps')
parser.add_argument('--backbone-net', '--bn', default='resnet152', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(env=args.visdom_env, port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

def load_data(csv, input_dir, img_size=224, batch_size=args.batch_size):
    all_img_paths = glob.glob(os.path.join(input_dir, './*/*.jpg'))
    jir = pd.read_csv(csv, names=['ImageId', 'TrueLabel'])

    label_maps = dict(zip(jir['TrueLabel'].tolist(), jir['ImageId'].tolist()))
    all_lables = []
    for img_path in all_img_paths:
        trueLabel = img_path.split('/')[-2]
        all_lables.append(label_maps[trueLabel])

    train_data = pd.DataFrame({'image_path': all_img_paths, 'label_idx': all_lables})
    transformer = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
                                      transforms.ToTensor(),
                                      transforms.Lambda(lambda img: img * 2.0 - 1.0),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ])
    datasets = {
        'dev_data': ImageSet(train_data, transformer),

    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        # image = self.transformer(Image.open(image_path))  # .convert('RGB'))
        b = Image.fromarray(scipy.misc.imread(image_path))
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample


train_input_dir = '../data/imagenet_all/images/'
csv_file = r'../data/imagenet_all/dev.csv'

train_loader = load_data(csv_file, train_input_dir)['dev_data']
val_loader = load_data(csv_file, train_input_dir)['dev_data']
test_loader = load_data(csv_file, train_input_dir)['dev_data']


print(len(train_loader),len(val_loader),len(test_loader))

model = WSDAN(num_classes=10, M=args.num_attentions, net=args.backbone_net, pretrained=True)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

feature_center = torch.zeros(10, args.num_attentions * model.num_features).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.adv == 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

max_loss = torch.log(torch.tensor(10.)).item()  # for callback
best_acc = 0
best_epoch = 0


# augment function         attention crop 和 attention drop
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()  # B,C,H,W

    if mode == 'crop':  # attention crop
        crop_images = []  # 用于存放crop的图像
        for batch_index in range(batches):  # 当前处理第batch_index张图片
            # attention_map attention_map[:, :1, :, :]
            atten_map = attention_map[batch_index:batch_index + 1]  # 提取第batch_index张图片的attention_map记为atten_map
            # print("atten_map ", atten_map.shape)
            # imh_cam = atten_map[0]
            # imh_cam = imh_cam.cpu().numpy()  # FloatTensor转为ndarray
            # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(imh_cam)
            # plt.show()
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()  # 处理atten_map最大的得到阀值theta_c
            else:
                theta_c = theta * atten_map.max()

            # Q&A
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c  # 大于阀值的置为1.[n,c,h,w]
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            # 取出第batch_index张图，第0个crop_maskp的为ture的位置，因为取出的只有1张图片，是1维的，第一个位置只能是0
            # 第二个0，是因为网络生成的两个attention_map，通道0用来crop。
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            # height_min肯定是大于0的数，为了选择盖住所选的区域，要比小更小
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':  # attention drop      attention_map[:, 1:, :, :]
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            # Q&A
            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()  # 在images中删除dropimages部分
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


# General loss functions
##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter()

accs = AverageMeter()
raw_metric = AverageMeter()
crop_metric = AverageMeter()
drop_metric = AverageMeter()
attack_norms = AverageMeter()

loss_crop_adv = AverageMeter()
loss_drop = AverageMeter()

valacc_final = 0

# path_jpeg = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/jpeg"

for epoch in range(args.epochs):
    scheduler.step()
    cudnn.benchmark = True
    model.train()
    requires_grad_(model, True)
    # print("len(train_loader) ", len(train_loader))
    length = len(train_loader)
    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    i = 0
    for batch_data in pbar(train_loader):
        i += 1
        X, y =batch_data['image'].to(DEVICE),batch_data['label_idx'].to(DEVICE)

        y_pred_raw, feature_matrix, attention_map = model(X)
        # Update Feature Center###########################################重点代码#####################################
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        # Q&A
        feature_center[y] += args.beta * (feature_matrix.detach() - feature_center_batch)
        ###############################################################################################################

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.3, 0.35), padding_ratio=0.1)
            # 网络生成两个attention_map，通道位置是2，通道0，用来crop，通道1用来drop
            # crop images forward
        # crop得到的图像再次送入网络得到精细预测
        # crop images forward
        # imh_cam = crop_images[0]
        # imh_cam = imh_cam.cpu().numpy()  # FloatTensor转为ndarray
        # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # 把channel那一维放到最后
        # plt.imshow(imh_cam)
        # plt.show()
        y_pred_crop, _, _ = model(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(X, attention_map[:, :1, :, :], mode='drop', theta=(0.1, 0.15))

        # drop images forward
        # drop得到图像再次送入net,去掉当前关注的特征，让net关注更多的特征。
        # print("drop ", drop_images.shape)
        # imh_cam = drop_images[0]
        # imh_cam = imh_cam.cpu().numpy()  # FloatTensor转为ndarray
        # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # 把channel那一维放到最后
        # plt.imshow(imh_cam)
        # plt.show()
        y_pred_drop, _, _ = model(drop_images)

        # loss                      # cross_entropy_loss(y_pred_crop, y) / 3. - \
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_crop, y) / 3. - \
                     cross_entropy_loss(y_pred_drop, y) / 3. *0.001 + \
                     center_loss(feature_matrix, feature_center_batch)

    print('\nEpoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, loss_container.avg, accs.avg))




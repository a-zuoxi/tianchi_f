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
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
from fast_adv.models.wsdan.wsdan import WSDAN
from fast_adv.models.cifar10.wide_resnet import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from fast_adv.utils.messageUtil import send_email

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
parser.add_argument('--backbone-net', '--bn', default='wide_resnet', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
# http://119.45.5.175
CALLBACK = VisdomLogger(env=args.visdom_env, port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)



def load_data_for_defense(input_dir, img_size=32, batch_size=args.batch_size):
    all_img_paths = glob.glob(os.path.join(input_dir, './*/*.jpg'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path': all_img_paths, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


def prepareData(input_dir, img_size=32, batch_size=args.batch_size):  # img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.4, test_size=0.1)
    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data': ImageSet(val_data, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
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


train_loader = prepareData(args.data+'/train')[
    'train_data']  # data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(45000)))
val_loader = prepareData(args.data+'/train')[
    'val_data']  # data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
test_loader = load_data_for_defense(args.data+'/test')


print(len(train_loader), len(val_loader), len(test_loader))
m = WSDAN(num_classes=10, M=args.num_attentions, net=args.backbone_net, pretrained=True)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
cifar10_WSDAN_best = './weights/cifar10_WSDAN_best/cifar10_0.87_low.pth'
model_dict = torch.load(cifar10_WSDAN_best)
model.load_state_dict(model_dict)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

feature_center = torch.zeros(10, args.num_attentions * m.num_features).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.adv == 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

attacker = DDN(steps=args.steps, device=DEVICE)

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
        # print("X.shape", X.shape, "y.shape",y.shape)

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

        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(m, False)
            adv = attacker.attack(model, X, y)
            l2_norms = (adv - X).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - X, p=2, dim=0, maxnorm=args.max_norm) + X
            attack_norms.append(mean_norm.item())
            requires_grad_(m, True)
            model.train()
            y_pred_raw_adv, feature_matrix_adv, attention_map_adv = model(adv.detach())

            # Update Feature Center###########################################重点代码#####################################
            feature_center_batch_adv = F.normalize(feature_center[y], dim=-1)
            # Q&A
            feature_center[y] += args.beta * (feature_matrix_adv.detach() - feature_center_batch_adv)
            ###############################################################################################################

            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images_adv = batch_augment(adv, attention_map_adv[:, :1, :, :], mode='crop', theta=(0.3, 0.35),
                                            padding_ratio=0.1)
                # 网络生成两个attention_map，通道位置是2，通道0，用来crop，通道1用来drop
                # crop images forward
            # crop得到的图像再次送入网络得到精细预测
            # crop images forward
            y_pred_crop_adv, _, _ = model(crop_images_adv)

            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_images_adv = batch_augment(adv, attention_map_adv[:, :1, :, :], mode='drop', theta=(0.1, 0.15))

            # drop images forward
            # drop得到图像再次送入net,去掉当前关注的特征，让net关注更多的特征。
            y_pred_drop_adv, _, _ = model(drop_images_adv)

            # loss cross_entropy_loss(y_pred_drop_adv, y) / 3. *0.001 + \
            batch_loss += cross_entropy_loss(y_pred_raw_adv, y) / 3. + \
                         cross_entropy_loss(y_pred_crop_adv, y) / 3. - \
                          cross_entropy_loss(y_pred_drop_adv, y) / 3. * 0.001 +\
                            center_loss(feature_matrix_adv, feature_center_batch_adv)

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            accs.append((y_pred_raw.argmax(1) == y).float().mean().item())
            loss_container.append(batch_loss.item())

            # loss_crop_adv.append(cross_entropy_loss(y_pred_crop_adv, y))
            loss_drop.append(cross_entropy_loss(y_pred_drop, y))
            # epoch_raw_acc = raw_metric(y_pred_raw, y)
            # epoch_crop_acc = crop_metric(y_pred_crop, y)
            # epoch_drop_acc = drop_metric(y_pred_drop, y)
        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, loss_container.last_avg)
            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)

            # CALLBACK.scalar('Tr_Loss_crop_adv', epoch + i / length, loss_crop_adv.last_avg)
            CALLBACK.scalar('Tr_Loss_drop', epoch + i / length, loss_drop.last_avg)

            if args.adv is not None and epoch >= args.adv:
                pass
                # CALLBACK.scalar('L2', epoch + i / length, attack_norms.last_avg)

    print('\nEpoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, loss_container.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(model, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()
    val_accs_raw = AverageMeter()
    val_accs_crop = AverageMeter()
    widgets = ['val :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    with torch.no_grad():
        for batch_data in pbar(val_loader):
            X, y = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
            # Raw Image
            y_pred_raw, _, attention_map = model(X)

            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = model(crop_images)

            # Final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            val_losses.append(batch_loss.item())

            # int("y_pred ", y_pred.shape, "y", y.shape)
            # metrics: top-1,5 error
            val_accs.append((y_pred.argmax(1) == y).float().mean().item())
            val_accs_raw.append((y_pred_raw.argmax(1) == y).float().mean().item())
            val_accs_crop.append((y_pred_crop.argmax(1) == y).float().mean().item())
    if CALLBACK:
        CALLBACK.scalar('Val_Loss', epoch + 1, val_losses.last_avg)
        CALLBACK.scalar('Val_Acc', epoch + 1, val_accs.last_avg)
        CALLBACK.scalar('Val_Acc_Raw', epoch + 1, val_accs_raw.last_avg)
        CALLBACK.scalar('Val_Acc_Crop', epoch + 1, val_accs_crop.last_avg)

    print('\nEpoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}, Accs_raw: {:.4f}, Accs_crop: {:.4f}'.format(epoch, val_losses.avg, val_accs.avg, val_accs_raw.avg, val_accs_crop.avg))

    valacc_final = val_accs.avg


    if epoch % 5 == 0 :
        save_checkpoint(model.state_dict(), os.path.join(args.save_folder,
                                                         args.save_name + '_valacc' + str(valacc_final) + '_' + str(
                                                             epoch) + '.pth'), cpu=True)
print('\nSaving model...')
save_checkpoint(model.state_dict(), os.path.join(args.save_folder, args.save_name + '_valacc' + str(valacc_final) + '.pth'), cpu=True)
CALLBACK.save([args.visdom_env])
send_email("cifar10_wsdan 数据增强实验完成"+"\n"+args.visdom_env+"\nval acc = "+str(valacc_final))
import glob
import os
import sys
import argparse
import tqdm
from copy import deepcopy

import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)

from fast_adv.models.wsdan.wsdan import WSDAN
from fast_adv.models.cifar10.wide_resnet import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

parser.add_argument('--data', default='data/cifar10', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/cifar10_wsgan_smooth_add2loss_meanAttention/',
                    help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=16, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.9, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=2, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, default=0, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                    help='For visualization, which port visdom is running.')

parser.add_argument('--visdom-env', '--ve', type=str, default="cifar10_wsgan_smooth_add2loss_meanAttention",
                    help='which env visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

parser.add_argument('--num-attentions', '--na', default=32, type=int, help='number of attention maps')
parser.add_argument('--backbone-net', '--bn', default='wide_resnet', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')
parser.add_argument('--noise_sd', default=0.25, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")

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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),  # ?????????p????????????
    transforms.ColorJitter(brightness=0.126, saturation=0.5),  # ????????????????????????????????????
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ?????????
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ?????????
])

train_set = data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(30000)))
val_set = data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                      list(range(48000, 50000)))
test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)

print(len(train_set), len(val_set), len(test_set))
print(len(train_loader), len(val_loader), len(test_loader))

m = WSDAN(num_classes=10, M=args.num_attentions, net=args.backbone_net, pretrained=True)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
# model_file = './weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
# if len(model_file) > 0:
#     pretext_model = torch.load(model_file)
#     model_dict = model.state_dict()
#     model_new_dict = {}
#     for k, v in pretext_model.items():
#         k = k.replace("model.conv1.weight", "model.features.0.weight")
#         k = k.replace("model.block1","model.features.1")
#         k = k.replace("model.block2", "model.features.2")
#         k = k.replace("model.block3", "model.features.3")
#         k = k.replace("model.bn1", "model.features.4")
#         if k == "model.fc.weight":
#             continue
#         model_new_dict[k] = v
#     state_dict = {k: v for k, v in model_new_dict.items() if k in model_dict.keys()}
#     print(len(pretext_model.keys()))
#     print(len(model_dict.keys()))
#     print(len(state_dict.keys()))
#     model_dict.update(state_dict)
#     model.load_state_dict(model_dict)
model_file = './weights/cifar10_WSDAN_smooth3/cifar10_valacc0.865999984741211_10.pth'
weight_wsdan_smooth = './weights/cifar10_WSDAN_smooth3/cifar10_valacc0.8644999772310257.pth'
weight_wsdan_smooth_add2loss = './weights/cifar10_wsgan_smooth_add2loss/cifar10_valacc0.8439999759197235_40.pth'
model_dict = torch.load(weight_wsdan_smooth_add2loss)
model.load_state_dict(model_dict)
print("load wight:", model_file)
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


# augment function         attention crop ??? attention drop
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()  # B,C,H,W

    if mode == 'crop':  # attention crop
        crop_images = []  # ????????????crop?????????
        for batch_index in range(batches):  # ???????????????batch_index?????????
            # attention_map attention_map[:, :1, :, :]
            atten_map = attention_map[batch_index:batch_index + 1]  # ?????????batch_index????????????attention_map??????atten_map
            # print("atten_map ", atten_map.shape)
            # imh_cam = atten_map[0]
            # imh_cam = imh_cam.cpu().numpy()  # FloatTensor??????ndarray
            # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # ???channel?????????????????????
            # plt.imshow(imh_cam)
            # plt.show()
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()  # ??????atten_map?????????????????????theta_c
            else:
                theta_c = theta * atten_map.max()

            # Q&A
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c  # ?????????????????????1.[n,c,h,w]
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            # ?????????batch_index????????????0???crop_maskp??????ture?????????????????????????????????1???????????????1?????????????????????????????????0
            # ?????????0?????????????????????????????????attention_map?????????0??????crop???
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            # height_min???????????????0????????????????????????????????????????????????????????????
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
        drop_images = images * drop_masks.float()  # ???images?????????dropimages??????
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
loss_raw = AverageMeter()
loss_crop = AverageMeter()
loss_drop = AverageMeter()
loss_raw_adv = AverageMeter()
loss_crop_adv = AverageMeter()
loss_drop_adv = AverageMeter()
loss_logist_y = AverageMeter()
loss_logist_att = AverageMeter()

accs = AverageMeter()
raw_metric = AverageMeter()
crop_metric = AverageMeter()
drop_metric = AverageMeter()
attack_norms = AverageMeter()
valacc_final = 0
# raw_metric = TopKAccuracyMetric(topk=(1, 5))
# crop_metric = TopKAccuracyMetric(topk=(1, 5))
# drop_metric = TopKAccuracyMetric(topk=(1, 5))

for epoch in range(args.epochs):
    scheduler.step()
    cudnn.benchmark = True
    model.train()
    requires_grad_(model, True)
    # print("len(train_loader) ", len(train_loader))
    length = len(train_loader)
    for i, (X, y) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # print(i, X.shape, y.shape)

        if args.adv is not None and epoch >= args.adv:
            pass

        ##################################
        # Raw Image
        ##################################
        # raw images forward
        # print(X)
        # imh_cam = X[0]
        # imh_cam = imh_cam.cpu().numpy()  # FloatTensor??????ndarray
        # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # ???channel?????????????????????
        # plt.imshow(imh_cam)
        # plt.show()
        y_pred_raw, feature_matrix, attention_map = model(X)
        # print("X: ", X.shape, "y_pred_raw ", y_pred_raw.shape, "feature_matrix", feature_matrix.shape, "attention_map", attention_map.shape)
        # Update Feature Center###########################################????????????#####################################
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        # Q&A
        feature_center[y] += args.beta * (feature_matrix.detach() - feature_center_batch)
        ###############################################################################################################

        noise = torch.randn_like(X, device='cuda') * args.noise_sd
        image_shape = X + noise
        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(image_shape, attention_map[:, :1, :, :], mode='crop', theta=(0.3, 0.35),
                                        padding_ratio=0.1)
            # ??????????????????attention_map??????????????????2?????????0?????????crop?????????1??????drop
            # crop images forward
        # crop???????????????????????????????????????????????????
        # crop images forward
        # imh_cam = crop_images[0]
        # imh_cam = imh_cam.cpu().numpy()  # FloatTensor??????ndarray
        # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # ???channel?????????????????????
        # plt.imshow(imh_cam)
        # plt.show()
        y_pred_crop, _, _ = model(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(X, attention_map[:, :1, :, :], mode='drop', theta=(0.1, 0.15))

        # drop images forward
        # drop????????????????????????net,?????????????????????????????????net????????????????????????
        # print("drop ", drop_images.shape)
        # imh_cam = drop_images[0]
        # imh_cam = imh_cam.cpu().numpy()  # FloatTensor??????ndarray
        # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # ???channel?????????????????????
        # plt.imshow(imh_cam)
        # plt.show()
        y_pred_drop, _, _ = model(drop_images)
        # print(X.detach().shape, image_shape.detach().shape)
        logits = model.forward_attention(X.detach(), image_shape.detach())
        loss_attention = F.cross_entropy(logits, y)
        # loss
        batch_loss = loss_attention / 3. + \
                     cross_entropy_loss(y_pred_crop, y) / 3. - \
                     cross_entropy_loss(y_pred_drop, y) / 3. * 0.0001 + \
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

            # Update Feature Center###########################################????????????#####################################
            feature_center_batch_adv = F.normalize(feature_center[y], dim=-1)
            # Q&A
            feature_center[y] += args.beta * (feature_matrix_adv.detach() - feature_center_batch_adv)
            ###############################################################################################################

            # noise = torch.randn_like(adv, device='cuda') * args.noise_sd
            # image_adv_shape = adv + noise
            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images_adv = batch_augment(adv, attention_map_adv[:, :1, :, :], mode='crop', theta=(0.3, 0.35),
                                                padding_ratio=0.1)
                # ??????????????????attention_map??????????????????2?????????0?????????crop?????????1??????drop
                # crop images forward
            # crop???????????????????????????????????????????????????
            # crop images forward
            y_pred_crop_adv, _, _ = model(crop_images_adv)

            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_images_adv = batch_augment(adv, attention_map_adv[:, :1, :, :], mode='drop', theta=(0.1, 0.15))

            # drop images forward
            # drop????????????????????????net,?????????????????????????????????net????????????????????????
            y_pred_drop_adv, _, _ = model(drop_images_adv)

            # logits_adv = model.forward_attention(adv.detach(), image_adv_shape.detach())
            # loss_attention = F.cross_entropy(logits_adv, y)
            # loss
            batch_loss += cross_entropy_loss(y_pred_raw_adv, y) / 3 + \
                          cross_entropy_loss(y_pred_crop_adv, y) / 3. - \
                          cross_entropy_loss(y_pred_drop_adv, y) / 3. * 0.0001 + \
                          center_loss(feature_matrix_adv, feature_center_batch_adv)

        # add two loss :??????????????????????????????????????????????????????????????????????????????
        loss_add_y = F.mse_loss(y_pred_raw_adv, y_pred_raw)
        loss_add_att = F.mse_loss(attention_map_adv, attention_map)
        batch_loss += 0.2 * loss_add_y + 0.1 * loss_add_att
        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            accs.append((y_pred_raw.argmax(1) == y).float().mean().item())
            loss_container.append(batch_loss.item())
            loss_raw.append(cross_entropy_loss(y_pred_raw, y))
            loss_crop.append(cross_entropy_loss(y_pred_crop, y))
            loss_drop.append(cross_entropy_loss(y_pred_drop, y))
            loss_raw_adv.append(cross_entropy_loss(y_pred_raw_adv, y))
            loss_crop_adv.append(cross_entropy_loss(y_pred_crop_adv, y))
            loss_drop_adv.append(cross_entropy_loss(y_pred_drop_adv, y))

            loss_logist_y.append(loss_add_y)
            loss_logist_att.append(loss_add_att)
            # epoch_raw_acc = raw_metric(y_pred_raw, y)
            # epoch_crop_acc = crop_metric(y_pred_crop, y)
            # epoch_drop_acc = drop_metric(y_pred_drop, y)
        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, loss_container.last_avg)
            CALLBACK.scalar('Tr_Loss_raw', epoch + i / length, loss_raw.last_avg)
            CALLBACK.scalar('Tr_Loss_crop', epoch + i / length, loss_crop.last_avg)
            CALLBACK.scalar('Tr_Loss_drop', epoch + i / length, loss_drop.last_avg)
            CALLBACK.scalar('Tr_Loss_raw_adv', epoch + i / length, loss_raw_adv.last_avg)
            CALLBACK.scalar('Tr_Loss_crop_adv', epoch + i / length, loss_crop_adv.last_avg)
            CALLBACK.scalar('Tr_Loss_drop_adv', epoch + i / length, loss_drop_adv.last_avg)

            CALLBACK.scalar('Tr_Loss_logist_y', epoch + i / length, loss_logist_y.last_avg)
            CALLBACK.scalar('Tr_Loss_logist_att', epoch + i / length, loss_logist_att.last_avg)

            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)
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
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
            X, y = X.to(DEVICE), y.to(DEVICE)

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

    print('\nEpoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}, Accs_raw: {:.4f}, Accs_crop: {:.4f}'.format(epoch,
                                                                                                             val_losses.avg,
                                                                                                             val_accs.avg,
                                                                                                             val_accs_raw.avg,
                                                                                                             val_accs_crop.avg))

    valacc_final = val_accs.avg

    if epoch % 10 == 0 or batch_loss.item() < 0.01:
        save_checkpoint(model.state_dict(), os.path.join(args.save_folder,
                                                         args.save_name + '_valacc' + str(valacc_final) + '_' + str(
                                                             epoch) + '.pth'), cpu=True)
print('\nSaving model...')
save_checkpoint(model.state_dict(),
                os.path.join(args.save_folder, args.save_name + '_valacc' + str(valacc_final) + '.pth'), cpu=True)
CALLBACK.save([args.visdom_env])

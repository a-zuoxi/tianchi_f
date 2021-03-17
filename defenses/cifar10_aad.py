import glob
import math
import os
import sys
import argparse

import torchvision
import tqdm
from copy import deepcopy
import cv2

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


BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)


from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

parser.add_argument('--data', default='data/cifar10', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/cifar10_aad4/', help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of total epochs to run')
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
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')



args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
# http://119.45.5.175
CALLBACK = VisdomLogger(server="http://121.248.50.95", env="cifar10_aad_lossC:lossR_1:0.05", port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),  # 依概率p水平翻转
    transforms.ColorJitter(brightness=0.126, saturation=0.5),  # 修改亮度、对比度和饱和度
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(30000)))
val_set = data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                      list(range(48000, 50000)))
test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)

m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=args.drop)
model =NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range

weight_AT='./weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
# model_file='/home/frankfeng/projects/researchData/AI_security/code/PLP/fast_adv/defenses/weights/cifar10_base/cifar10_valacc0.8339999794960022.pth'
model_dict = torch.load(weight_AT)
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

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # print("class_idx,",class_idx)
    for i in range(bz):
        for idx in class_idx:

            cam = weight_softmax[idx[i]].dot(feature_conv[i].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
            cam_img = np.uint8(255 * cam_img)
            print(cam_img.shape)
            # heatmap = cv2.applyColorMap(cv2.resize(cam_img, (w, h)), cv2.COLORMAP_JET)
            # heatmap = cv2.resize(heatmap, size_upsample)
            # heatmap = heatmap.transpose(2, 0, 1)
            # output_cam.append(heatmap)
            heatmap = cv2.resize(cam_img, size_upsample)
            heatmap = heatmap[np.newaxis, :]
            heatmap = np.vstack((heatmap, heatmap, heatmap))
            # print("output_cam ", heatmap.shape)
            output_cam.append(heatmap)
    return output_cam


from torchvision import utils as vutils


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

def getXm(images, cams):
    cams = cams/255
    cams = torch.sign(F.relu(cams - 0.4))

    xx = 1 - cams

    # Xm = torch.einsum('bijk,bikm->bijm', images, xx)

    Xm = xx * images
    return Xm

valacc_final = 0

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for epoch in range(args.epochs):
    scheduler.step()
    cudnn.benchmark = True
    model.train()
    requires_grad_(m, True)
    accs = AverageMeter()
    losses = AverageMeter()
    losses_r = AverageMeter()
    losses_c = AverageMeter()
    losses_p = AverageMeter()
    attack_norms = AverageMeter()

    length = len(train_loader)

    loss_r = loss_r1 = loss_r2 = 0
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #原图loss
        # print("imaggges : ", images.shape)
        logits = model(images)
        loss_c = F.cross_entropy(logits, labels)
        # print("loss_c", loss_c.item())
        h_x = F.softmax(logits, dim=1).data.squeeze()

        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        pre_class = logits.argmax(1).cpu().numpy()
        # print('label:', target_class, 'pre:', pre_class)
        params = list(model.parameters())
        # print(params)
        weight_softmax = np.squeeze(params[80].data.detach().cpu().numpy())
        features_blobs = model.feature_map(images).detach().cpu().numpy()
        # generate class activation mapping for the top1 prediction
        # print("images",images.shape, "features_blobs", features_blobs.shape, "weight_softmax",weight_softmax.shape)
        CAMs = returnCAM(features_blobs, weight_softmax, [pre_class])
        # print("CAMs", len(CAMs))
        cams = torch.Tensor(CAMs).to(DEVICE)
        Xm = getXm(images,cams)
        print(Xm.shape)
        for b in range(Xm.shape[0]):
            print(Xm.min(),images.min())
            img = Xm[b]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
            img = img.cpu().numpy()  # FloatTensor转为ndarray
            img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
            # 显示图片
            # img_org = images[b]
            # img_org = img_org.cpu().numpy()  # FloatTensor转为ndarray
            # img_org = np.transpose(img_org, (1, 2, 0))  # 把channel那一维放到最后
            # imh_cam = cams[b]
            # imh_cam = imh_cam.cpu().numpy()  # FloatTensor转为ndarray
            # imh_cam = np.transpose(imh_cam, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(imh_cam)
            # plt.show()
            # plt.imshow(img_org)
            # plt.show()
            # plt.imshow(img)
            # plt.show()
            # np.savetxt(r'/home/frankfeng/桌面/img0.txt', img[:, :, 0])
            # np.savetxt(r'/home/frankfeng/桌面/img1.txt', img[:, :, 1])
            # np.savetxt(r'/home/frankfeng/桌面/img2.txt', img[:, :, 2])
            # np.savetxt(r'/home/frankfeng/桌面/img_org0.txt', img_org[:, :, 0])
            # np.savetxt(r'/home/frankfeng/桌面/img_org1.txt', img_org[:, :, 1])
            # np.savetxt(r'/home/frankfeng/桌面/img_org2.txt', img_org[:, :, 2])
            if b >5 :
                break
        # logits_m = model(Xm)

        # loss_r1 = -F.mse_loss(logits, logits_m)
        # loss_r1 = torch.exp(loss_r1)

        # test for print cam and image
        # for bz_i in range(10):
        #
        #     CAM_PATH_o = './cams/0_7_CAM_aad_' + str(bz_i) + '.jpg'
        #     CAMs_np = np.array(CAMs)
        #     imagesnp = images.cpu().numpy()*255
        #     result = CAMs_np[bz_i] * 0.8 + imagesnp[bz_i] * 0.5
        #     result = result.transpose(1,2,0)
        #     print("s",CAMs_np[bz_i],"\n\n\n\n",imagesnp[bz_i])
        #
        #     cv2.imwrite(CAM_PATH_o, result)


        cams_adv = cams
        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(m, False)
            # print("images, ",images)
            adv = attacker.attack(model, images, labels)
            l2_norms = (adv - images).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - images, p=2, dim=0, maxnorm=args.max_norm) + images
            attack_norms.append(mean_norm.item())
            requires_grad_(m, True)
            model.train()
            logits_adv = model(adv.detach())

            h_x = F.softmax(logits_adv, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            pre_class = logits_adv.argmax(1).cpu().numpy()
            # print('label:', target_class, 'pre:', pre_class)
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[80].data.detach().cpu().numpy())
            features_blobs = model.feature_map(adv.detach()).detach().cpu().numpy()
            # generate class activation mapping for the top1 prediction
            CAMs_adv = returnCAM(features_blobs, weight_softmax, [pre_class])
            cams_adv = torch.Tensor(CAMs_adv).to(DEVICE)
            adv = attacker.attack(model, Xm, labels)
            l2_norms = (adv - Xm).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - Xm, p=2, dim=0, maxnorm=args.max_norm) + Xm
            attack_norms.append(mean_norm.item())
            requires_grad_(m, True)
            model.train()
            logits_adv_m = model(adv.detach())

            # print("logits_adv,", logits_adv.max(),logits_adv.min(), "logits_adv_m", logits_adv_m.max(),logits_adv_m.min())
            # bz, label_size = logits.size()
            loss_r2 = -F.mse_loss(logits_adv, logits_adv_m)
            loss_r2 = torch.exp(loss_r2)
        loss_r = loss_r2
        bz,c,w,h = cams.shape
        # loss_p = F.mse_loss(cams/255, cams_adv/255)
        total_loss = loss_c+ 0.5 * loss_r*0.1

        #loss = loss+ loss_adv + 0.5*F.mse_loss(logits_adv,logits)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        accs.append((logits.argmax(1) == labels).float().mean().item())
        losses.append(total_loss.item())
        losses_c.append(loss_c.item())
        losses_r.append(loss_r.item())
        # losses_p.append(loss_p.item())

        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, losses.last_avg)
            CALLBACK.scalar('Tr_Loss_c', epoch + i / length, losses_c.last_avg)
            CALLBACK.scalar('Tr_Loss_r', epoch + i / length, losses_r.last_avg)
            # CALLBACK.scalar('Tr_Loss_p', epoch + i / length, losses_p.last_avg)
            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)
            if args.adv is not None and epoch >= args.adv:
                CALLBACK.scalar('L2', epoch + i / length, attack_norms.last_avg)
    print('Epoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, losses.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(m, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
        files2remove = glob.glob(os.path.join(save_path, 'AT_*'))
        for _i in files2remove:
            os.remove(_i)
        strsave = "AT_cifar10_ep_%d_val_acc%.4f.pth" % (epoch, best_acc)
        torch.save(model.cpu().state_dict(),
                   os.path.join(save_path, strsave))
        model.to(DEVICE)

    if args.adv is None and val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())

    if not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(), os.path.join(args.save_folder, args.save_name + '3loss_acc{}_{}.pth'.format(val_accs.avg,(epoch + 1))), cpu=True)
    valacc_final = val_accs.avg

if args.adv is None:
    model.load_state_dict(best_dict)

test_accs = AverageMeter()
test_losses = AverageMeter()
# test_accs_m = AverageMeter()
# test_losses_m = AverageMeter()

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # h_x = F.softmax(logits, dim=1).data.squeeze()
        #
        # probs, idx = h_x.sort(0, True)
        # probs = probs.cpu().numpy()
        # idx = idx.cpu().numpy()
        # pre_class = logits.argmax(1).cpu().numpy()
        # # print('label:', target_class, 'pre:', pre_class)
        # params = list(model.parameters())
        # weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())
        # features_blobs = model.feature_map(images).detach().cpu().numpy()
        # # generate class activation mapping for the top1 prediction
        # # print("images",images.shape, "features_blobs", features_blobs.shape, "weight_softmax",weight_softmax.shape)
        # CAMs = returnCAM(features_blobs, weight_softmax, [pre_class])
        # cams = torch.Tensor(CAMs).to(DEVICE)
        # Xm = getXm(images, cams)
        # logits_m = model(Xm)
        # test_accs_m.append((logits_m.argmax(1) == labels).float().mean().item())
        test_accs.append((logits.argmax(1) == labels).float().mean().item())
        test_losses.append(loss.item())

if args.adv is not None:
    print('\nTest accuracy with final model: {:.4f} with loss: {:.4f}'.format(test_accs.avg, test_losses.avg))
    # print('\nTest accuracy_m with final model: {:.4f} '.format(test_accs_m.avg))
else:
    print('\nTest accuracy with model from epoch {}: {:.4f} with loss: {:.4f}'.format(best_epoch,
                                                                                      test_accs.avg, test_losses.avg))

print('\nSaving model...')
save_checkpoint(model.state_dict(), os.path.join(args.save_folder, args.save_name + 'valacc_' + str(valacc_final) + '.pth'), cpu=True)

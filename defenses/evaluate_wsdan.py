import os
import tqdm
import random

import torch
import torch.nn.functional as F
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import CIFAR10

from fast_adv.models.wsdan.wsdan import WSDAN
# augment function         attention crop 和 attention drop
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()  # B,C,H,W

    if mode == 'crop':  # attention crop
        crop_images = []  # 用于存放crop的图像
        for batch_index in range(batches):  # 当前处理第batch_index张图片
            # attention_map attention_map[:, :1, :, :]
            atten_map = attention_map[batch_index:batch_index + 1]  # 提取第batch_index张图片的attention_map记为atten_map
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


ToPILImage = transforms.ToPILImage()
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)


DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() ) else 'cpu')
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

input='data/cifar10'
#path='/media/wanghao/000F5F8400087C68/CYJ-5-29/DDN/fast_adv/attacks/DeepFool'
#test_set=CIFAR10(input, train=False, transform=test_transform, download=True)
test_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(0,30000)))
train_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(48000, 50000)))

test_loader = data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

model = WSDAN(num_classes=10, M=32, net='vgg19', pretrained=True)
model = model.to(DEVICE)
# model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
weight_wsdan = '/home/frankfeng/projects/researchData/AI_security/code/PLP/fast_adv/defenses/weights/cifar10_WSDAN/cifar10_valacc0.8589999794960022.pth'
model_dict = torch.load(weight_wsdan)
model.load_state_dict(model_dict)
model.eval()
savepath = "/home/frankfeng/datasets/wsdan-te1/"
# os.makedirs(savepath)
batch_size = 128
# os.makedirs(savepath)

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


with torch.no_grad():
    for i, (X, y) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # WS-DAN 骨干网返回粗预测结果
        y_pred_raw, _, attention_maps = model(X)
        # print("attention_maps: ", attention_maps)
        # if i == 0:
        #     break
        # Augmentation with crop_mask
        # attention crop 返回crop 精细预测结果，粗细结合，得到最佳。
        ############################################################重点代码###################################################
        crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

        y_pred_crop, _, _ = model(crop_image)
        y_pred = (y_pred_raw + y_pred_crop) / 2.
        ######################################################################################################################

        # reshape attention maps
        attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
        attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

        # get heat attention maps  #生成热力图
        heat_attention_maps = generate_heatmap(attention_maps)

        # raw_image, heat_attention, raw_attention
        raw_image = X.cpu() * image_std + image_mean
        heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5  # 热力图合并到原图上
        raw_attention_image = raw_image * attention_maps  # attention_maps关注的地方

        for batch_idx in range(X.size(0)):
            rimg = ToPILImage(raw_image[batch_idx])
            raimg = ToPILImage(raw_attention_image[batch_idx])
            haimg = ToPILImage(heat_attention_image[batch_idx])
            rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * batch_size + batch_idx)))
            raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * batch_size + batch_idx)))
            haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * batch_size + batch_idx)))

        # Top K
        # epoch_raw_acc = raw_accuracy(y_pred_raw, y)
        # epoch_ref_acc = ref_accuracy(y_pred, y)

        # end of this batch
        # batch_info = 'Val Acc: Raw ({:.2f}, {:.2f}), Refine ({:.2f}, {:.2f})'.format(
        #     epoch_raw_acc[0], epoch_raw_acc[1], epoch_ref_acc[0], epoch_ref_acc[1])
        # pbar.update()
        # pbar.set_postfix_str(batch_info)
    #
    # pbar.close()
#

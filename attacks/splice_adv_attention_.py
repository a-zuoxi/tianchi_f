import glob
import os
import argparse
import random
import socket
import cv2

import imageio
import scipy
import tqdm
import numpy as np
import torch
from PIL import Image
import logging


from utils.utils import load_model

logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')

from progressbar import *


from torchvision import transforms
import sys


sys.path.append("..")
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from utils.messageUtil import send_email


import warnings
import pandas as pd
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument("--model_name", type=str, default='resnet152')
parser.add_argument("--threshold", type=int, default=60)
parser.add_argument("--clean_img_dir", type=str, default='../data/imagenet_all/images/')
parser.add_argument("--csv_file", type=str, default='../data/imagenet_all/dev.csv')
parser.add_argument("--adv_out_dir", type=str, default='./tianchi2021_adv/m_di2_fgsm_efficientnetb8_advprop_l210_step50/')
parser.add_argument("--adv_attention_out_dir", type=str, default='./mdi2fgsm_dsn161jpeg_rn152jpegl210_step50_attention_')

parser.add_argument("--model_name", type=str, default='resnet152')

parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--imgsize', default=32, type=int, help='pic size')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch_size', '-b', default=2, type=int, help='mini-batch size')
parser.add_argument('--steps', default=50, type=int, help='iteration steps')
parser.add_argument('--max_norm', default=10, type=float, help='Linf limit')
parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')

parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--fgsm", type=bool, default=False)
parser.add_argument("--deepfool", type=bool, default=False)
parser.add_argument("--pgd", type=bool, default=False)
parser.add_argument("--ddn", type=bool, default=False)
parser.add_argument("--randomMaskAttack", type=bool, default=False)
parser.add_argument("--cropMaskAttack", type=bool, default=False)
parser.add_argument("--clipAttack", type=bool, default=False)
parser.add_argument("--m_di2_fgsm", type=bool, default=True)


args = parser.parse_args()
print(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')


def load_data(csv, input_dir, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
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
        b = Image.fromarray(imageio.imread(image_path))
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample


data_loader = load_data(args.csv_file, args.adv_out_dir)['dev_data']
print("data_loader :", len(data_loader))

model = load_model(args.model_name).to(DEVICE).eval()
epoch = 0
accs = 0
accs_advs = 0


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256

    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    output_cam2 = []
    for idx in class_idx:
        one = np.ones_like(weight_softmax[idx])

        cam2 = one.dot(feature_conv.reshape((nc, h * w)))
        cam2 = cam2.reshape(h, w)
        cam_img2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())  # normalize
        cam_img2 = np.uint8(255 * cam_img2)
        output_cam2.append(cv2.resize(cam_img2, size_upsample))

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)

        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam, output_cam2


params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())

for batch_data in tqdm.tqdm(data_loader):
    images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
        'filename']
    # predictions = model(images).argmax(1)
    #
    # accuracy = (predictions == labels).float().mean()
    # accs += accuracy.item()
    #
    # if epoch < 5 or (epoch + 1) % 50 == 0:
    #     print(epoch, accs / (epoch + 1))


    if args.m_di2_fgsm:
        for i in range(images.shape[0].item()):
            pre_img = images[i].unsqueeze(0)
            features_blobs = model.feature_map(pre_img).cpu().detach().numpy()
            CAMs, Feature = returnCAM(features_blobs, weight_softmax, [labels[i]])
            clean_path1 = args.clean_img_dir + filenames[i]
            img_clean = cv2.imread(clean_path1)
            height, width, _ = img_clean.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

            feature_outside = cv2.resize(Feature[0], (width, height)) <= args.threshold
            feature_outside = np.expand_dims(feature_outside, -1).repeat(3, axis=-1)

            result2 = img_clean * feature_outside

            feature_center = cv2.resize(Feature[0], (width, height)) > args.threshold
            feature_center = np.expand_dims(feature_center, -1).repeat(3, axis=-1)

            clean_path2 = args.adv_out_dir + filenames[i]
            img_adv = cv2.imread(clean_path2)
            result_adv = img_adv * feature_center

            result = result_adv + result2
            out_path = args.adv_attention_out_dir + str(args.threshold)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            Feature_PATH = os.path.join(out_path, filenames[i])
            cv2.imwrite(Feature_PATH, result)

hostname = socket.gethostname()
send_email("对抗样本attention拼接完成", title=hostname+": 对抗样本拼接完毕")

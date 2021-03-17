import glob
import os
import argparse

import scipy
import tqdm
import numpy as np
from copy import deepcopy
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import CIFAR10
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.utils.messageUtil import send_email

from fast_adv.attacks import DDN

from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
import foolbox
import warnings

warnings.filterwarnings("ignore")
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack, \
    SpatialAttack, CarliniWagnerL2Attack

import logging
logging.basicConfig(level = logging.ERROR,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loggger = logging.getLogger()

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--imgsize', default=32, type=int, help='pic size')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--ddn", type=int, default=None)
parser.add_argument("--fgsm", type=bool, default=True)
parser.add_argument("--deepfool", type=bool, default=True)
parser.add_argument("--pgd", type=bool, default=True)
parser.add_argument("--data-loader", type=str, default='train')
# parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')

args = parser.parse_args()
print(args)
path = "/home/frankfeng/projects/attack"
input = '../defenses/data/cifar10'

attackers = {'FGSM': FGSM,
             'C&W': CarliniWagnerL2Attack,  # 距离无限制
             'DeepFoolAttack': DeepFoolL2Attack,  # 源码上没有限制
             'PGD': PGD,  # clip——epsilon=0.3
             'DDN': DDN,
             'LocalSearchAttack': LocalSearchAttack,
             'GaussianBlurAttack': GaussianBlurAttack,
             'BinarizationRefinementAttack': BinarizationRefinementAttack,
             'ContrastReductionAttack': ContrastReductionAttack,
             'SaltAndPepperNoiseAttack': SaltAndPepperNoiseAttack,
             'SpatialAttack': SpatialAttack}
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
DEVICE = torch.device('cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
])


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(Image.open(image_path).convert('RGB'))#
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

def load_data_for_defense(input_data, img_size=32,batch_size=32):

    all_imgs = []
    all_labels = []
    #for input_dir in jpg_data:
    for input_dir in input_data:
        one_imgs = glob.glob(input_dir)  # (os.path.join(input_dir, './*/*.jpg'))
        one_labels = [int(img_path.split('/')[-2]) for img_path in one_imgs]
        all_imgs.extend(one_imgs)
        all_labels.extend(one_labels)
    print(len(all_labels))
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    # print(all_labels)
    transformer = transforms.Compose([
        # transforms.Resize((img_size, img_size), interpolation=PIL.Image.BILINEAR),
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
    #
    # path=os.path.join(input_dir, '/*/*.jpg')
    # print(path)


m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
weight_AT = '../defenses/weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
weight_norm = '../defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
model_dict = torch.load(weight_AT)
model.load_state_dict(model_dict)


def attack(image, label, attack_name):
    fmodel = foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),
                                         num_classes=10)  # , preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity  # MeanAbsoluteDistance
    attacker = attackers[attack_name](fmodel, criterion=criterion1, distance=distance)

    image = image.cpu().numpy()
    label = label.cpu().numpy()

    adversarials = image.copy()
    advs = attacker(image, label)  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
    for i in tqdm.tqdm(range(len(advs)), ncols=80):
        if advs is not None:
            adv = torch.renorm(torch.from_numpy(advs[i] - image[i]), p=2, dim=0, maxnorm=100).numpy() + image[i]

            adversarials[i] = adv
    adversarials = torch.from_numpy(adversarials).to(DEVICE)

    return adversarials



path_CW = '../data/cifar10/grey/CW'
path_DF = '../data/cifar10/grey/Deepfool'
path_FGSM = '../data/cifar10/ae/FGSM/'+args.data_loader
path_deepfool = '../data/cifar10/ae/deepfool/'+args.data_loader
path_pgd = '../data/cifar10/ae/PGD/'+args.data_loader

def init_dataloader(input_data):
    test_loader = load_data_for_defense(input_data)['dev_data']
    print("data_loader :" , len(data_loader))
    for i, (images, labels) in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        print("\nepoch " +str(i)+'\n')
        images, labels = images.to(DEVICE), labels.to(DEVICE)


        if args.fgsm is True:
            FGSM = attack(images, labels, 'FGSM')
            for t in range(args.batch_size):
                ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
                name = '/FGSM_' + str(i) + str(t) + '.png'
                out_path = os.path.join(path_FGSM, str(labels[t].cpu().numpy()))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                # print(out_path)
                out = out_path + name
                scipy.misc.imsave(out, ddn2)

        if args.deepfool is True:
            deepfool = attack(images, labels, 'DeepFoolAttack')
            for t in range(args.batch_size):
                ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
                name = '/deepfool_' + str(i) + str(t) + '.png'
                out_path = os.path.join(path_deepfool, str(labels[t].cpu().numpy()))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                # print(out_path)
                out = out_path + name
                scipy.misc.imsave(out, ddn2)

        if args.pgd is True:
            PGD = attack(images, labels, 'PGD')
            for t in range(args.batch_size):
                ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
                name = '/PGD_' + str(i) + str(t) + '.png'
                out_path = os.path.join(path_pgd, str(labels[t].cpu().numpy()))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                # print(out_path)
                out = out_path + name
                scipy.misc.imsave(out, ddn2)

send_email("FSGM, deepfoll, PGD 对抗样本生成", title="对抗样本生成完毕")

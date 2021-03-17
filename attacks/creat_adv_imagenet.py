import glob
import os
import argparse
import random
import socket

import imageio
import scipy
import tqdm
import numpy as np
import torch
from PIL import Image
import logging

#
from utils.utils import load_model

logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
from foolbox.models import PyTorchModel
# from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack, \
    SpatialAttack, CarliniWagnerL2Attack
from progressbar import *
from torch.utils import data


from torchvision import transforms
import sys

# from process.preprocess import JPEGcompression, _gridmask, _TVM, _jpeg_compression2, _jpeg_compression

sys.path.append("..")
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from utils.messageUtil import send_email


import warnings
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from attacks import DDN
from attacks import M_DI2_FGSM_Attacker

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Extend sample')
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
def attack_integrated(image, label, attack_name):
    fmodel = PyTorchModel(model, bounds=(0, 1),
                          num_classes=1000)  # , preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity  # MeanAbsoluteDistance
    attacker = attackers[attack_name](fmodel, criterion=criterion1, distance=distance)

    image = image.cpu().numpy()
    label = label.cpu().numpy()

    adversarials = image.copy()
    advs = attacker(image, label)  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
    # for i in range(len(advs)):
    #     if advs is not None:
    #         adv = torch.renorm(torch.from_numpy(advs[i] - image[i]), p=2, dim=0, maxnorm=1).numpy() + image[i]
    #
    #         adversarials[i] = adv
    adversarials = torch.from_numpy(advs).to(DEVICE)

    return adversarials

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')


def load_data(csv, input_dir, img_size=224, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    transformer = transforms.Compose([

        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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


train_input_dir = '../data/imagenet_all/images/'
# train_input_dir = './tianchi2021_adv/m_di2_fgsm_resnet152/'
# train_input_dir = '/media/unknown/frankfeng/researchData/code/Attack-ImageNet/results_20/images/'


csv_file = r'../data/imagenet_all/dev.csv'
data_loader = load_data(csv_file, train_input_dir)['dev_data']
print("data_loader :", len(data_loader))

model = load_model(args.model_name).to(DEVICE).eval()
epoch = 0
accs = 0
accs_advs = 0
m_di2_fgsm_attacker = M_DI2_FGSM_Attacker(steps=args.steps,
                                          max_norm=args.max_norm / 255.0,
                                          div_prob=args.div_prob,
                                          device=DEVICE)

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
        adv = m_di2_fgsm_attacker.attack(model, images, labels)

        # predictions_advs = model(adv).argmax(1)
        #
        # accuracy = (predictions_advs == labels).float().mean()
        # accs_advs += accuracy.item()
        #
        # if epoch < 5 or (epoch + 1) % 50 == 0:
        #     print(epoch,'adv prediction ', accs_advs / (epoch + 1))
        out_path = './tianchi2021_adv/m_di2_fgsm_efficientnetb8_advprop_l210_step50/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for t in range(args.batch_size):
            adv_img = np.transpose(adv[t].detach().cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            adv_img = scipy.misc.imresize(adv_img, size=(500, 500))
            scipy.misc.imsave(out, adv_img)
    if args.clipAttack:
        out_path = './tianchi2021_adv/clipAttack/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        clip = Image.open("./clip.png").convert("RGB").resize((500, 500))
        clip = np.array(clip) / 255.0
        # clip = transforms.ToTensor()(clip)
        for t in range(args.batch_size):
            img = np.transpose(images[t].cpu().numpy(), (1, 2, 0))
            img = img * 0.7 + clip * 0.3
            img = torch.from_numpy(img)
            print(type(img))
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(img, size=(500, 500))
            scipy.misc.imsave(out, ddn2)
    if args.cropMaskAttack:
        out_path = './tianchi2021_adv/cropMask/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        start = 125

        zeros = torch.zeros([250, 250]).to(DEVICE)
        zeros_up = torch.ones([250, start]).to(DEVICE)
        zeros_down = torch.ones([250, 500 - start - 250]).to(DEVICE)
        zeros_left = torch.ones([start, 500]).to(DEVICE)
        zeros_right = torch.ones([500 - start - 250, 500]).to(DEVICE)

        mask = torch.cat((zeros_up, zeros, zeros_down), dim=1)
        mask = torch.cat((zeros_left, mask, zeros_right), dim=0)
        mask = torch.repeat_interleave(mask.unsqueeze(dim=0), repeats=3, dim=0)
        mask = torch.repeat_interleave(mask.unsqueeze(dim=0), repeats=args.batch_size, dim=0)

        images = images * mask
        for t in range(args.batch_size):
            ddn2 = np.transpose(images[t].cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(ddn2, size=(500, 500))
            scipy.misc.imsave(out, ddn2)
    if args.randomMaskAttack:
        out_path = './tianchi2021_adv/randomMask/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # mask = torch.ones([args.batch_size, 3, 500, 500]).to(DEVICE)
        start = random.randint(1, 333)

        zeros = torch.zeros([166, 166]).to(DEVICE)
        zeros_up = torch.ones([166, start]).to(DEVICE)
        zeros_down = torch.ones([166, 500 - start - 166]).to(DEVICE)
        zeros_left = torch.ones([start, 500]).to(DEVICE)
        zeros_right = torch.ones([500 - start - 166, 500]).to(DEVICE)

        mask = torch.cat((zeros_up, zeros, zeros_down), dim=1)
        mask = torch.cat((zeros_left, mask, zeros_right), dim=0)
        mask = torch.repeat_interleave(mask.unsqueeze(dim=0), repeats=3, dim=0)
        mask = torch.repeat_interleave(mask.unsqueeze(dim=0), repeats=args.batch_size, dim=0)
        # for batch in range(mask.shape[0]):
        #     for channel in range(mask.shape[1]):
        #         for x in range(start, start+166):
        #             for y in range(start, start+166):
        #                 mask[batch][channel][x][y] = 0
        images = images * mask
        for t in range(args.batch_size):
            ddn2 = np.transpose(images[t].cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(ddn2, size=(500, 500))
            scipy.misc.imsave(out, ddn2)
    if args.ddn is True:
        out_path = './tianchi2021_adv/ddn_res152_new/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        attacker = DDN(steps=1000, device=DEVICE)
        # print(images.max(),images.min())
        adv = attacker.attack(model, images, labels)
        for t in range(args.batch_size):
            ddn2 = np.transpose(adv[t].detach().cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(ddn2, size=(500, 500))
            scipy.misc.imsave(out, ddn2)

    # if args.PGD is True:
    #     attacker = PGD_L2(steps=100, device=DEVICE, max_norm=args.max)
    #     out_path = './tianchi2021_adv/pgd_resnet50_at/'
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)
    #     pgd = attacker.attack(model, inputs=images, labels=labels)
    #     for t in range(args.batch_size):
    #         pgd = np.transpose(pgd[t].detach().cpu().numpy(), (1, 2, 0))
    #         name = filenames[t]
    #         if not os.path.exists(out_path):
    #             os.makedirs(out_path)
    #         out = out_path + name
    #         scipy.misc.imsave(out, pgd)

    if args.fgsm is True:
        FGSM = attack_integrated(images, labels, 'FGSM')
        predictions = model(FGSM).argmax(1)
        print(FGSM.shape, FGSM.max(), FGSM.min(), predictions)
        adv_accuracy = (predictions == labels).float().mean()
        print("adv_accuracy ", adv_accuracy)
        accs_advs += adv_accuracy.item()
        if epoch < 5 or (epoch + 1) % 50 == 0:
            print(epoch, "fgsm adv: ", accs_advs / (epoch + 1))
        for t in range(args.batch_size):
            ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            out_path = './tianchi2021_adv/fgsm_resnet152/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            # ddn2 = scipy.misc.imresize(ddn2)
            scipy.misc.imsave(out, ddn2)

    if args.deepfool is True:
        deepfool = attack_integrated(images, labels, 'DeepFoolAttack')
        predictions = model(images).argmax(1)
        adv_accuracy = (predictions == labels).float().mean()
        print(epoch, "deepfool adv: ", adv_accuracy.item())
        for t in range(args.batch_size):
            ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            path_deepfool = './tianchi2021_adv/deepfool_resnet152/'
            out_path = path_deepfool
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(ddn2, size=(500, 500))
            scipy.misc.imsave(out, ddn2)

    if args.pgd is True:
        PGD = attack_integrated(images, labels, 'PGD')
        predictions = model(PGD).argmax(1)
        adv_accuracy = (predictions == labels).float().mean()
        accs_advs += adv_accuracy.item()
        # if (epoch + 1) % 50 == 0:
        print(epoch, "PGD adv: ", accs_advs / (epoch + 1))
        for t in range(args.batch_size):
            ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
            name = filenames[t]
            path_pgd = './tianchi2021_adv/pgd_resnet152/'
            out_path = path_pgd
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            ddn2 = scipy.misc.imresize(ddn2, size=(500, 500))
            scipy.misc.imsave(out, ddn2)
    epoch += 1

print(accs / epoch, accs_advs / epoch)
hostname = socket.gethostname()
send_email("FSGM对抗样本生成", title=hostname+": 对抗样本生成完毕")

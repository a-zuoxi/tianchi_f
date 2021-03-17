import os
import PIL
import glob
import argparse
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
from shape_precess import shape
from sklearn.metrics import accuracy_score
#from torchvision.datasets import CIFAR10
import sys
#from fast_adv.models.cifar10 import wide_resnet
#from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.models.cifar10.model_mixed_attention import wide_resnet

from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
import scipy.misc


sys.path.append("..")



class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(scipy.misc.imread(image_path))#
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

def load_data_for_defense(img_size=32,batch_size=32):
    input_data = [
        #'../attacks/DDN_1000_2/*/*.png']
        #'../defenses/data/shape_train2/*/*.png']

        '../attacks/PGD/*/*.png']
         #'../attacks/DeepFool2/*/*.png']
        #'../attacks/FGSM/*/*.png']
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





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='test_FGSM', help='path to dataset')
   # parser.add_argument('--input_dir', default='jpg_test_PGD',
                     #   help='Input directory with images.', type=str)
    parser.add_argument('--output_file', default='output.csv',
                        help='Output file to save labels', type=str)
    parser.add_argument('--target_model', default='densenet161',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=128,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    return parser.parse_args()


if __name__ == '__main__':
        args = parse_args()
        gpu_ids = args.gpu_id
        if isinstance(gpu_ids, int):

            gpu_ids = [gpu_ids]
        batch_size = args.batch_size
        target_model = args.target_model
        #inputDir = args.input_dir
        outputFile = args.output_file
    ################## Defense #######################
        m = wide_resnet(num_classes=10, depth=28, widen_factor=10,
                            dropRate=0.3)  # densenet_cifar(num_classes=110)##densenet121(num_classes=110)#wide_resnet(num_classes=110, depth=28, widen_factor=10, dropRate=0.3) ######
        # Loading data for ...densenet161(num_classes=110)
        image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')  # torch.device('cpu')
        #pth_file ='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth'
        pth_file = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_mixed_attention_cifar10_ep_12_val_acc0.8895.pth'
        #pth_file = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth'

        model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(device)
        #model = model.to(device)
        print('loading data for defense using %s ....' % target_model)

        test_loader = load_data_for_defense()['dev_data']



        # pth_file = os.path.join(weights_path, 'cifar10acc0.9090267625855811_40.pth')#18.24944_17.2555_ep_31__acc0.7470.pthep_36_val_acc0.9844.pthep_36_val_acc0.9844.pth#glob.glob(os.path.join(weights_path, 'cifar10_20_0.73636.pth'))#[0]
        print('loading weights from : ', pth_file)
        #model_dict = torch.load('jpeg_weight/18.5459_0.9_jpeg_WRN_DDNacc0.9553740539334037_20_0.60.pth')
        #model_dict = torch.load('jpeg_weight/ALP_smooth_p_44_val_acc0.9409.pth')
        #model_dict = torch.load('jpeg_weight/JPEG_ALP_smooth_acc0.9401_all_0.786.pth')
        model_dict = torch.load(pth_file)
        model.load_state_dict(model_dict, False)
        model.eval()
        test_accs = AverageMeter()
        test_losses = AverageMeter()
        widgets = ['test :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        with torch.no_grad():
            for batch_data in pbar(test_loader):
                images, labels = batch_data['image'].to(device), batch_data['label_idx'].to(device)
                #print(labels)
                #logits = model(images)

                image_shape = torch.zeros_like(images)
                #print('shapeshape')
                for i in range(args.batch_size):
                    try:
                        new = images[i]
                        new = new.cpu()
                        new = new.numpy()  # * 255
                        new = shape(new)
                        # new = new.transpose((2, 0, 1))
                        image_shape[i] = torch.from_numpy(new)
                    except:
                        continue

                # print('attack finish'+str(time.clock()))
                l2_norms = (image_shape - images).view(args.batch_size, -1).norm(2, 1)
                # mean_norm = l2_norms.mean()
                image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=1) + images

                logits = model.forward_attention(images.detach(), image_shape.detach())
                #logits = model(images.detach())
                #logits = model(image_shape.detach())


                test_accs.append((logits.argmax(1) == labels).float().mean().item())
               # print(test_accs)
                #test_losses.append(loss.item())
        print('\nTest accuracy ',test_accs.avg)

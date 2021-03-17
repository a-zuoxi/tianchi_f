
import argparse
import glob
import json

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from progressbar import *
from torchvision import transforms
import sys
#from fast_adv.models.cifar10 import wide_resnet
from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.models.wsdan.wsdan import WSDAN
# from fast_adv.process.jpeg_compression import JpegCompression
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
import scipy.misc


sys.path.append("..")

# from fast_adv.process.preprocess import TVMcompression, JPEGcompression, GridMaskCompression


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

input_data = [
    # '/home/frankfeng/桌面/cifar10/org/*/*.png',
    '../data/cifar10/org/*/*.png',
    '../data/cifar10/white/1_AT_PGD/*/*.png',
    '../data/cifar10/white/2_AT_PGD/*/*.png',
    '../data/cifar10/white/4_AT_PGD/*/*.png',
    '../data/cifar10/white/Deepfool/*/*.png',
    '../data/cifar10/white/CW/*/*.png',
    # '../data/cifar10/grey/3PGD/*/*.png',
    # '../data/cifar10/white/0.5_AT_DDN/*/*.png',
    # '../data/cifar10/white/1_AT_PGD_NEW/*/*.png',
    # '../data/cifar10/white/0.5_MIX_PGD/*/*.png'
]

def load_data_for_defense(input_index, img_size=32,batch_size=16):



    all_imgs = []
    all_labels = []
    #for input_dir in jpg_data:
    # for input_dir in input_data:
    #     one_imgs = glob.glob(input_dir)  # (os.path.join(input_dir, './*/*.jpg'))
    #     one_labels = [int(img_path.split('/')[-2]) for img_path in one_imgs]
    #     all_imgs.extend(one_imgs)
    #     all_labels.extend(one_labels)
    input_dir =  input_data[input_index]
    one_imgs = glob.glob(input_dir)  # (os.path.join(input_dir, './*/*.jpg'))
    one_labels = [int(img_path.split('/')[-2]) for img_path in one_imgs]
    all_imgs.extend(one_imgs)
    all_labels.extend(one_labels)

    print(len(all_labels))
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    # print(all_labels)
    transformer = transforms.Compose([
        # JpegCompression(),
        transforms.ToTensor(),
        # JPEGcompression()
        # TVMcompression()
        # GridMaskCompression()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
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
        # m = wide_resnet(num_classes=10, depth=28, widen_factor=10,
        #                     dropRate=0.3)  # densenet_cifar(num_classes=110)##densenet121(num_classes=110)#wide_resnet(num_classes=110, depth=28, widen_factor=10, dropRate=0.3) ######
        #
        m = WSDAN(num_classes=10, M=32, net='wide_resnet', pretrained=True)
        # device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        # model = model.to(device)


        # Loading data for ...densenet161(num_classes=110)
        image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')  # torch.device('cpu')
        model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(device)
        model = model.to(device)
        print('loading data for defense using %s ....' % target_model)

        # weight_norm = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
        # weight_AT = './weights/cifar10_AT/cifar10acc0.8709999859333039_45.pth'
        # weight_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/AT+ALP/cifar10acc0.8699999809265136_50.pth'
        #
        # weight_conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_mixed_Attention/cifar10acc0.8759999752044678_100.pth'
        # weight_025conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
        # weight_05conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_cifar10_mixed_Attention/cifar10acc0.8434999763965607_130.pth'
        # weight_1conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1MixedAttention_mixed_attention_cifar10_ep_25_val_acc0.7080.pth'
        #
        # weight_shape_alp='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/shape_ALP_cifar10_ep_79_val_acc0.7625.pth'
        # weight_attention = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_Attention/cifar10acc0.8729999780654907_120.pth'
        #
        # weight_025conv_mixatten_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25Mixed+ALP_cifar10_ep_85_val_acc0.8650.pth'
        #
        # weight_smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
        # weight_05smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'
        # weight_025smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
        # weight_1smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1random_smooth_cifar10_ep_107_val_acc0.5380.pth'

        weight_wsdan = './weights/cifar10_WSDAN/cifar10_valacc0.8849999785423279.pth'
        weight_aad = './weights/cifar10_aad/cifar10_3loss_valacc_new0.7284999847412109.pth'
        weight_AT='./weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
        weight_aad2 = './weights/cifar10_aad2/cifar10_3loss_valacc_new0.769999983906746.pth'
        weight_aad3 = './weights/cifar10_aad4/cifar103loss_acc0.8179999828338623_30.pth'
        weigth_aad4='./weights/cifar10_aad4/cifar103loss_acc0.816499975323677_40.pth'
        weight_wsdan_new = './weights/cifar10_WSDAN_best/cifar10_valacc0.8784999758005142.pth'
        weight_wsdan2 = './weights/cifar10_WSDAN_smooth3/cifar10_valacc0.8644999772310257.pth'
        weight_wsdan_smooth_add2loss = './weights/cifar10_wsgan_smooth_add2loss/cifar10_valacc0.8439999759197235_40.pth'
        weight_wsdan_smooth_add2loss_meanAtt = './weights/cifar10_wsgan_smooth_add2loss_meanAttention/cifar10_valacc0.8394999742507935.pth'
        weight_wsdan_dropCrop = './weights/cifar10_wsgan_dropCropLoss/cifar10_valacc0.8774999737739563.pth'
        weight_wsdan3 = '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/defenses/weights/cifar10_wsgan_3/cifar10_valacc0.pth'
        weight_wsdan_final= '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/defenses/weights/cifar10_wsdan_final/cifar10_valacc0.871999979019165.pth'
        weight_wsdan_best_jpeg = './weights/cifar10_wsgan_jpeg/cifar10_valacc0.9990079365079365.pth'
        weight_wsdan_best_jpeg2 = './weights/cifar10_wsgan_jpeg2/cifar10_valacc0.9679878048780488.pth'
        weight_wsdan_best_jpeg3 = './weights/cifar10_wsgan_jpeg3/cifar10_valacc0.9732505809970018.pth'
        weight_wsdan_best_jpeg4 = './weights/cifar10_wsgan_jpeg4_epoch10/cifar10_valacc0.958079268292683.pth'

        weight_wsdan_best = "./weights/cifar10_WSDAN_best/cifar10_0.87_low.pth"

        weight_wsdan_best_cd2 = './weights/cifar10_WSDAN_best/cropdrop_proportion_0.2/cifar10_valacc0.8834999799728394.pth'
        weight_wsdan_best_cd4 = './weights/cifar10_WSDAN_best/cropdrop_proportion_0.4/cifar10_valacc0.8804999709129333.pth'
        weight_wsdan_best_cd6 = './weights/cifar10_WSDAN_best/cropdrop_proportion_0.6/cifar10_valacc0.8759999811649323.pth'
        weight_wsdan_best_cd8 = './weights/cifar10_WSDAN_best/cropdrop_proportion_0.8/cifar10_valacc0.8799999743700028.pth'

        weight_cifar10_base_jpeg ='./weights/cifar10_jpeg/cifar10acc0.946656050955414_10.pth'
        weight_cifar10_base_tvm = './weights/cifar10_tvm/cifar10acc0.9386942675159236_10.pth'
        weight_cifar10_base_gridmask = './weights/cifar10_gridmask/cifar10acc0.8640525477707006_10.pth'
        model_dict = torch.load(weight_wsdan)

        model.load_state_dict(model_dict)
        model.eval()

        test_result ={}
        for i in range(len(input_data)):
            test_accs = AverageMeter()
            test_losses = AverageMeter()
            widgets = ['test :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                       ' ', ETA(), ' ', FileTransferSpeed()]

            pbar = ProgressBar(widgets=widgets)

            test_loader = load_data_for_defense(i)['dev_data']
            with torch.no_grad():
                for batch_data in pbar(test_loader):
                    images, labels = batch_data['image'].to(device), batch_data['label_idx'].to(device)
                    # print(images, labels)
                    # noise = torch.randn_like(images, device='cuda') * 0.2
                    # image_shape = images + noise
                    #image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=1) + images
                    #logits,_ = model.forward_attention(images.detach(), image_shape.detach())
                    logits,_,_ = model(images.detach())
                    #logits = model(image_shape.detach())
                    test_accs.append((logits.argmax(1) == labels).float().mean().item())

            print('\nTest accuracy of %s:%s'%(input_data[i].split('/')[-3],test_accs.avg))
            test_result[input_data[i].split('/')[-3]] = test_accs.avg
        data = json.dumps(test_result, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
        dic = json.loads(data)
        print(dic)
        for d in dic.keys():
            print(d,'\t\t\t\t\t',dic[d])

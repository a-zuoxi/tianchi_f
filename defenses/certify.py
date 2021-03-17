# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from torch.utils import data
from time import time
from torchvision import transforms
from torchvision.datasets import CIFAR10
#from architectures import get_architecture
from core import Smooth
#from core_mix import Smooth
#from datasets import get_dataset, DATASETS, get_num_classes
import torch
#from fast_adv.models.cifar10.model_attention import Attention_integration
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
#from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
from fast_adv.models.cifar10.model_attention import wide_resnet
#from fast_adv.models.cifar10.wide_resnet import wide_resnet

parser = argparse.ArgumentParser(description='Certify many examples')
#parser.add_argument("dataset", choices=DATASETS, help="which dataset")
#parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
#parser.add_argument("sigma", type=float, default=0.25,help="noise hyperparameter")
#parser.add_argument("outfile", type=str, default='certification_output',help="output file")
parser.add_argument("--batch", type=int, default=200, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--data', default='data/cifar10', help='path to dataset')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=args.drop)
    DEVICE = torch.device('cuda:0' )
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    # model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth')
    weight_025conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
    weight_conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_mixed_Attention/cifar10acc0.8759999752044678_100.pth'
    weight_025conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
    weight_05conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_cifar10_mixed_Attention/cifar10acc0.8434999763965607_130.pth'
    weight_1conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1MixedAttention_mixed_attention_cifar10_ep_25_val_acc0.7080.pth'
    weight_shape_alp = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/shape_ALP_cifar10_ep_79_val_acc0.7625.pth'
    weight_attention = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_Attention/cifar10acc0.8729999780654907_120.pth'
    weight_025conv_mixatten_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25Mixed+ALP_cifar10_ep_85_val_acc0.8650.pth'
    weight_smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
    weight_05smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'
    weight_025smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
    weight_1smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1random_smooth_cifar10_ep_107_val_acc0.5380.pth'

    model_file = weight_025smooth
    model_dict = torch.load(model_file)
    model.load_state_dict(model_dict)

    # create the smooothed classifier g
    smoothed_classifier = Smooth(model, 10, 0.25)

    # prepare output file
    f = open('out_certify_025_smo100000', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    #dataset = get_dataset(args.dataset, args.split)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    print('56')
    dataset = data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                          list(range(48000, 50000)))
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        #print('123')
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        #print('232133',i, label, prediction, radius, correct, time_elapsed)
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()

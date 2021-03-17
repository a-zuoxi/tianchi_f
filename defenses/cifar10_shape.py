import glob
import os
import argparse
import tqdm
from progressbar import *
from copy import deepcopy
from shape_precess import shape
import os
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
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import CIFAR10

#from fast_adv.models.cifar10.model_attention import Attention_integration
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from fast_adv.models.cifar10.model_attention import wide_resnet
#from fast_adv.models.cifar10.wide_resnet import wide_resnet

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

parser.add_argument('--data', default='data/cifar10', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/cifar10/', help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.2, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=10, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int,default=0, help='epoch to start training with adversarial images')
parser.add_argument("--shape",type=int,default=None)
parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=100, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, default=8097,help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

def prepareData(input_dir='/media/unknown/Data/PLP/fast_adv/defenses/data/shape_train2',img_size=32):#img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    datasets = {
        'train_data': ImageSet(train, train_transform)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=64,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders
def prepareData_test(input_dir='/media/unknown/Data/PLP/fast_adv/attacks/shape_train_renorm',img_size=32):#img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    datasets = {
        'test_data': ImageSet(train, train_transform)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=64,
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
        image = self.transformer(Image.open(image_path))  # .convert('RGB'))
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
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])
#####注意之前没有把
train_loader = prepareData()['train_data']
val_loader=prepareData_test()['test_data']
train_set = data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(30000)))
val_set = data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                      list(range(48000, 50000)))
test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

#train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
 #                              drop_last=True, pin_memory=True)
#val_loader = data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)

m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=args.drop)
model =NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
#model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth')
model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth')
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

for epoch in range(args.epochs):

    scheduler.step()
    cudnn.benchmark = True
    model.train()
    requires_grad_(m, True)
    accs = AverageMeter()
    losses = AverageMeter()
    attack_norms = AverageMeter()

    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    i = 0
    sigma = 0.5
    for batch_data in pbar(train_loader):

        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        #images, labels = images.to(DEVICE), labels.to(DEVICE)

        #原图loss
        #logits_clean = model.forward(images)
        logits_clean = model.forward(images)
        loss = F.cross_entropy(logits_clean, labels)

        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(m, False)
            adv = attacker.attack(model, images, labels)
            l2_norms = (adv - images).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - images, p=2, dim=0, maxnorm=args.max_norm) + images

            attack_norms.append(mean_norm.item())
            requires_grad_(m, True)
            model.train()


            logits_adv = model(adv.detach())
            loss_adv = F.cross_entropy(logits_adv, labels)
            loss=loss+ loss_adv #+ 0.5*F.mse_loss(logits_adv,logits)

        if args.shape is not None and epoch >= args.shape:
            model.eval()
            requires_grad_(m, False)
            image_shape=torch.zeros_like(images)
            print('shapeshape')
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
            #mean_norm = l2_norms.mean()
            if args.max_norm:
                image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=args.max_norm) + images
            requires_grad_(m, True)
            model.train()

            logits_shape = model(image_shape.detach())
            #loss_shape = F.cross_entropy(logits_shape, labels)

            logits=model.forward_attention(images.detach(),image_shape.detach())
            loss_attention=F.cross_entropy(logits, labels)
            #attention
            #loss=loss_adv+loss_attention
            #attention+plp
            loss = loss_adv + loss_attention+ 0.2*F.mse_loss(logits_adv,logits_clean)+0.2*F.mse_loss(logits_clean,logits_shape)




        #loss = loss+ loss_adv + 0.5*F.mse_loss(logits_adv,logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.append((logits_clean.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())

        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, min(losses.last_avg, max_loss))
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
        widgets = ['val :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        length = len(val_loader)
        i = 0
        sigma = 0.5
        for batch_data in pbar(val_loader):
            images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
            logits=model(images)

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
        files2remove = glob.glob(os.path.join(save_path, '2SAT_*'))
        for _i in files2remove:
            os.remove(_i)
        strsave = "2SAT_cifar10_ep_%d_val_acc%.4f.pth" % (epoch, best_acc)
        torch.save(model.cpu().state_dict(),
                   os.path.join(save_path, strsave))
        model.to(DEVICE)

    if args.adv is None and val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())

    if not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(), os.path.join(args.save_folder, args.save_name + 'acc{}_{}.pth'.format(val_accs.avg,(epoch + 1))), cpu=True)

if args.adv is None:
    model.load_state_dict(best_dict)

test_accs = AverageMeter()
test_losses = AverageMeter()

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        test_accs.append((logits.argmax(1) == labels).float().mean().item())
        test_losses.append(loss.item())

if args.adv is not None:
    print('\nTest accuracy with final model: {:.4f} with loss: {:.4f}'.format(test_accs.avg, test_losses.avg))
else:
    print('\nTest accuracy with model from epoch {}: {:.4f} with loss: {:.4f}'.format(best_epoch,
                                                                                      test_accs.avg, test_losses.avg))

print('\nSaving model...')
save_checkpoint(model.state_dict(), os.path.join(args.save_folder, args.save_name + '.pth'), cpu=True)

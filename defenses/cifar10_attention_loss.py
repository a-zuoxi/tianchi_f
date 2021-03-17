import glob
import os
import argparse
import tqdm
from copy import deepcopy
from shape_precess import shape
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
from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
#from fast_adv.models.cifar10.model_attention import wide_resnet
#from fast_adv.models.cifar10.wide_resnet import wide_resnet

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

parser.add_argument('--data', default='data/cifar10', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/ALP_shape/', help='folder to save state dicts')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')
parser.add_argument('--noise_sd', default=0.25, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.2, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=10, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int,default=0, help='epoch to start training with adversarial images')
parser.add_argument("--shape",type=int,default=0)
parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, default=8097,help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(port=args.visdom_port) if args.visdom_port else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

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
#model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth')
weight_025conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
'''
model_file=weight_025conv_mixatten
#'/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
    #'/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth'
    #'/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth'
model_dict=model.state_dict()
save_model = torch.load(model_file,map_location=lambda storage, loc: storage)
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
#print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
model.load_state_dict(model_dict)
'''
#model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth')
#model.load_state_dict(model_dict)
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

    length = len(train_loader)
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #原图loss
        #logits_clean = model.forward(images)
        logits_clean,feature_clean = model.forward(images)
        #loss = F.cross_entropy(logits_clean, labels)

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


            logits_adv,feature_adv = model(adv.detach())
            loss_adv = F.cross_entropy(logits_adv, labels)
            #loss=loss+ loss_adv #+ 0.5*F.mse_loss(logits_adv,logits)

        if args.shape is not None and epoch >= args.shape:
            model.eval()
            requires_grad_(m, False)
            noise = torch.randn_like(images, device='cuda') * args.noise_sd
            image_shape = images + noise
            #if args.max_norm:
               # image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=args.max_norm) + images
            requires_grad_(m, True)
            model.train()

            #logits_shape = model(image_shape.detach())
            #loss_shape = F.cross_entropy(logits_shape, labels)


            logits,feature_shape=model(image_shape.detach())
            loss_ALP_shape=F.mse_loss(feature_clean, feature_shape)
            #attention
            #loss=loss_adv+loss_attention
            #attention+plp
            loss = loss_adv + 0.5*loss_ALP_shape+ 0.2*F.mse_loss(logits_adv,logits_clean)+0.1*F.mse_loss(feature_adv,feature_clean)




        #loss = loss+ loss_adv + 0.5*F.mse_loss(logits_adv,logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.append((logits.argmax(1) == labels).float().mean().item())
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
        for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits,_ = model(images.detach())




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
        files2remove = glob.glob(os.path.join(save_path, 'shape_ALP_*'))
        for _i in files2remove:
            os.remove(_i)
        strsave = "shape_ALP_cifar10_ep_%d_val_acc%.4f.pth" % (epoch, best_acc)
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

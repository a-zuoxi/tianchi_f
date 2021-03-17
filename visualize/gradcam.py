"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import glob
import os
import sys
import time
import cv2
import imageio
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as mpl_color_map
import torch.nn.functional as F
import torch
from progressbar import *
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models
from torchvision import transforms
sys.path.append("..")

from models.resnet.resnet import resnet152

# from tianchi2021.models.cifar10.model_mixed_attention import wide_resnet
from process.preprocess import _jpeg_compression2, _TVM, _gridmask
from utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        # im_as_arr[channel] -= mean[channel]
        # im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    # ('smooth_and_adv/0_grey.png', 1),
    # ('../data/cifar10/grey/100PGD/7/ddn_1205.png', 1),
    example_list = (
        ('../data/imagenet_all/images/0.jpg', 0),
        ('../data/cifar10/org/0/org_309.png', 0),
        ('../data/cifar10/org/7/org_1205.png', 7),
        ('../data/cifar10/org/8/org_1146.png', 8),
        ('../data/cifar10/org/1/org_7211.png', 1)
    )
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

    prep_img = preprocess_image(original_image).to(DEVICE)
    # Define model

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    model = resnet152(pretrained=True).to(DEVICE).eval()
    # pretrained_model = models.alexnet(pretrained=True)
    weight = ''
    return (img_path,
            prep_img,
            target_class,
            weight,
            model)

def get_example_params2(img_path, target_class):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    # ('smooth_and_adv/0_grey.png', 1),
    # ('../data/cifar10/grey/100PGD/7/ddn_1205.png', 1),

    file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    m_AT = wide_resnet(num_classes=10, depth=28, widen_factor=10,dropRate=0.3)
    m = cifar10_WRN(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
    m_jpeg = wide_resnet(num_classes=10, depth=28, widen_factor=10,dropRate=0.3)
    m_gridmask = wide_resnet(num_classes=10, depth=28, widen_factor=10,dropRate=0.3)
    m_tvm = wide_resnet(num_classes=10, depth=28, widen_factor=10,dropRate=0.3)

    m_ours = WSDAN(num_classes=10, M=32, net='wide_resnet', pretrained=True)

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    device = 'cpu'  # torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')  # torch.device('cpu')
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(device)
    model_ours = NormalizedModel(model=m_ours, mean=image_mean, std=image_std).to(device)
    model_AT= NormalizedModel(model=m_AT, mean=image_mean, std=image_std).to(device)
    model_jpeg= NormalizedModel(model=m_jpeg, mean=image_mean, std=image_std).to(device)
    model_tvm = NormalizedModel(model=m_tvm, mean=image_mean, std=image_std).to(device)
    model_gridmask = NormalizedModel(model=m_gridmask, mean=image_mean, std=image_std).to(device)

    weight_norm = '/home/frankfeng/projects/researchData/AI_security/code/PLP/fast_adv/defenses/weights/cifar10_aad/cifar10_valacc0.8984999746084213.pth'
    weight_AT = '../defenses/weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'

    weight_ALP = '../defenses/weights/AT+ALP/cifar10acc0.8699999809265136_50.pth'
    weight_smooth = '../defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
    weight_025smooth = '../defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
    weight_05smooth = '../defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'
    weight_cifar10_base_jpeg = '../defenses/weights/cifar10_jpeg/cifar10acc0.946656050955414_10.pth'
    weight_025conv_mixatten = '../defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'

    weight_cifar10_base_jpeg = '../defenses/weights/cifar10_jpeg/cifar10acc0.946656050955414_10.pth'

    weight_wsdan_best = "../defenses/weights/cifar10_WSDAN_best/cifar10_0.87_low.pth"
    weight_wsdan_smooth_add2loss_meanAtt = './weights/cifar10_wsgan_smooth_add2loss_meanAttention/cifar10_valacc0.8394999742507935.pth'
    weight = '../defenses/weights/best/cifar10_80.pth'
    model_dict = torch.load(weight)
    model.load_state_dict(model_dict)
    model.eval()

    model_dict = torch.load(weight_AT)
    model_AT.load_state_dict(model_dict)
    model_AT.eval()

    model_dict = torch.load(weight_cifar10_base_jpeg)
    model_jpeg.load_state_dict(model_dict)
    model_jpeg.eval()

    weight_cifar10_base_tvm = '../defenses/weights/cifar10_tvm/cifar10acc0.9386942675159236_10.pth'
    weight_cifar10_base_gridmask = '../defenses/weights/cifar10_gridmask/cifar10acc0.8640525477707006_10.pth'
    model_dict = torch.load(weight_cifar10_base_tvm)
    model_tvm.load_state_dict(model_dict)
    model_tvm.eval()


    model_dict = torch.load(weight_cifar10_base_gridmask)
    model_gridmask.load_state_dict(model_dict)
    model_gridmask.eval()

    model_dict = torch.load(weight_wsdan_best)
    model_ours.load_state_dict(model_dict)
    model_ours.eval()

    # pretrained_model = models.alexnet(pretrained=True)
    return (img_path,
            prep_img,
            target_class,
            weight,
            model,
            model_AT,
            model_jpeg,
            model_tvm,
            model_gridmask,
            model_ours)

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

def draw_cam(timetrap, model,model_name,pa , prep_img,pre_class, clean_path1,probs):
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[pa].data.detach().numpy())

    # noise = torch.randn_like(prep_img, device='cpu') * 8
    # prep_img = prep_img + noise
    # print("prep_img:", prep_img.shape)
    features_blobs = model.feature_map(prep_img).detach().numpy()
    # print("features_blobs:", features_blobs.shape, "weight_softmax", weight_softmax.shape)
    # generate class activation mapping for the top1 prediction
    # print(pre_class)
    CAMs, Feature = returnCAM(features_blobs, weight_softmax, [pre_class])
    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % target_class)

    # clean_path1 = clean_path[target_example]

    img = cv2.imread(clean_path1)
    # #img=cv2.imread(img_path)
    height, width, _ = img.shape
    # print("img.shape", img.shape)

    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(cv2.resize(Feature[0], (width, height)), cv2.COLORMAP_JET)
    # print("s",heatmap, "\n\n\n\n\n\n",img)
    result = heatmap * 0.8 + img * 0.5
    result2 = heatmap2 * 0.8 + img * 0.5
    CAM_PATH = './test/5/' + timetrap+'_'+ model_name + '_' + str(pre_class) + '_CAM_' + str(probs) + '.jpg'
    # Feature_PATH = './test/' + timetrap+'_'+ model_name + '_' + str(pre_class) + '_Feature_' + str(probs) + '.jpg'
    cv2.imwrite(CAM_PATH, result)
    # cv2.imwrite(Feature_PATH, result2)
def get_some_example():
    # Get params
    all_img_paths = glob.glob(os.path.join("../data/cifar10/org/5/", './*.png'))
    errors_count= 0
    for target_example in range(len(all_img_paths)):
        img_path = all_img_paths[target_example]
        target_class = 5
        (img_path, prep_img, target_class, weight, model, model_AT, model_jpeg,model_tvm, model_gridmask, model_ours) = \
            get_example_params2(img_path, target_class)


        logits = model.forward(prep_img)
        h_x = F.softmax(logits, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_base = probs.numpy()
        # idx = idx.numpy()
        pre_class = logits.argmax(1).numpy()
        # for i in range(0, 5):
        #     print('{:.3f} -> {}'.format(probs[i], idx[i]))

        logits_AT = model_AT.forward(prep_img)
        pre_class_AT = logits_AT.argmax(1).numpy()
        h_x = F.softmax(logits_AT, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_AT = probs.numpy()


        prep_img_jpeg = _jpeg_compression2(prep_img.squeeze(0))
        logits_jpeg = model_jpeg.forward(prep_img_jpeg)
        pre_class_jpeg = logits_jpeg.argmax(1).numpy()
        h_x = F.softmax(logits_jpeg, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_jpeg = probs.numpy()

        prep_img_tvm = _TVM(prep_img)
        logits_tvm = model_tvm.forward(prep_img_tvm)
        pre_class_tvm = logits_tvm.argmax(1).numpy()
        h_x = F.softmax(logits_tvm, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_tvm = probs.numpy()

        prep_img_gridmask = _gridmask(prep_img)
        logits_gridmask = model_gridmask.forward(prep_img_gridmask)
        pre_class_gridmask = logits_gridmask.argmax(1).numpy()
        h_x = F.softmax(logits_gridmask, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_gridmask = probs.numpy()


        logits_ours,_,_ = model_ours.forward(prep_img)
        pre_class_ours = logits_ours.argmax(1).numpy()
        h_x = F.softmax(logits_ours, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs_ours = probs.numpy()

        print(pre_class, pre_class_AT, pre_class_jpeg, pre_class_tvm, pre_class_gridmask, pre_class_ours)
        print(probs_base[0],probs_AT[0],probs_jpeg[0], probs_tvm[0], probs_gridmask[0], probs_ours[0])

        if pre_class!= target_class:
            errors_count +=1
        if pre_class_AT != target_class:
            errors_count += 1
        if pre_class_jpeg != target_class:
            errors_count += 1
        if pre_class_tvm != target_class:
            errors_count += 1
        if pre_class_gridmask != target_class:
            errors_count += 1
        if pre_class_ours == target_class:
            if errors_count >=3:
                timetrap = str(time.time())
                draw_cam(timetrap,model,"base",-2,prep_img,pre_class,img_path,probs_base[0])
                draw_cam(timetrap,model_AT, "at", 80,prep_img, pre_class_AT, img_path, probs_AT[0])
                draw_cam(timetrap,model_jpeg, "jpeg", 80,prep_img_jpeg, pre_class_jpeg, img_path, probs_jpeg[0])
                draw_cam(timetrap,model_tvm, "tvm", 80, prep_img_tvm, pre_class_tvm, img_path, probs_tvm[0])
                draw_cam(timetrap,model_gridmask, "gridmask", 80, prep_img_gridmask, pre_class_gridmask, img_path, probs_gridmask[0])
                draw_cam(timetrap,model_ours, "ours", 80, prep_img, pre_class_ours, img_path, probs_ours[0])
        print()

batch_size = 10
def load_data(csv, input_dir, img_size=224, batch_size=10):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    transformer = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
                                      # transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      # transforms.Lambda(lambda img: img * 2.0 - 1.0),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ])
    # transformer = transforms.Compose([
    #     transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
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

csv_file = r'../data/imagenet_all/dev.csv'
train_loader = load_data(csv_file, train_input_dir)[
    'dev_data']

if __name__ == '__main__':


    widgets = ['create advs :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    epoch = 0
    accs = 0
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    model = resnet152(pretrained=True).to(DEVICE).eval()
    for batch_data in pbar(train_loader):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        logits = model.forward(images)
        # print('logits:', logits, 'labels:', labels)
        h_x = F.softmax(logits, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        pre_class = logits.argmax(1).cpu().numpy()

        k = 0
        params = list(model.parameters())

        weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())


        for i in range(batch_size):
            pre_img = images[i].unsqueeze(0)
            print(pre_class[i])
            features_blobs = model.feature_map(pre_img).cpu().detach().numpy()
            CAMs, Feature = returnCAM(features_blobs, weight_softmax, [pre_class[i]])
            clean_path1 = '../data/imagenet_all/images/'+filenames[i]
            img_clean = cv2.imread(clean_path1)
            height, width, _ = img_clean.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            # heatmap2 = cv2.applyColorMap(cv2.resize(Feature[0], (width, height)), cv2.COLORMAP_JET)
            clip = True
            mix = False
            if clip :
                threshold = 50
                feature_outside = cv2.resize(Feature[0], (width, height)) <= threshold
                feature_outside = np.expand_dims(feature_outside, -1).repeat(3, axis=-1)

                result2 = img_clean*feature_outside

                feature_center = cv2.resize(Feature[0], (width, height)) > threshold

                feature_center = np.expand_dims(feature_center, -1).repeat(3, axis=-1)
                clean_path2 = '../attacks/tianchi2021_adv/mdi2fgsm_dsn161jpeg_rn152jpeg_l210_step50/' + filenames[i]
                img_adv = cv2.imread(clean_path2)
                result_adv = img_adv * feature_center

                result = result_adv + result2
                # result = heatmap * 0.8 + img * 0.5
                # result2 = heatmap2 * 0.8 + img * 0.5
                # CAM_PATH = './imagenet/' + str(pre_class[i]) + '_CAM_' + str(i) + '.jpg'
                out_path = './mdi2fgsm_dsn161jpeg_rn152jpegl210_step50_attention_'+str(threshold)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                Feature_PATH = os.path.join(out_path, filenames[i])
                # cv2.imwrite(CAM_PATH, result)
                cv2.imwrite(Feature_PATH, result)
            if mix:
                threshold=60
                feature_outside = cv2.resize(Feature[0], (width, height)) <= threshold
                feature_outside = np.expand_dims(feature_outside, -1).repeat(3, axis=-1)

                result2 = img_clean * feature_outside

                feature_center = cv2.resize(Feature[0], (width, height)) > threshold

                feature_center = np.expand_dims(feature_center, -1).repeat(3, axis=-1)
                clean_path2 = '../attacks/tianchi2021_adv/m_di2_fgsm_resnet152_ft_jpeg_l210/' + filenames[i]
                img_adv = cv2.imread(clean_path2)
                result_adv = img_adv * feature_center
                img_clean_center = img_clean * feature_center
                result_adv = 0.9 * result_adv + 0.1 * img_clean_center

                result = result_adv + result2
                # result = heatmap * 0.8 + img * 0.5
                # result2 = heatmap2 * 0.8 + img * 0.5
                # CAM_PATH = './imagenet/' + str(pre_class[i]) + '_CAM_' + str(i) + '.jpg'
                out_path = './mdi2fgsmResnet152jpegl210_attention_mix'+str(threshold)+'_91'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                Feature_PATH = os.path.join(out_path, filenames[i])
                # cv2.imwrite(CAM_PATH, result)
                cv2.imwrite(Feature_PATH, result)

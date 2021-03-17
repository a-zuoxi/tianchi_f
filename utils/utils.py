from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append("..")
from models.resnet.FaceBook_resnet import resnet101_denoise, resnet152_denoise
from torchvision import transforms


def load_model(model_name):
    model = None
    if 'resnet152'.__eq__(model_name):
        model = models.resnet152(pretrained=True)
        print("load model: resnet152")
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std)
    elif 'resnet50'.__eq__(model_name):
        model = models.resnet50(pretrained=True)
        print("load model: resnet50")
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std)

    elif 'resnet50_ddn_jpeg'.__eq__(model_name):
        m = models.resnet50(pretrained=False)
        print("load model: resnet50_ddn_jpeg")
        weight = '../defenses/weights/imagenet_resnet50_jpeg/Imagenetacc0.954000004529953_20.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'resnet152_ddn_jpeg'.__eq__(model_name):
        print("load model resnet152_ddn_jpeg")
        m = models.resnet152(pretrained=False)
        weight = '../defenses/weights/imagenet_resnet152_jpeg/Imagenetacc0.9553571428571429_20.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model
        # class Resize(nn.Module):
        #
        #     def __init__(self, size):
        #         super(Resize, self).__init__()
        #         self.size = size
        #
        #     def forward(self, input):
        #         # input = transforms.Resize(self.size)(input)
        #         print(input.shape)
        #         return input
        # model = nn.Sequential(
        #     Resize(224),
        #     pretrained_model
        # )

    elif 'resnet152_ddn_gridmask'.__eq__(model_name):
        m = models.resnet152(pretrained=False)
        weight = '../defenses/weights/imagenet_resnet152_gridmask/Imagenetacc0.9662698412698413_10.pth'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'resnet152_ddn_tvm'.__eq__(model_name):
        m = models.resnet152(pretrained=False)
        weight = '../defenses/weights/imagenet_resnet152_tvm/Imagenetacc0.9593253968253969_20.pth'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'densenet161_ddn_jpeg'.__eq__(model_name):
        m = models.densenet161(pretrained=False)
        weight = '../defenses/weights/imagenet_densenet161_jpeg/Imagenetacc0.9543650793650794_20.pth'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'inception_v3_ddn_jpeg'.__eq__(model_name):
        m = models.inception_v3(pretrained=False)
        weight = '../defenses/weights/imagenet_inception_v3_jpeg/Imagenetacc0.9682539682539683_20.pth'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'inception_v3_ddn_jpeg'.__eq__(model_name):
        m = models.inception_v3(pretrained=False)
        weight = '../defenses/weights/imagenet_inception_v3_jpeg/Imagenetacc0.9682539682539683_20.pth'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'Adv_Denoise_Resnext101'.__eq__(model_name):
        m = resnet101_denoise()
        weight = '../defenses/weights/Adv_Denoise_Resnext101.pytorch'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict, strict=True)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'Adv_Denoise_Resnext101_ddn_jpeg'.__eq__(model_name):
        print('Adv_Denoise_Resnext101_ddn_jpeg')

        class Normalize(nn.Module):

            def __init__(self, mean, std):
                super(Normalize, self).__init__()
                self.mean = mean
                self.std = std

            def forward(self, input):
                size = input.size()
                x = input.clone()
                for i in range(size[1]):
                    x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

                return x

        class Permute(nn.Module):

            def __init__(self, permutation=[2, 1, 0]):
                super().__init__()
                self.permutation = permutation

            def forward(self, input):
                return input[:, self.permutation]
        pretrained_model1 = resnet101_denoise()

        model = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model1
        )
        weight = '../defenses/weights/imagenet_Adv_Denoise_Resnext101_jpeg/Imagenetacc0.9330000039935112_20.pth'
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)
    elif 'Adv_Denoise_Resnet152'.__eq__(model_name):
        m = resnet152_denoise()
        weight = '../defenses/weights/Adv_Denoise_Resnet152.pytorch'
        loaded_state_dict = torch.load(weight)
        m.load_state_dict(loaded_state_dict, strict=True)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'efficientnet_b8'.__eq__(model_name):

        pretrained_model = EfficientNet.from_pretrained('efficientnet-b8', advprop=True)
        class Normalize(torch.nn.Module):

            def __init__(self):
                super(Normalize, self).__init__()

            def forward(self, img):
                return img * 2.0 - 1.0

        model = torch.nn.Sequential(
            Normalize(),
            pretrained_model
        )

    elif 'efficientnet_b7'.__eq__(model_name):
        print("load model: efficientnet_b7" )
        pretrained_model = EfficientNet.from_pretrained('efficientnet-b7', advprop=False)
        class Normalize(torch.nn.Module):

            def __init__(self):
                super(Normalize, self).__init__()

            def forward(self, img):
                return img * 2.0 - 1.0

        model = torch.nn.Sequential(
            Normalize(),
            pretrained_model
        )
    elif 'efficientnet_b4'.__eq__(model_name):
        print("load model: efficientnet_b4")
        pretrained_model = EfficientNet.from_pretrained('efficientnet-b4', advprop=True)
        class Normalize(torch.nn.Module):

            def __init__(self):
                super(Normalize, self).__init__()

            def forward(self, img):
                return img * 2.0 - 1.0

        model = torch.nn.Sequential(
            Normalize(),
            pretrained_model
        )
    return model


def save_checkpoint(state: OrderedDict, filename: str = 'checkpoint.pth', cpu: bool = False) -> None:
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:

        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)
    def forward_attention(self, input: torch.Tensor,input2) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        normalized_input2 = (input2 - self.mean) / self.std
        return self.model.forward_attention(normalized_input,normalized_input2)
    def feature_map(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map(normalized_input)
    def feature_map2(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map2(normalized_input)



def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()
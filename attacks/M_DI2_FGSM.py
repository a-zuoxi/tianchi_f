

from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lpips
# from cnn_finetune import make_model
#
# model = make_model('senet154', num_classes=100, pretrained=True)
# print(model.original_model_info)
class M_DI2_FGSM_Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 100.0,
                 device: torch.device = torch.device('cpu')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.loss_fn = lpips.LPIPS(net='vgg').cuda()
        self.device = device

    def input_diversity(self, image, low=270, high=299):
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(low, high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        return padded

    def attack(self,
               model: nn.Module,
               inputs: torch.Tensor,
               labels_true: torch.Tensor) -> torch.Tensor:

        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # setup optimizer
        optimizer = optim.SGD([delta], lr=1, momentum=0.9)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)


        for step in range(self.steps):
            if self.max_norm:
                delta.data.clamp_(-self.max_norm, self.max_norm)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)

            adv = inputs + delta
            div_adv = self.input_diversity(adv)

            ######################
            img0=adv.cuda()
            img1=inputs.cuda()
            dist01 = self.loss_fn(img0, img1)
            lpipscore=dist01.sum()/batch_size
                # print("lpips=",lpipscore)
            #######################

            logits = model(div_adv)

            ce_loss_true = F.cross_entropy(logits, labels_true, reduction='none')
            # ce_loss_target = F.cross_entropy(logits, labels_target, reduction='none')

            # fuse targeted and untargeted
            loss = abs(lpipscore-0.2)*self.loss_amp - ce_loss_true
            #  loss = max(lpipscore,0.2)*10*self.loss_amp - ce_loss_true

            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]

            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            # if step == self.steps-1:
            #     print("img0:", img0.shape, "img1:", img1.shape)
            #     print("step:", str(step),": loss = ",str(loss.item()),": lpipscore = ",str(lpipscore.item()),": ce_loss_true = ",str(torch.mean(ce_loss_true).item()))
            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

        return inputs + best_delta

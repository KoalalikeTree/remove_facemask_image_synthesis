# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np
import copy
import gc

from LBFGS import FullBatchLBFGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19
from utils import Normalization
import glob
import PIL.Image as Image
import time

from dataloader import get_data_loader
from torchvision import transforms


def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        pretrain = torch.load(model_path)
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        # model_path = 'pretrained/%s.ckpt' % name
        model_path = './stylegan/pretrain/stylegan2-ffhq-config-e.pkl'
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
         return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""
    def __init__(self, args, model, z_dim):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style = args.model == 'stylegan'

    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            if self.latent == 'w':
                param = param.repeat(1, self.model.mapping.num_ws, 1)
            image = self.model.synthesis(param)
        return image

class PerceptualLoss():
    def __init__(self, layer):
        pass

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
        # raise NotImplementedError()

    def forward(self, img):
        # normalize img
        return (img-self.mean)/self.std

class Criterion(nn.Module):
    def __init__(self, args, apply_mask=False, layer=['conv_1']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.apply_mask = apply_mask
        self.perc = PerceptualLoss(layer)
        vgg = vgg19(pretrained=True).features.to(device).eval()

        # normalization
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        normalization = Normalization(norm_mean,norm_std)
        self.model = nn.Sequential(normalization)

        # load sketch
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # total_sketches = glob.glob("data/sketch/5.png")
        # img_loc = total_sketches[0]
        # sketch = Image.open(img_loc)
        # sketch = sketch.resize((64, 64))
        # rgb = sketch.convert("RGB")
        # self.sketch = self.transform(rgb).to(device)

        i = 0
        for vgglayer in vgg.children():
            if isinstance(vgglayer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(vgglayer, nn.ReLU):
                name = 'relu_{}'.format(i)
                vgglayer = nn.ReLU(inplace=False)
            elif isinstance(vgglayer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(vgglayer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(vgglayer.__class__.__name__))
            self.model.add_module(name, vgglayer)
            if name == layer[0]:
                break
        print("Criteria Model", self.model)

    def gram_matrix(self, activations):
        a, b, c, d = activations.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = activations.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        # 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        normalized_gram = G.div(a * b * c * d)

        return normalized_gram

    def forward(self, pred, target, mask):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        loss = 0

        # if self.apply_mask:
        #     target, mask = target
        #     target = target.detach()
        #     # todo: loss with mask
        #     loss = torch.norm(torch.mul(mask, self.sketch)-torch.mul(mask, pred))
        # else:
        # todo: loss w/o mask
        # target_feature = self.model(target.detach())
        target_feature = self.model(torch.mul(target.detach(), mask))
        target_gram = self.gram_matrix(target_feature)
        pred_feature = self.model(torch.mul(pred, mask))
        pred_gram = self.gram_matrix(pred_feature)
        loss += (1-self.perc_wgt) * F.mse_loss(torch.mul(pred, mask), torch.mul(target.detach(), mask))
        loss += self.perc_wgt*F.mse_loss(target_gram, pred_gram)
        return loss

def save_images(image, fname, col=8):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    imageio.mimsave(fname + '.gif', image_list)


def sample_noise(dim, device, latent, modelType, model, N=1, from_mean=False):
    z = torch.randn(N, dim, device=device)
    zs = []
    w = torch.zeros(N, dim, device=device)
    if latent == 'w+':
        w = torch.zeros((N, 10, dim), device=device)
    n = 50

    if latent == 'z':
        return z
    else:
        # w / w+
        if from_mean:
            for i in range(n):
                zs.append(torch.randn(N, dim, device=device))
            if modelType == 'vanilla':
                for one_z in zs:
                    w += one_z/n
            elif modelType == 'stylegan':
                # load pkl model from HW3 and pass z into it get the w
                for one_z in zs:
                    temp_w = model.mapping(one_z.to(device), None)
                    if latent == 'w+':
                        w += temp_w / n
                    elif latent == 'w':
                        w += torch.mean(temp_w, dim=1, keepdim=True)/n

            # todo: map a bunch of z, take their mean of w / w+.
            #  To see how to pass stylegan2, refer to stylegan/generate_gif.py L70:81
            # dummy:
            # w = torch.randn(N, dim, device=device)
        else:
            # todo: take a random z, map it to w / w+
            #  To see how to pass stylegan2, refer to stylegan/generate_gif.py L70:81
            if modelType == 'vanilla':
                w = z
            elif modelType == 'stylegan':
                w = model.mapping(z.to(device), None)
                if latent == 'w+':
                    pass
                elif latent == 'w':
                    w = torch.mean(w,dim=1,keepdim=True)
        return w

def optimize_para(mask, wrapper, param, target, criterion, num_step, save_prefix=None, res=False):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    param = param.requires_grad_().to(device)
    optimizer = FullBatchLBFGS([param], lr=.1, line_search='Wolfe')
    iter_count = [0]

    def closure():
        # todo: your optimiztion
        param.data.clamp_(0, 1)

        optimizer.zero_grad()
        image = wrapper(param)
        loss = criterion(image, target, mask)
        loss.backward(retain_graph=True)

        if iter_count[0] % 250 == 0 and save_prefix is not None:
            # visualization code
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            iter_result = image.data.clamp_(-1, 1)
            save_images(iter_result, save_prefix + '_%d' % iter_count[0])

        del image
        gc.collect()
        iter_count[0] += 1
        return loss

    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

    image = wrapper(param)
    return param, image

def sample(args):
    # 获取模型
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16

    # todo: complete sample_noise and wrapper
    # 生成z / w / w+
    noise = sample_noise(z_dim, device, args.latent, args.model, model, batch_size)
    # G(z/w/w+)
    image = wrapper(noise)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    loader = get_data_loader(args.input, args.nomask,  is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper_image = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    # todo: implement your criterion here.
    criterion = Criterion(args)
    # project each image
    for idx, (data, mask) in enumerate(loader):
        target = data.to(device)
        mask = (mask[:,0,:,:] > 0.5).to(device)
        save_images(mask, 'output/project/%d_mask' % idx, 1)
        save_images(data, 'output/project/%d_data' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, args.model, model)
        optimize_para(mask, wrapper_image, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g' % (idx, args.model, args.latent, args.perc_wgt))
        if idx >= 1:
            break

def draw(args):
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, alpha=True)
    criterion = Criterion(args, apply_mask=True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb , 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        # todo: optimize sketch 2 image
        param = sample_noise(z_dim, device, args.latent, args.model, model)
        optimize_para(wrapper, param, (rgb, mask), criterion, args.n_iters,
                      'output/draw/%d_%s_%s_%g' % (idx, args.model, args.latent, args.perc_wgt))
        if idx >= 0:
            break

def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input)
    criterion = Criterion(args)
    for idx, (image, _) in enumerate(loader):
        save_images(image, 'output/interpolate/%d' % (idx))
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, args.model, model, from_mean=True)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters)
        save_images(recon, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx % 2 == 0:
            src = param
            continue
        dst = param
        image_list = []
        with torch.no_grad():
            # todo: interpolation code
            for i in range(20):
                inter_param = src*(i*0.05)+dst*(1-i*0.05)
                image_list.append(wrapper(inter_param))

        save_gifs(image_list, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='project', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='w+', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0.1, help="perc loss lambda")
    parser.add_argument('--input', type=str, default='./data/ffhq_mask/*.jpg', help="path to the input image")
    parser.add_argument('--nomask', type=str, default='./data/mask/*.jpg', help="path to the input image")
    return parser.parse_args()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_arg()
    device = ""
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)


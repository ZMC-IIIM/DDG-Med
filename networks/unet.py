# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from networks.dsbn import DomainSpecificBatchNorm2d
from networks.mixstyle import DSU_MixStyle


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True  # Train: True, Test: False
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # Sample mu and var from an uniform distribution, i.e., mu ～ U(0.0, 1.0), var ～ U(0.0, 1.0)
        mu_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)
        var_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)

        lmda = self.beta.sample((N, C, 1, 1))
        bernoulli = torch.bernoulli(lmda).to(x.device)

        mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
        sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
        return x_normed * sig_mix + mu_mix


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def normalization(planes, norm='gn', num_domains=None):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z


class ConvDDynamic_v1(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvDDynamic, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)


        self.sf = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)


        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)

        dyres = self.conv_s1(x)*y[:,0] + self.conv_s2(x)*y[:,1] + \
            self.conv_s3(x)*y[:,2] + self.conv_s4(x)*y[:,3]
        out = dyres + self.conv2(x)

        #layer 2 conv, bn, relu
        # y = self.conv2(x)
        y = self.bn2(out)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        z += residual #+ dyres
        z = self.relu(z)

        return z


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        bs, _, height, width = x.size()
        # 生成坐标网格
        coord_tensor = torch.zeros(bs, 2, height, width, device=x.device)

        # for i in range(height):
        #     for j in range(width):
        #         coord_tensor[:, 0, i, j] = i/(height-1)
        #         coord_tensor[:, 1, i, j] = j/(width-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=x.device), torch.linspace(0, 1, width, device=x.device))
        coord_tensor[:, 0, :, :] = yy
        coord_tensor[:, 1, :, :] = xx

        x = torch.cat((x, coord_tensor), dim=1)
        grid = self.conv(x) # torch.Size([1, 3, 3, 3])

        return grid


class ConvDDynamic(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvDDynamic, self).__init__()

        self.first = first
        self.in_channels = inplanes

        self.mixstyle = DSU_MixStyle(p=0.5, alpha=0.1)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2  = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3  = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU(inplace=True)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)


    def forward(self, x):

        if not self.first:
            x = self.maxpool2D(x)

        x = self.mixstyle(x)
        x = self.conv1(x)
        out = self.bn1(x)


        #layer 2 conv, bn, relu
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        #layer 3 conv, bn
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)
        return out


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        return y


class ConvUDynamic(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvUDynamic, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1  = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.CoordConv = CoordConv2d(2, 2, kernel_size=1, padding=0)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

        self.squeeze = planes // 16
        self.sf = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            nn.Linear(planes, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        b, c, h, w = x.size()
        y1 = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y1 = self.sf(y1)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)


        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        dyres = self.conv_s1(y)*y1[:, 0] + self.conv_s2(y)*y1[:, 1] + \
            self.conv_s3(y)*y1[:, 2] + self.conv_s4(y)*y1[:, 3]

        y = self.conv3(y)+dyres
        y = self.CoordConv(y)

        #layer 3 conv, bn
        # y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)
        return y


class Unet2D(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet2D, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n, 16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        y1_pred = self.seg1(y1)
        return y1_pred


class Encoder(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', activation='relu'):
        super(Encoder, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n, 16*n, norm, activation=activation)

        self.random = TriD(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)

        x2 = self.convd2(x1)

        x3 = self.convd3(x2)

        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        return [x1, x2, x3, x4, x5]


class EncoderDynamic(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', activation='relu'):
        super(EncoderDynamic, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvDDynamic(n,   2*n, norm, activation=activation)
        self.convd3 = ConvDDynamic(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n, 16*n, norm, activation=activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, n=16, num_classes=2, norm='bn', activation='relu'):
        super(Decoder, self).__init__()
        self.convu4 = ConvU(16 * n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8 * n, norm, activation=activation)
        self.convu2 = ConvU(4 * n, norm, activation=activation)
        self.convu1 = ConvU(2 * n, norm, activation=activation)

        self.out1 = nn.Conv2d(2 * n, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        y4 = self.convu4(feats[-1], feats[-2])
        y3 = self.convu3(y4, feats[-3])
        y2 = self.convu2(y3, feats[-4])
        y1 = self.convu1(y2, feats[-5])
        y1_pred = self.out1(y1)
        return y1_pred


class DecoderDynamic(nn.Module):
    def __init__(self, n=16, num_classes=2, norm='bn', activation='relu'):
        super(DecoderDynamic, self).__init__()
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.CoordConv = CoordConv2d(2*n, 2*n, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(2*n, n, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n, 4, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(2*n, 2*n, 1, stride=1, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(2*n, 2*n, 1, stride=1, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(2*n, 2*n, 1, stride=1, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(2*n, 2*n, 1, stride=1, padding=0, bias=False)

        self.out1 = nn.Conv2d(2*n, num_classes, 3, padding=1)
        self.out2 = nn.Conv2d(4*n, num_classes, 3, padding=1)
        self.conv2 = nn.Conv2d(2*n, 2*n, 3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feats):
        y4 = self.convu4(feats[-1], feats[-2])
        y3 = self.convu3(y4, feats[-3])
        y2 = self.convu2(y3, feats[-4])
        y1 = self.convu1(y2, feats[-5])

        b, c, h, w = y1.size()
        y = self.fc(self.avg_pool(y1).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        dyres = self.conv_s1(y1)*y[:, 0] + self.conv_s2(y1)*y[:, 1] + \
            self.conv_s3(y1)*y[:, 2] + self.conv_s4(y1)*y[:, 3]
        y1 = dyres + self.conv2(y1)

        y1 = self.CoordConv(y1)

        # y1_new = torch.cat([y1, y1_coord], 1)

        y1_pred = self.out1(y1)
        # y1_new = self.out2(y1_new)
        # y1_pred = torch.add(y1_new, y1_pred)
        return y1_pred


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, n=16):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, n, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(n, 2*n, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(2*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(2*n, 4*n, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(4*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(4*n, 8*n, 4, padding=1),
                    nn.InstanceNorm2d(8*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(8*n, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
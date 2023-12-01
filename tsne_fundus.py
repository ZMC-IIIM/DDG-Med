#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn as nn

import torch
from tqdm import tqdm
from dataset.fundus import Fundus, Fundus_source
from torch.utils.data import DataLoader
import dataset.transform as trans
from torchvision.transforms import Compose
from utils.metrics import *
from dataset import utils
from utils.utils import postprocessing, save_per_img
from test_utils import *
from networks.unet import Encoder, DecoderDynamic
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel
import torch.nn.functional as F
from analysis import tsne, a_distance

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Test on Fundus dataset (2D slice)')
    # basic settings
    parser.add_argument('--model_file', type=str, default=None, required=True, help='Model path')
    parser.add_argument('--dataset', type=str, default='fundus', help='training dataset')
    parser.add_argument('--data_dir', default='../../../DG/DoFE/dataset', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--test_prediction_save_path', type=str, default=None, required=True, help='Path root for test image and mask')
    parser.add_argument('--save_result', action='store_true', help='Save Results')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default='3', help='training epochs')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze Batch Normalization')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def main(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(args.test_prediction_save_path):
        os.makedirs(args.test_prediction_save_path)

    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    transform = Compose([trans.Resize((256, 256)), trans.Normalize()])

    testset = Fundus(base_dir=data_dir, split='test', domain_idx=args.datasetTest, transform=transform)
    
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)

    #vis
    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]
    trainset = Fundus_source(base_dir=data_dir, split='test',
                                         domain_idx_list=domain_idx_list, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)

    encoder = Encoder(c=args.in_channels, norm=args.norm, activation=args.activation)
    seg_decoder = DecoderDynamic(num_classes=args.num_classes, norm=args.norm, activation=args.activation)
    
    state_dicts = torch.load(model_file)
    
    encoder.load_state_dict(state_dicts['encoder_state_dict'])
    seg_decoder.load_state_dict(state_dicts['seg_decoder_state_dict'])
    
    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()

    if not args.freeze_bn:
        encoder.eval()
        for m in encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        seg_decoder.eval()
        for m in seg_decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        encoder.eval()
        seg_decoder.eval()

    tbar = tqdm(testloader, ncols=150)
    tbar_source = tqdm(trainloader, ncols=150)

    all_features_S = []
    all_features_T = []
    # all_features_aug = []
    with torch.no_grad():
        for i, (data, target, target_orgin, ids) in enumerate(tbar_source):
            data,  target = data.cuda(), target.cuda()
            feature = seg_decoder(encoder(data)).cpu()
            feature = feature.view(feature.size(0), -1)
            all_features_S.append(feature)
            # freq_feature = seg_decoder(encoder(freq_data)).cpu()
            # freq_feature = feature.view(freq_feature.size(0), -1)
            # all_features_S.append(freq_feature)
    source_feature = torch.cat(all_features_S, dim=0)
    # aug_feature = torch.cat(all_features_aug, dim=0)
    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()
            #vis
            feature = seg_decoder(encoder(data)).cpu()
            feature = feature.view(feature.size(0), -1)
            all_features_T.append(feature)
    target_feature = torch.cat(all_features_T, dim=0)

    # plot t-SNE
    tSNE_filename = os.path.join("results", "FFT_DeepAll.png")
    # osp.join(logger.visualize_directory, 'TSNE.pdf')
    tsne.visualize(source_feature, target_feature, tSNE_filename)
    print("Saving t-SNE to", tSNE_filename)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
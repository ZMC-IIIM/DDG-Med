import os
import argparse
import numpy as np
from networks.unet import Encoder, DecoderDynamic
from utils.utils import count_params
from tensorboardX import SummaryWriter
import random
import dataset.transform as trans
from torchvision.transforms import Compose

from dataset.fundus import Fundus_Multi_FFT_class, Fundus
from dataset.prostate import Prostate_Multi
import torch.backends.cudnn as cudnn

from torch.nn import BCELoss, CrossEntropyLoss, DataParallel, KLDivLoss, MSELoss
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.losses import dice_loss, dice_loss_multi
from utils.utils import decode_seg_map_sequence
import shutil
from utils.utils import postprocessing, _connectivity_region_analysis
from utils.metrics import *
import os.path as osp
import SimpleITK as sitk
from medpy.metric import binary
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')


batch_size = 16


def parse_args():
    parser = argparse.ArgumentParser(description='DG Medical Segmentation Train')
    # basic settings
    parser.add_argument('--data_root', type=str, default='../../../DG/DoFE/dataset', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'], help='training dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default=3, help='training epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='path of saved checkpoints')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def KD(input, target):
    consistency_criterion = KLDivLoss()
    loss_consistency = consistency_criterion(input, target) + consistency_criterion(target, input)
    return loss_consistency


def test_fundus(encoder, seg_decoder, epoch, data_dir, datasetTest, output_path, batch_size=8, dataset='fundus'):
    encoder.eval()
    seg_decoder.eval()
    data_dir = os.path.join(data_dir, dataset)
    transform = Compose([trans.Resize((256, 256)), trans.Normalize()])
    testset = Fundus(base_dir=data_dir, split='test',
                                         domain_idx=datasetTest, transform=transform)
    
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)
    
    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_num = 0
    tbar = tqdm(testloader, ncols=150)

    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()
            feature = encoder(data)
            prediction = torch.sigmoid(seg_decoder(feature))
            prediction = torch.nn.functional.interpolate(prediction, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")

            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=dataset, threshold=0.75)
                cup_dice, disc_dice = dice_coeff_2label(prediction_post, target_orgin[i])
                val_cup_dice += cup_dice
                val_disc_dice += disc_dice
                total_num += 1
        val_cup_dice /= total_num
        val_disc_dice /= total_num

        print('val_cup_dice : {}, val_disc_dice : {}'.format(val_cup_dice, val_disc_dice))
        
        return (val_cup_dice + val_disc_dice) * 100.0 / 2


def train_fundus(trainloader_list, encoder, seg_decoder, args, optimizer, dataloader_length_max):

    criterion = BCELoss()

    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()
    consistency_criterion = MSELoss()
    total_iters = dataloader_length_max * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        
        encoder.train()
        seg_decoder.train()

        tbar = tqdm(trainloader_list, ncols=150)

        for i, sample_batches in enumerate(tbar):

            # img, mask = sample_batches
            # img_multi,  mask_multi = img.cuda(), mask.cuda()
            #
            img, img_freq, mask = sample_batches
            img_multi, img_freq_multi, mask_multi = img.cuda(), img_freq.cuda(), mask.cuda()

            img_feats = encoder(img_multi)
            pred_soft_1 = torch.sigmoid(seg_decoder(img_feats))
            loss_bce_1 = criterion(pred_soft_1, mask_multi)
            loss_dice_1 = dice_loss(pred_soft_1, mask_multi)

            img_freq_feats = encoder(img_freq_multi)
            pred_soft_2 = torch.sigmoid(seg_decoder(img_freq_feats))
            loss_bce_2 = criterion(pred_soft_2, mask_multi)
            loss_dice_2 = dice_loss(pred_soft_2, mask_multi)

            #loss_consistency = consistency_criterion(pred_soft_2, pred_soft_1)

            loss = loss_bce_1 + loss_dice_1 + loss_bce_2 + loss_dice_2
            # loss = loss_bce_1 + loss_dice_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr

            # writer.add_scalar('lr', lr, iter_num)
            # writer.add_scalar('loss/loss_bce_1', loss_bce_1, iter_num)
            # writer.add_scalar('loss/loss_dice_1', loss_dice_1, iter_num)
            
            iter_num = iter_num + 1

        if (epoch + 1) % 10 == 0:
          with torch.no_grad():
            avg_dice = test_fundus(encoder, seg_decoder, epoch, args.data_root,
                                       args.test_domain_idx,  args.save_path,
                                       args.test_batch_size, dataset=args.dataset)
            if avg_dice >= previous_best:
                if previous_best != 0:
                    model_path = os.path.join(args.save_path, 'model_%.2f.pth' % (previous_best))
                    if os.path.exists(model_path):
                        os.remove(model_path)

                checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                  "seg_decoder_state_dict": seg_decoder.module.state_dict()}
                torch.save(checkpoint, os.path.join(args.save_path, 'model_%.2f.pth' % (avg_dice)))
                previous_best = avg_dice
                
    save_mode_path = os.path.join(args.save_path, 'final_model_BN16_FFT_DyClass.pth')

    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                      "seg_decoder_state_dict": seg_decoder.module.state_dict()}
    torch.save(checkpoint, save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))


def main(args):

    #数据文件检查
    data_root = os.path.join(args.data_root, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    #运行点记录文件
    writer = SummaryWriter(args.save_path + '/log')

    #数据集加载
    dataset_zoo = {'fundus': Fundus_Multi_FFT_class, 'prostate': Prostate_Multi}
    #数据变换
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.RandomScaleCrop((256, 256))]),
                 'prostate': None}


    # 训练数据集列表
    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]


    #所以训练域数据集加载
    trainset = dataset_zoo[args.dataset](base_dir=data_root, split='train',
                                         domain_idx_list=domain_idx_list, transform=transform[args.dataset],
                                         test_domain_idx=args.test_domain_idx)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

    dataloader_length_max = len(trainloader)

    encoder = Encoder(c=args.in_channels, norm=args.norm, activation=args.activation)
    seg_decoder = DecoderDynamic(num_classes=args.num_classes, norm=args.norm, activation=args.activation)

    optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr},
                          {"params": seg_decoder.parameters(), 'lr': args.lr}],
                          lr=args.lr, betas=(0.9, 0.999))

    print('\nEncoder Params: %.3fM' % count_params(encoder))
    print('\nSeg Decoder Params: %.3fM' % count_params(seg_decoder))

    if args.dataset == 'fundus':
        train_fundus(trainloader, encoder, seg_decoder,
                     args, optimizer, dataloader_length_max)
    else:
        raise ValueError('Not support Dataset {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:
        args.epochs = {'fundus': 400, 'prostate': 200}[args.dataset]
    if args.lr is None:
        args.lr = {'fundus': 2e-3, 'prostate': 1e-3}[args.dataset]
    if args.num_classes is None:
        args.num_classes = {'fundus': 2, 'prostate': 2}[args.dataset]

    print(args)

    main(args)
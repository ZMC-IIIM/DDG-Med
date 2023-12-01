import os
import random
import numpy as np

from PIL import Image
from dataset.transform import to_multilabel
from scipy.special import comb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.randint(1, 10) / 10

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg,  L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)


class Fundus(Dataset):
    def __init__(self, domain_idx=None, base_dir=None, split='train', num=None, transform=None, is_ra=False):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx = domain_idx
        self.split = split
        self.is_ra = is_ra
        self.id_path = []
        if split == 'test':
            with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'train.list'), 'r') as f:
                self.id_path = self.id_path + f.readlines()
        # elif split == 'test':
            with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'test.list'), 'r') as f:
                self.id_path = self.id_path + f.readlines()
        
        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        # cur_domain_name = self.domain_name[self.domain_idx]
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir,  id.split(' ')[0]))

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}
            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            # crop during test #
            # mask = mask[..., 144:144+512, 144:144+512]

            if self.transform:
                sample = self.transform(sample)
            return sample['img'], sample['mask'], mask, id
        
        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['img'], sample['mask']


class Fundus_Multi(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_freq=False,  test_domain_idx=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_freq = is_freq
        self.test_domain_idx = test_domain_idx

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        # elif split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
        
        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])

        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()

            if self.transform:
                sample = self.transform(sample)

            return sample['img'], sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
        
        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        img = sample['img']
        mask = sample['mask']
        img = np.array(img).astype(np.float32)
            
        img /= 127.5
        img -= 1.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        __mask = np.array(mask).astype(np.uint8)
        _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        _mask[__mask > 200] = 255
        _mask[(__mask > 50) & (__mask < 201)] = 128
        _mask[(__mask > 50) & (__mask < 201)] = 128

        __mask[_mask == 0] = 2
        __mask[_mask == 255] = 0
        __mask[_mask == 128] = 1

        mask = to_multilabel(__mask)
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(np.array(mask)).float()
        return img, mask


class Fundus_Multi_FFT(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_freq=True,
                 is_out_domain=False, test_domain_idx=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_freq = is_freq
        self.is_out_domain = is_out_domain
        self.test_domain_idx = test_domain_idx

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])),
                          'r') as f:
                    self.id_path = self.id_path + f.readlines()

            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))
        cur_domain_name = id.split(' ')[0].split('/')[0]

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()

            if self.transform:
                sample = self.transform(sample)

            return sample['img'], sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        if self.is_freq:
            img = sample['img']
            mask = sample['mask']

            domain_list = train_domain_name.copy()
            if self.is_out_domain:
                domain_list.remove(cur_domain_name)
            print(domain_list)

            with open(os.path.join(self.base_dir, other_domain_name, 'train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]

            # other_id = np.random.choice(self.id_path).replace('\n', '').split(' ')[0]
            other_img = Image.open(os.path.join(self.base_dir, other_id)).resize((256, 256), Image.BILINEAR)
            other_img = np.array(other_img).astype(np.float32)

            img = np.array(img).astype(np.float32)
            amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))
            img_freq = source_to_target_freq(img, amp_trg, L=0.1)
            img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

            img /= 127.5
            img -= 1.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()

            img_freq = img_freq.transpose(2, 0, 1)
            img_freq = torch.from_numpy(img_freq).float()
            img_freq /= 127.5
            img_freq -= 1.0

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            return img, img_freq, mask
        else:
            return sample['img'], sample['mask']


class Fundus_source(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_ra=False):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_ra = is_ra
        self.id_path = []
        if split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir, self.domain_name[domain_idx], 'train.list'), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
            # elif split == 'test':
                with open(os.path.join(self.base_dir, self.domain_name[domain_idx], 'test.list'), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        # cur_domain_name = self.domain_name[self.domain_idx]
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}
            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            # crop during test #
            # mask = mask[..., 144:144+512, 144:144+512]

            if self.transform:
                sample = self.transform(sample)
            return sample['img'], sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['img'], sample['mask']


class Fundus_source_aug(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_ra=False,test_domain_idx=3):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.test_domain_idx = test_domain_idx
        self.split = split
        self.is_ra = is_ra
        self.id_path = []
        if split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir, self.domain_name[domain_idx], 'train.list'), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
            # elif split == 'test':
                with open(os.path.join(self.base_dir, self.domain_name[domain_idx], 'test.list'), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])

        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))
        cur_domain_name = id.split(' ')[0].split('/')[0]
        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}
            if self.transform:
                sample = self.transform(sample)
            img = sample['img']

            domain_list = train_domain_name.copy()

            train_domain_name.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name, 'train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
            other_img = Image.open(os.path.join(self.base_dir, other_id)).resize((256, 256),
                                                                                         Image.BILINEAR)
            other_img = np.array(other_img).astype(np.float32) #256, 256, 3)
            img = np.array(img).astype(np.float32).transpose(1, 2, 0) #(3, 256, 256)
            amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))
            img_freq = source_to_target_freq(img, amp_trg, L=0.1)
            img_freq = np.clip(img_freq, 0, 255).astype(np.float32)
            img_freq = img_freq.transpose(2, 0, 1)
            img_freq = torch.from_numpy(img_freq).float()

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            return sample['img'], img_freq, sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['img'], sample['mask']


class Fundus_Multi_F(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_freq=True,
                 is_out_domain=False, test_domain_idx=None):
        self.transform = transform
        self.base_dir = base_dir
        # self.i = epoch
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_freq = is_freq
        self.is_out_domain = is_out_domain
        self.test_domain_idx = test_domain_idx

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])),
                          'r') as f:
                    self.id_path = self.id_path + f.readlines()

            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])

        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()

            if self.transform:
                sample = self.transform(sample)

            return sample['img'], sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        if self.is_freq:
            img = sample['img']
            mask = sample['mask']

            other_id_1 = np.random.choice(self.id_path).replace('\n', '').split(' ')[0]
            other_img1 = Image.open(os.path.join(self.base_dir, other_id_1)).resize((256, 256), Image.BILINEAR)
            other_array_1 = np.array(other_img1).astype(np.float32)
            other_array_1 /= 127.5
            other_array_1 -= 1.0
            mean_1 = np.mean(other_array_1)
            std_1 = np.std(other_array_1)

            other_id_2 = np.random.choice(self.id_path).replace('\n', '').split(' ')[0]
            other_img2 = Image.open(os.path.join(self.base_dir, other_id_2)).resize((256, 256), Image.BILINEAR)
            other_array_2 = np.array(other_img2).astype(np.float32)
            other_array_2 /= 127.5
            other_array_2 -= 1.0
            mean_2 = np.mean(other_array_2)
            std_2 = np.std(other_array_2)

            img_array = np.array(img).astype(np.float32)
            img_array /= 127.5
            img_array -= 1.0
            mean = np.mean(img_array)
            std = np.std(img_array)

            a = np.power(mean - mean_1, 2) + np.power(std - std_1, 2)
            b = np.power(mean - mean_2, 2) + np.power(std - std_2, 2)

            img = np.array(img).astype(np.float32)

            if a >= b:
                amp_trg = extract_amp_spectrum(np.array(other_img1).astype(np.float32).transpose(2, 0, 1))
                img_freq = source_to_target_freq(img, amp_trg, L=0.1)
                img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

            else:
                amp_trg = extract_amp_spectrum(np.array(other_img2).astype(np.float32).transpose(2, 0, 1))
                img_freq = source_to_target_freq(img, amp_trg, L=0.1)
                img_freq = np.clip(img_freq, 0, 255).astype(np.float32)


            img_freq /= 127.5
            img_freq -= 1.0
            img_freq = img_freq.transpose(2, 0, 1)
            img_freq = torch.from_numpy(img_freq).float()

            img /= 127.5
            img -= 1.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            return img, img_freq, mask
        else:
            return sample['img'], sample['mask']


class Fundus_Multi_FFT_class(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_freq=True,
                 is_out_domain=False, test_domain_idx=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_freq = is_freq
        self.is_out_domain = is_out_domain
        self.test_domain_idx = test_domain_idx

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_train.list".format(self.domain_name[domain_idx])),
                          'r') as f:
                    self.id_path = self.id_path + f.readlines()

            for domain_idx in self.domain_idx_list:
                with open(os.path.join(self.base_dir + "/{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])

        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))
        cur_domain_name = id.split(' ')[0].split('/')[0]

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()

            if self.transform:
                sample = self.transform(sample)

            return sample['img'], sample['mask'], mask, id

        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        if self.is_freq:
            img = sample['img']
            mask = sample['mask']

            domain_list = train_domain_name.copy()
            domain_list.remove(cur_domain_name)
            other_domain_name = np.random.choice(domain_list, 1)[0]
            with open(os.path.join(self.base_dir, other_domain_name, 'train.list'), 'r') as f:
                other_id_path = f.readlines()
            other_id = np.random.choice(other_id_path).replace('\n', '')
            img_id = other_id.split(' ')[0]
            mask_id = other_id.split(' ')[1]
            other_img = Image.open(os.path.join(self.base_dir, img_id)).resize((256, 256), Image.BILINEAR)
            other_mask = Image.open(os.path.join(self.base_dir, mask_id)).resize((256, 256), Image.NEAREST)

            other_img = np.array(other_img).astype(np.float32)
            other_mask = np.array(other_mask).astype(np.uint8)

            # global FFT
            img = np.array(img).astype(np.float32)
            amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))
            img_freq_gl = source_to_target_freq(img, amp_trg, L=0.1)
            img_freq_gl = np.clip(img_freq_gl, 0, 255).astype(np.float32)

            # local FFT
            im_bg = img.copy()
            ot_bg = other_img.copy()
            im_bg[mask != 255] = 0
            ot_bg[other_mask != 255] = 0

            im_disc = img.copy()
            ot_disc = other_img.copy()
            im_disc[mask == 255] = 0
            ot_disc[other_mask == 255] = 0

            # im_cup = img.copy()
            # ot_cup = other_img.copy()
            # im_cup[mask != 0] = 0
            # ot_cup[other_mask != 0] = 0

            ma_bg = np.ones_like(img)
            ma_bg[mask != 255] = 0
            amp_trg_bg = extract_amp_spectrum(ot_bg.transpose(2, 0, 1))
            img_freq_bg = source_to_target_freq(im_bg, amp_trg_bg, L=0.1)
            img_freq_bg = np.clip(img_freq_bg, 0, 255).astype(np.float32)
            img_freq_bg = img_freq_bg * ma_bg

            ma_disc = np.ones_like(img)
            ma_disc[mask == 255] = 0
            amp_trg_disc = extract_amp_spectrum(ot_disc.transpose(2, 0, 1))
            img_freq_disc = source_to_target_freq(im_disc, amp_trg_disc, L=0.1)
            img_freq_disc = np.clip(img_freq_disc, 0, 255).astype(np.float32)
            img_freq_disc = img_freq_disc * ma_disc


            # ma_cup = np.ones_like(img)
            # ma_cup[mask != 0] = 0
            # amp_trg_cup = extract_amp_spectrum(ot_cup.transpose(2, 0, 1))
            # img_freq_cup = source_to_target_freq(im_cup, amp_trg_cup, L=0.1)
            # img_freq_cup = np.clip(img_freq_cup, 0, 255).astype(np.float32)
            # img_freq_cup = img_freq_cup * ma_cup

            img_freq_local = img_freq_bg + img_freq_disc

            # 生成随机数，范围为 [0, 1]
            p = random.uniform(0, 1)
            if p > 0.5:
                img_freq = img_freq_local
            else:
                img_freq = img_freq_gl

            img /= 127.5
            img -= 1.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()

            img_freq /= 127.5
            img_freq -= 1.0
            img_freq = img_freq.transpose(2, 0, 1)
            img_freq = torch.from_numpy(img_freq).float()

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            return img, img_freq, mask
        else:
            return sample['img'], sample['mask']


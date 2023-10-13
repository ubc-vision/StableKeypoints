"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License
Copyright (c) 2018 akanazawa
"""

import json
import os.path as osp

import cv2

from utils.utils import pil_loader, pad_if_smaller

cv2.setNumThreads(0)
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode

from utils import image as image_utils

parts = np.array([[  0,   0,   0],
        [  0, 100,   0],
        [ 16,  78, 139],
        [ 50, 205,  50],
        [ 70, 130, 180],
        [127, 255, 212],
        [144, 238, 144],
        [211, 211, 211],
        [220, 220, 220],
        [245, 222, 179],
        [250, 235, 215],
        [255,   0,   0],
        [255, 140,   0],
        [255, 250, 205],
        [255, 250, 250],
        [255, 255,   0]], dtype=int)
gt_classes = ['background', 'eyeglass', 'face', 'accessories', 'leggings', 'headwear', 'skin', 'pants', 'outer', 'footwear', 'skirt', 'hair', 'bag', 'dress', 'top']


class DFDataset(Dataset):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.img_size = opts.input_size
        self.split = opts.split
        self.dataset_root = opts.dataset_root
        self.dataset = 'deepfashion'
        split = opts.split

        self.image_dir = osp.join(self.dataset_root, 'DeepFashion/In-shop Clothes Retrieval Benchmark/')

        if split == 'train':
            self.images = self.only_file_names(json.load(open(f'{self.image_dir}/Anno/segmentation/DeepFashion_segmentation_train.json', 'r'))['images'])
        else:
            val_files = [f'{self.image_dir}/Anno/segmentation/{f}' for f in ['DeepFashion_segmentation_query.json', 'DeepFashion_segmentation_gallery.json']]
            self.images = self.only_file_names(json.load(open(val_files[0], 'r'))['images']) + self.only_file_names(json.load(open(val_files[1], 'r'))['images'])
        self.num_imgs = len(self.images)
        print('%d images' % self.num_imgs)

        sum_colors = parts.sum(1)
        self.col2idx = -np.ones(sum_colors.max() + 1, dtype=np.uint8)
        for i in range(sum_colors.shape[0]):
            self.col2idx[sum_colors[i]] = i

    @staticmethod
    def only_file_names(lst):
        return [e['file_name'] for e in lst]

    def forward_img(self, index):

        path = self.images[index]
        img = pil_loader(self.image_dir + '/Anno/segmentation/' + path, 'RGB')
        mask = pil_loader(self.image_dir + '/Anno/segmentation/' + path.replace('.jpg', '_segment.png'), 'RGB')
        mask = np.array(mask)
        mask = self.col2idx[mask.sum(2)]
        assert -1 not in mask
        mask = Image.fromarray(mask)
        if self.split == 'train':
            i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(img, scale=[0.5, 1.0], ratio=[3. / 4., 4. / 3.])
            img = TF.resized_crop(img, i, j, h, w, size=[self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, size=[self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        img = np.array(img)
        mask = np.array(mask)
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(mask, 2)
        h,w,_ = mask.shape

        if self.split != 'train':
            # scale image, and mask. And scale kps.
            img, mask = self.scale_image(img, mask)

        # Mirror image on random.
        if self.split == 'train':
           img, mask = self.mirror_image(img, mask)

        img = Image.fromarray(img)
        mask = np.asarray(mask, np.uint8)
        return img, mask, path


    def scale_image(self, img, mask):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale, interpolation=cv2.INTER_NEAREST)
        mask_scale = np.expand_dims(mask_scale, 2)

        img_scale = pad_if_smaller(img_scale, self.img_size)
        mask_scale = pad_if_smaller(mask_scale, self.img_size, fill=0)
        return img_scale, mask_scale

    def mirror_image(self, img, mask):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            return img_flip, mask_flip
        else:
            return img, mask

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, seg, img_path = self.forward_img(index)
        mask = (seg != 0).astype(np.uint8)
        elem = {
            'img': img,
            'mask': mask,
            'seg': seg,
            'inds': index,
            'img_path': img_path,
        }
        return elem
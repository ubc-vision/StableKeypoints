r"""SPair-71k dataset"""
import json
import glob
import os

from PIL import Image
import numpy as np
import torch

from .dataset import random_crop


class SPairDataset():
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, augmentation, feature_size):
        

    def __getitem__(self, idx):
        # idx = 5125
        r"""Constructs and return a batch for SPair-71k dataset"""
        batch = {}
        
        
        mini_batch['src_kps']
        mini_batch['og_src_img']
        mini_batch['og_trg_img']

        
        batch['src_bbox_og'] = None
        batch['trg_bbox_og'] = None
        

        return batch

    def get_image(self, img_names, idx):
        r"""Returns image tensor"""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_bbox(self, bbox_list, idx, imsize):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.imside / imsize[0])
        bbox[1::2] *= (self.imside / imsize[1])
        return bbox

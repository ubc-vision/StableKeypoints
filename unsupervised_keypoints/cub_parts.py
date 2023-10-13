"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License

Copyright (c) 2018 akanazawa
"""

import os.path as osp

import cv2
import math
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import functional as TF, InterpolationMode

padding_frac = 0.05
jitter_frac = 0.05

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def resize_img(img, scale_factor, interpolation=None):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=interpolation)
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


def peturb_bbox(bbox, pf=0, jf=0):
    '''
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[1] -= (pf*bheight) + (1-2*np.random.random())*jf*bheight
    pet_bbox[2] += (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[3] += (pf*bheight) + (1-2*np.random.random())*jf*bheight

    return pet_bbox


def square_bbox(bbox):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))
    
    dw_b_2 = int(round((maxdim-bwidth)/2.0))
    dh_b_2 = int(round((maxdim-bheight)/2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1
    
    return sq_bbox

    
def crop(img, bbox, bgval=0):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image        
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]
    
    img_out = np.ones((bheight, bwidth, nc))*bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2]+1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3]+1)
    
    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out


def compute_dt(mask):
    """
    Computes distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1-mask) / max(mask.shape)
    return dist

def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(1-mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1. / (1 + np.exp(k * -dist_diff))
    return dist

class CUBDataset(Dataset):
    def __init__(self, img_size=512, split='train', unsup_mask= False, dataset_root= "/ubc/cs/home/i/iamerich/scratch/datasets/cub/", single_class=None):
        super().__init__()

        self.img_size = img_size
        self.jitter_frac = jitter_frac
        self.padding_frac = padding_frac
        self.split = split
        self.unsup_mask = unsup_mask

        self.data_dir = f'{dataset_root}/CUB_200_2011/'
        self.data_cache_dir = f'{self.data_dir}/cachedir/cub'  # https://github.com/akanazawa/cmr/issues/3#issuecomment-451757610

        self.img_dir = osp.join(self.data_dir, 'images')
        self.pmask_dir = self.data_dir.replace('CUB_200_2011', 'pseudolabels')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % self.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % self.split)
        # import ipdb; ipdb.set_trace()
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import pdb; pdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.labels = [int(self.anno[index].rel_path.split('.')[0]) for index in range(len(self.anno))]
        if single_class is not None:
            idx = [i for i, c in enumerate(self.labels) if c == single_class]
            self.anno = [self.anno[i] for i in idx]
            self.anno_sfm = [self.anno_sfm[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)

    def forward_img(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))
        #img_path = img_path.replace("JPEG", "jpg")
        img = np.array(Image.open(img_path))

        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        if self.unsup_mask:
            if self.split == 'train':
                mask = data.mask
            else:
                mask = TF.resize(torch.from_numpy(np.array(Image.open(osp.join(self.pmask_dir, str(data.rel_path).replace('.jpg', '.png'))))).unsqueeze(0), img.shape[:2], interpolation=InterpolationMode.NEAREST).squeeze(0).numpy()/255.
        else:
            mask = data.mask
        mask = np.expand_dims(mask, 2)
        h,w,_ = mask.shape

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.split == 'train':
            bbox = peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.
        img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.split == 'train':
           img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        img = Image.fromarray(np.asarray(img, np.uint8))
        mask = np.asarray(mask, np.float32)
        return img, kp_norm, mask, sfm_pose, img_path

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = crop(img, bbox, bgval=1)
        mask = crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = resize_img(mask, scale)
        mask_scale, _ = resize_img(mask, scale, interpolation=cv2.INTER_NEAREST)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        kp_perm = self.kp_perm
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # Flip sfm_pose Rot.
            R = quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip, sfm_pose
        else:
            return img, mask, kp, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, kp, mask, sfm_pose, img_path = self.forward_img(index)
        sfm_pose[0].shape = 1
        mask = np.expand_dims(mask, 2)
        
        kpts = (kp[:, :2] + 1)/2
        # swap x and y
        kpts = kpts[:, [1, 0]]
        visibility = kp[:, 2]
        
        img = self.transform(img)

        elem = {
            'img': img,
            'kpts': torch.tensor(kpts),
            'visibility': torch.tensor(visibility),
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'inds': index,
            'label': self.labels[index],
            'img_path': img_path,
        }

        return elem
    
if __name__ == "__main__":
    dataset = CUBDataset(split="train", single_class=1)
    dataset = CUBDataset(split="test", single_class=1)
    
    batch = dataset[2]
    
    
    # display batch['img'] with matplotlib
    # overlay batch['kp']
    import matplotlib.pyplot as plt
    plt.imshow(batch['img'])
    
    kp = (batch['kp']+1)*256
    plt.scatter(kp[:,0], kp[:,1])
    plt.savefig("outputs/test.png")
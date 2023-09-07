import math
import torch
import random
from torchvision.transforms import functional as TF
from torch import Tensor
from typing import Any, List, Optional, Tuple, Union


def _get_inverse_affine_matrix(
    center: List[float],
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    inverted: bool = True,
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def compute_theta(angle, translate, scale, shear, img_size):
    # center = [img_size[0] / 2, img_size[1] / 2]
    center = [0, 0]
    inverted_matrix = _get_inverse_affine_matrix(
        center, angle, translate, scale, shear, inverted=True
    )
    theta = torch.tensor(inverted_matrix).reshape(1, 2, 3).float()
    return theta


def transform_keypoints(keypoints, angle, translate, scale, shear, img_size):
    theta = compute_theta(angle, translate, scale, shear, img_size)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor(
        [0.5 * img_size[1], 0.5 * img_size[0]], dtype=theta.dtype, device=theta.device
    )
    # rescaled_theta = theta.transpose(1, 2)

    # make in the same range as image coordinates
    keypoints_transformed = keypoints - 0.5
    keypoints_transformed = keypoints_transformed * torch.tensor(img_size)[None]

    # Homogenize the keypoints_transformed to apply affine transformation
    keypoints_homogenized = torch.cat(
        (
            keypoints_transformed,
            torch.ones(
                keypoints_transformed.size(0), 1, dtype=keypoints_transformed.dtype
            ),
        ),
        dim=1,
    )

    transformed_keypoints = keypoints_homogenized[None].bmm(rescaled_theta)
    transformed_keypoints = transformed_keypoints.squeeze()
    transformed_keypoints = transformed_keypoints / 2 + 0.5
    # keypoints_transformed = keypoints_transformed / torch.tensor(img_size)[None]
    # transformed_keypoints = transformed_keypoints + 0.5

    return transformed_keypoints


class RandomAffineWithInverse:
    def __init__(self, degrees, scale, translate, shear):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.shear = shear

        # initialize self.last_params to 0s
        self.last_params = {
            "translations_percent": (0, 0),
            "angle": 0,
            "scale": 1.0,
            "shear": (0, 0),
        }

    def __call__(self, img_tensor, keypoints=None):
        # Calculate random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        translations_percent = (
            random.uniform(-self.translate[0], self.translate[0]),
            random.uniform(-self.translate[1], self.translate[1]),
        )

        translations_pixels = [
            translations_percent[0] * img_tensor.shape[-1] / 2,
            translations_percent[1] * img_tensor.shape[-2] / 2,
        ]
        shear = [
            random.uniform(-self.shear[0], self.shear[0]),
            random.uniform(-self.shear[1], self.shear[1]),
        ]

        # Store them for inverse transformation
        self.last_params = {
            "angle": angle,
            "scale": scale_factor,
            "translations_percent": translations_percent,
            "shear": shear,
        }

        # Apply transformation
        transformed_img = TF.affine(
            img_tensor,
            angle=angle,
            translate=translations_pixels,
            scale=scale_factor,
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        if keypoints is None:
            return transformed_img

        transformed_keypoints = transform_keypoints(
            keypoints,
            angle,
            -1 * torch.tensor([translations_pixels[1], translations_pixels[0]]),
            1 / scale_factor,
            shear,
            img_tensor.shape[-2:],
        )

        return transformed_img, transformed_keypoints

    def inverse(self, img_tensor):
        # Retrieve stored parameters and compute translations in pixels for current size
        angle = -self.last_params["angle"]  # Inverse of rotation
        translations_pixels = (
            -self.last_params["translations_percent"][0] * img_tensor.shape[-1] / 2,
            -self.last_params["translations_percent"][1] * img_tensor.shape[-2] / 2,
        )  # Inverse of translation
        scale_factor = 1.0 / self.last_params["scale"]  # Inverse of scaling
        shear = self.last_params["shear"]  # Inverse of shear

        # Apply inverse transformation
        return TF.affine(
            img_tensor,
            angle=angle,
            translate=translations_pixels,
            scale=scale_factor,
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

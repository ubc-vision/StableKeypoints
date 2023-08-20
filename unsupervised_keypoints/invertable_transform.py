import torch
import random
from torchvision.transforms import functional as TF


class RandomAffineWithInverse:
    def __init__(self, degrees, scale, translate):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate

    def __call__(self, img_tensor):
        # Calculate random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        translations_percent = (
            random.uniform(-self.translate[0], self.translate[0]),
            random.uniform(-self.translate[1], self.translate[1]),
        )

        translations_pixels = (
            translations_percent[0] * img_tensor.shape[-1],
            translations_percent[1] * img_tensor.shape[-2],
        )

        # Store them for inverse transformation
        self.last_params = {
            "angle": angle,
            "scale": scale_factor,
            "translations_percent": translations_percent,
        }

        # Apply transformation
        return TF.affine(
            img_tensor,
            angle=angle,
            translate=translations_pixels,
            scale=scale_factor,
            shear=0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    def inverse(self, img_tensor):
        # Retrieve stored parameters and compute translations in pixels for current size
        angle = -self.last_params["angle"]  # Inverse of rotation
        translations_pixels = (
            -self.last_params["translations_percent"][0] * img_tensor.shape[-1],
            -self.last_params["translations_percent"][1] * img_tensor.shape[-2],
        )  # Inverse of translation
        scale_factor = 1.0 / self.last_params["scale"]  # Inverse of scaling

        # Apply inverse transformation
        return TF.affine(
            img_tensor,
            angle=angle,
            translate=translations_pixels,
            scale=scale_factor,
            shear=0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

# from skimage.transform import AffineTransform, warp
# import torch
# import numpy as np


# class CustomTransform:
#     def __init__(self, degrees=30, scale=(1.0, 1.1), translate=(0.1, 0.1)):
#         # Define ranges for your transformations
#         self.degrees = degrees
#         self.scale_range = scale
#         self.translate_range = translate

#     def __call__(self, image, keypoints):
#         # Convert image to numpy and transpose to (height, width, channels)
#         image_np = image.numpy().transpose(1, 2, 0)

#         # Generate random values for transformations
#         angle = np.random.uniform(-self.degrees, self.degrees)
#         scale = (
#             np.random.uniform(self.scale_range[0], self.scale_range[1]),
#             np.random.uniform(self.scale_range[0], self.scale_range[1]),
#         )
#         translate = (
#             np.random.uniform(
#                 -self.translate_range[0] * image_np.shape[1],
#                 self.translate_range[0] * image_np.shape[1],
#             ),
#             np.random.uniform(
#                 -self.translate_range[1] * image_np.shape[0],
#                 self.translate_range[1] * image_np.shape[0],
#             ),
#         )

#         # Scale keypoints to image resolution
#         keypoints[:, 0] *= image_np.shape[1]
#         keypoints[:, 1] *= image_np.shape[0]

#         # Translate the keypoints so that the center is at (0, 0)
#         keypoints -= np.array([image_np.shape[1] * 0.5, image_np.shape[0] * 0.5])

#         # Create the AffineTransform
#         affine_tf = AffineTransform(
#             rotation=np.deg2rad(angle), scale=scale, translation=translate
#         )

#         # Apply the transformation to the image
#         transformed_image_np = warp(image_np, affine_tf)

#         # Apply the transformation to the keypoints
#         transformed_keypoints = affine_tf(keypoints)

#         # Translate the keypoints back so that the center is at (0.5, 0.5)
#         transformed_keypoints += np.array(
#             [image_np.shape[1] * 0.5, image_np.shape[0] * 0.5]
#         )

#         # Scale keypoints back to the 0-1 range
#         transformed_keypoints[:, 0] /= image_np.shape[1]
#         transformed_keypoints[:, 1] /= image_np.shape[0]

#         transformed_keypoints = torch.tensor(transformed_keypoints).float()

#         # Transpose the transformed image back to (channels, height, width)
#         transformed_image = torch.tensor(
#             transformed_image_np.transpose(2, 0, 1)
#         ).float()

#         return transformed_image, transformed_keypoints

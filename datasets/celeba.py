import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """

    def __init__(
        self,
        max_len=-1,
        split="train",
        align=True,
        dataset_loc="~",
        iou_threshold= 0.3,
    ):
        self.dataset_loc = dataset_loc
        self.mafl_loc = os.path.join(dataset_loc, "MAFL")
        
        self.max_len = max_len

        if align:
            landmark_loc = os.path.join(
                self.dataset_loc, "Anno", "list_landmarks_align_celeba.txt"
            )
        else:
            landmark_loc = os.path.join(
                self.dataset_loc, "Anno", "list_landmarks_celeba.txt"
            )

        # load the .txt file
        self.landmarks = open(landmark_loc, "r")
        self.landmarks = self.landmarks.readlines()

        self.num_kps = 5

        self.align = align
        
        self.split = split

        if split == "test":
            self.file_names = open(os.path.join(self.mafl_loc, "testing.txt"), "r")
        elif split == "train":
            self.file_names = open(os.path.join(self.mafl_loc, "training.txt"), "r")
        self.file_names = self.file_names.readlines()
        
        # filter file_names to only include images where the bounding box covers a certain threshold of the image
        if not align:
            
            bboxes= open(os.path.join(self.dataset_loc, "Anno", "list_bbox_celeba.txt"), "r")
            bboxes = bboxes.readlines()[2:]
            
            indices_to_remove = []

            for i in range(len(self.file_names)):
                this_file_index = self.find_local_index(i)
                
                this_bbox = bboxes[this_file_index].split()[1:]
                this_bbox = [int(x) for x in this_bbox]
                
                width, height = Image.open(self.return_img_path(this_file_index)).size
                
                if this_bbox[2]*this_bbox[3] < height*width*iou_threshold:
                    indices_to_remove.append(i)

            # Remove the elements
            for i in reversed(indices_to_remove):
                self.file_names.pop(i)
                    


    def __len__(self):
        if self.max_len != -1:
            return self.max_len
        return len(self.file_names)

    def find_local_index(self, global_index):
        local_file_name = self.file_names[global_index]
        # remove everything after the "."
        local_file_name = local_file_name.split(".")[0]

        # convert to int
        local_file_name = int(local_file_name)

        # convert to 0 indexing
        local_file_name = local_file_name - 1

        return local_file_name

    def __getitem__(self, index):
        local_index = self.find_local_index(index)

        img = self.load_image(local_index)

        kpts = self.load_keypoints(local_index)

        return {"img": img, "kpts": kpts}

    def load_image(self, index):
        image = Image.open(self.return_img_path(index)).convert("RGB")

        image = image.resize((512, 512), Image.BILINEAR)

        image = np.array(image)

        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image) / 255.0

        return image

    def load_keypoints(self, index):
        width, height = Image.open(self.return_img_path(index)).size

        # Get the line corresponding to the index
        line = self.landmarks[index + 2]  # +2 to skip the header lines

        # Split the line by spaces and ignore the image name
        parts = line.split()[1:]

        # Convert to numbers
        keypoints = [float(p) for p in parts]

        # Reshape keypoints into [5, 2] and convert to torch tensor
        keypoints = torch.tensor(keypoints).reshape(5, 2)

        # normalize by image size
        keypoints = keypoints / torch.tensor([width, height])

        # swap the x and y
        keypoints = keypoints[:, [1, 0]]

        return keypoints

    def return_img_path(self, index):
        # img_name = self.landmarks.iloc[index].image_id

        img_name = f"{index+1:06d}" + (".png" if self.align else ".jpg")

        if self.align:
            return os.path.join(
                self.dataset_loc, "Img", "img_align_celeba_png", img_name
            )
        else:
            return os.path.join(self.dataset_loc, "Img", "img_celeba", img_name)


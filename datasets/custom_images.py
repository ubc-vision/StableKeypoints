import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.image_files = [f for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f))]
        # sort by name
        self.image_files.sort()
        self.transform = transforms.Compose([
            # transforms.Lambda(lambda img: img.crop((280, 0, img.width - 280, img.height))),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        sample = {'img': img, 'kpts': torch.zeros(15, 2), 'visibility': torch.zeros(15)}
        return sample

    def __len__(self):
        return len(self.image_files)

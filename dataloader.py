import glob
import os

import torch
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms


class CustomDataSet(Dataset):
    def __init__(self, main_dir, data_path_nomask, load_alpha, ):
        self.main_dir = main_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.load_alpha = load_alpha
        self.total_imgs =  glob.glob(main_dir)
        self.total_mask =  glob.glob(data_path_nomask)


        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc)
        image = image.resize((1024, 1024))

        rgb = image.convert("RGB")
        tensor_image = self.transform(rgb)

        img_loc = self.total_mask[idx]
        image = Image.open(img_loc)
        image = image.resize((1024, 1024))

        rgb = image.convert("RGB")
        mask = self.transform(rgb)

        return tensor_image, mask

def get_data_loader(data_path, data_path_nomask, alpha=False, is_train=False):
    """Creates training and test data loaders."""
    dataset = CustomDataSet(data_path, data_path_nomask, alpha)
    dloader = DataLoader(dataset=dataset, batch_size=1, shuffle=is_train, drop_last=is_train,
                         num_workers=10)

    return dloader

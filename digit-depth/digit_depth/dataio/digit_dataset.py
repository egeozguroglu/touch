import glob

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DigitRealImageAnnotDataset(Dataset):
    def __init__( self, dir_dataset, annot_file, transform=None, annot_flag=True, img_type="png"):
        self.dir_dataset = dir_dataset
        print(f"Loading dataset from {dir_dataset}")
        self.transform = transform
        self.annot_flag = annot_flag

        # a list of image paths sorted. dir_dataset is the root dir of the datasets (color)
        self.img_files = sorted(glob.glob(f"{self.dir_dataset}/*.{img_type}"))
        print(f"Found {len(self.img_files)} images")

        if self.annot_flag:
            self.annot_dataframe = pd.read_csv(annot_file, sep=",")
            print(len(self.annot_dataframe))

    def __getitem__(self, idx):
        """Returns a tuple of (img, annot) where annot is a tensor of shape (3,1)"""

        # read in image
        img = Image.open(self.img_files[idx])
        img = self.transform(img)
        img = img.permute(0, 2, 1)  # (3,240,320) -> (3,320,240)
        # read in region annotations
        if self.annot_flag:
            img_name = self.img_files[idx]
            row_filter = self.annot_dataframe["img_name"] == img_name
            region_attr = self.annot_dataframe.loc[
                row_filter, ["center_x", "center_y", "radius"]
            ]
            annot = (torch.tensor(region_attr.values, dtype=torch.int32) if (len(region_attr) > 0) else torch.tensor([]))
        data = img
        if self.annot_flag:
            data = (img, annot)
        return data

    def __len__(self):
        return len(self.img_files)

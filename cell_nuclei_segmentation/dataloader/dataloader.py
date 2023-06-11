from __future__ import annotations
import os
import sys
import inspect
import torch
import cv2
from typing import Union, List, Tuple, Callable, Optional, Any
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Resize
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import albumentations as A


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class CellNucleiDataset(Dataset):

    def __init__(
        self,
        X: List[Image.Image],
        y: List[Image.Image],
        target_size: Tuple[int, int],
        transform: Callable[[Any], Any] = None,  # TODO: better type
    ) -> None:
        """
        :param X: List of images.
        :param y: List of groundtruth masks.
        :param transform: Used on training dataset, transform to be applied on each sample.
            Should be at least ToTensor()
        """
        assert len(X) == len(y), f"X and y must have the same length. Given {len(X)} and {len(y)}."
        self._X = X
        self._y = y
        self._target_size = target_size
        self._transform = transform


    @staticmethod
    def create(
        images_path: str, 
        groundtruth_path: str, 
        target_size: Tuple[int, int],
        transform: Callable[[Any], Any],  # TODO: better type
    ) -> CellNucleiDataset:
        """
        Creates a dataset from the images in the data folder.

        :param images_path: Path to the .tif rawimages folder.
        """
        images_filenames_list = os.listdir(images_path)
        groundtruth_filenames_list = os.listdir(groundtruth_path)

        X = []
        y = []
        for image_filename in images_filenames_list:
            with Image.open(os.path.join(images_path, image_filename)) as img:
                X.append(img.convert(mode="L"))

        for groundtruth_filename in groundtruth_filenames_list:
            with Image.open(os.path.join(groundtruth_path, groundtruth_filename)) as img:
                y.append(img.convert(mode="L"))

        return CellNucleiDataset(
            X=X,
            y=y,
            target_size=target_size,
            transform=transform,
        )

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x, y =  self._X[idx], self._y[idx]

        # Resize the image and mask
        resize_transform = Resize(self._target_size)
        x = resize_transform(x)
        y = resize_transform(y)

        # TODO: y contains ints from 0 to 300, we probably want binary tensors instead of floats with values between 0 and 1
        if self._transform is not None:
            x = self._transform(x)
            y = torch.from_numpy(np.array(y, dtype=np.int32))
        return x, y
    

if __name__ == "__main__":
    # Create the dataset.
    dataset = CellNucleiDataset.create(
        images_path=os.path.join(CURRENT_DIR, '..', 'dataset', 'rawimages'),
        groundtruth_path=os.path.join(CURRENT_DIR, '..', 'dataset', 'groundtruth'),
        target_size=(1024, 1360),  # biggest image in the dataset
        transform=ToTensor(),
    )

    # Create the dataloader.
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
    )

    # Iterate over the dataset.
    for X, y in dataloader:
        print(X.shape)
        print(y.shape)
        break
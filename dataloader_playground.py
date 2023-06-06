import os
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision
from PIL import Image
# from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,  # model output channels (number of classes in your dataset)
)

from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")

DATASET_ROOT = Path("./S-BSST265/dataset/")

# raw_images_dataset = ImageFolder(DATASET_ROOT / 'rawimages')


class CustomDataSet(Dataset):
    def __init__(self, dir: Path, transform=None):
        self.dir = dir
        self.transform = transform
        # img_filenames =
        self.img_filenames = sorted( os.listdir(dir / "rawimages"))
        # self.masks = sorted( os.listdir(dir / "groundtruth")) # use filenames only here and reuse them for groundtruth
        # self.img_filenames = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        raw_img_loc = self.dir / "rawimages" / self.img_filenames[idx]
        mask_loc = self.dir / "groundtruth" / self.img_filenames[idx]
        img = img.open(raw_img_loc).convert("RGB")
        mask = img.open(raw_img_loc).convert("RGB")
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, mask


raw_images_dataset = CustomDataSet(DATASET_ROOT)
# raw_images_dataset = CustomDataSet(DATASET_ROOT / "rawimages")

print(set(map(lambda i: i.size, raw_images_dataset)))

# for img in raw_images_dataset:
#     # print(img)
#     # plt.imshow(img)
#     # plt.show()
#     print(img.size)
#     # break

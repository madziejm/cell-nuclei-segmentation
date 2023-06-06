import os
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchvision
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Dataset

# not used now
# model = smp.Unet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=3,  # model output channels (number of classes in your dataset)
# )


DATASET_ROOT = Path("./S-BSST265/dataset/")


class CustomDataSet(Dataset):
    def __init__(self, dir: Path, img_transform=None, mask_transform=None):
        self.dir = dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_filenames = sorted(os.listdir(dir / "rawimages"))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        raw_img_loc = self.dir / "rawimages" / self.img_filenames[idx]
        mask_loc = self.dir / "groundtruth" / self.img_filenames[idx]
        img = torchvision.transforms.functional.pil_to_tensor(
            Image.open(raw_img_loc).convert("RGB")
        )
        mask = torch.from_numpy(
            np.asarray(Image.open(mask_loc), dtype=np.uint8)
        ).unsqueeze(
            0
        )  # introduce "channel" dim that is supposedly needed

        crop_coords = torchvision.transforms.RandomCrop.get_params(
            img,
            output_size=(
                416,
                512,
            ),  # min of all the sizes in the dataset seems to be (430, 512); height and width are supposed to be divisible by 32 for the chosen model
        )

        img = torchvision.transforms.functional.crop(img, *crop_coords)
        mask = torchvision.transforms.functional.crop(mask, *crop_coords)

        mask = (
            mask != 0
        )  # let's pretend this is binary semantic segmentation, not instance segmentation # TODO remove me and adjust the model

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask


image_dataset = CustomDataSet(DATASET_ROOT)

train_size = int(0.8 * len(image_dataset))
valid_size = len(image_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    image_dataset, [train_size, valid_size]
)

# image sizes
print(
    set(map(lambda i: i[0].size(), train_dataset))
)  # {(1225, 914), (550, 430), (1360, 1024), (1014, 763), (1280, 1024), (1024, 1024)}


## adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
class SegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.outputs = dict(train=[], valid=[], test=[])

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        self.outputs[stage].append(
            {
                "loss": loss.detach(),
                "tp": tp.detach(),
                "fp": fp.detach(),
                "fn": fn.detach(),
                "tn": tn.detach(),
            }
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, stage):
        outputs = self.outputs[stage]
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(
            metrics,
            prog_bar=True,
        )
        self.outputs[stage].clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


## adapted end

model = SegmentationModel("Unet", "resnet101", in_channels=3, out_classes=1)


import tracemalloc

tracemalloc.start()
trainer = pl.Trainer(
    # gpus=1, # TODO not this time
    max_epochs=2,  # TODO increase me
    log_every_n_steps=1,
)


n_cpu = os.cpu_count()
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=(n_cpu // 2),  #  TODO more workers
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=(n_cpu // 2),  #  TODO more workers
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

from comet_ml import Experiment
import os
import torch
import numpy as np
from models.maskrcnn import MaskRCNN
import torch.nn.functional as F


def mask_to_bboxes(mask: torch.Tensor):
    bboxes = []

    masks_ids = np.unique(mask)[1:]
    for mask_id in masks_ids:

        # Get the coordinates of the non-zero values of the mask
        pos = np.where(mask == mask_id)

        # Calculate the minimum and maximum coordinates
        xmin = pos[1].min()
        xmax = pos[1].max()
        ymin = pos[0].min()
        ymax = pos[0].max()

        # Append the bounding box to the list
        bboxes.append([xmin, ymin, xmax, ymax])

    # Convert the list to a FloatTensor and return
    return torch.tensor(bboxes, device=mask.device, dtype=torch.int32)


def convert_masks(mask):
    # Get all unique non-zero values
    unique_values = torch.unique(mask)[1:]  # Skip 0 as it represents background

    # Create a binary mask for each unique value
    binary_masks = [(mask == value).float() for value in unique_values]

    # Stack all binary masks into a single tensor
    binary_masks_tensor = torch.stack(binary_masks)

    return binary_masks_tensor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Trainer:

    def __init__(
        self, 
        model: MaskRCNN, 
        device: str,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        experiment: Experiment,
    ) -> None:
        self._model = model
        self._device = device
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._experiment = experiment

    
    def train(self, num_epochs: int) -> None:
        self._model.train()
        self._model.to(self._device)

        total_steps: int = 0
        with self._experiment.train():
            for epoch in range(num_epochs):
                self._model.save(path=os.path.join(CURRENT_DIR, "..", "saved_models", f"maskrcnn_epoch_{epoch}"))
                self._experiment.log_current_epoch(epoch)
                for i, (images, targets) in enumerate(self._dataloader):
                    total_steps += 1

                    images = list(image.to(self._device) for image in images)
                    targets = [
                        { 
                            'masks': convert_masks(target).to(self._device),
                            'boxes': mask_to_bboxes(target).to(self._device),
                            'labels': torch.ones((len(np.unique(target) - 1)), dtype=torch.int64).to(self._device), 
                        } for target in targets
                    ]

                    # Forward pass
                    loss_dict = self._model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    # Backward pass and optimization
                    self._optimizer.zero_grad()
                    losses.backward()
                    self._optimizer.step()

                    self._experiment.log_metric(
                        name="each_sample_loss", 
                        value=losses.item(), 
                        step=total_steps
                    )
                    if i % 10 == 0:
                        print(f"Epoch: {epoch}, Iteration: {i}, Total steps: {total_steps}, Loss: {losses.item()}")

                self._model.save(path=os.path.join(CURRENT_DIR, "..", "saved_models", f"maskrcnn_epoch_{epoch}"))

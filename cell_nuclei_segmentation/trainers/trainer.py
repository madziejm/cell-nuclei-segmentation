from typing import List
from comet_ml import Experiment
import os
import torch
import numpy as np
from models.maskrcnn import MaskRCNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from trainers.utils import mask_to_bboxes_and_training_masks


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Trainer:

    def __init__(
        self, 
        model: MaskRCNN, 
        device: str,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        experiment: Experiment,
    ) -> None:
        self._model = model
        self._device = device
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._experiment = experiment
        self._validation_image_sent = False

    
    def _log_file_to_cometml(self, images, epoch) -> None:
        if not self._validation_image_sent:
            self._experiment.log_image(
                images[0][0].detach().cpu(),
                name=f'validation_image_{epoch}'
            )
            self._validation_image_sent = True
        
        image = images[0][0].detach().cpu()
        self._model.eval()
        prediction = self._model(images)
        self._model.train()
        pred = prediction[0]
        masks = pred['masks'].cpu().detach().numpy()
        boxes = pred['boxes'].cpu().detach().numpy()

        for threshold in (0.1, 0.5, 0.9):
            fig, ax = plt.subplots(1, figsize=(15, 15))
            ax.imshow(image)
            
            for i in range(masks.shape[0]):
                mask = masks[i][0]
                ax.imshow(mask > threshold, alpha=0.3)
                box = boxes[i]
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            filename = f"validation_pred_epoch_{epoch}_threshold_{threshold}.png"
            plt.savefig(filename)
            self._experiment.log_image(
                image_data=filename,
                name=filename
            )
                        
    def _evaluate_model(self, epoch: int) -> None:
        # TODO: model should be put to .eval() to disable batchnorm and dropout
        #   but then loss_dict is not avaiable anymore
        #   see: https://discuss.pytorch.org/t/how-to-calculate-validation-loss-for-faster-rcnn/96307/13
        
        avg_losses = {
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_mask": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
        }
        avg_loss_sum = []
        for i, (images, targets) in enumerate(self._test_dataloader):
            with torch.no_grad():
                model_targets = self._create_model_targets(targets=targets)
                images = list(image.to(self._device) for image in images)
                loss_dict = self._model(images, model_targets.to(self._device))
                """
                {'loss_classifier': tensor(2.7450),
                'loss_box_reg': tensor(0.5253),
                'loss_mask': tensor(1.3972),
                'loss_objectness': tensor(6.8426),
                'loss_rpn_box_reg': tensor(0.6993)}
                """
                losses = sum(loss for loss in loss_dict.values())
                avg_loss_sum.append(losses.item())
                for loss_name in avg_losses.keys():
                    avg_losses[loss_name].append(loss_dict[loss_name].item())
                    
                if i == 0:
                    self._log_file_to_cometml(images=images, epoch=epoch)
                
        print(f"Eval epoch: {epoch}, avg loss: {np.mean(avg_loss_sum)}")
        self._experiment.log_metric(
            name=f"validation_loss_sum", 
            value=np.mean(avg_loss_sum), 
            step=epoch
        )
        for loss_name in avg_losses.keys():
            self._experiment.log_metric(
                name=f"validation_{loss_name}", 
                value=np.mean(avg_losses[loss_name]), 
                step=epoch
            )
            print(f"Eval epoch: {epoch}, avg {loss_name}: {np.mean(avg_losses[loss_name])}")
                    
        
    def _create_model_targets(self, targets) -> List[dict]:
        model_targets = []
        for target in targets:
            boxes, masks = mask_to_bboxes_and_training_masks(mask=target)
            model_targets.append(
                {
                    'masks': masks.to(self._device),
                    'boxes': boxes.to(self._device),
                    'labels': torch.ones((len(np.unique(target) - 1)), dtype=torch.int64).to(self._device),    
                }
            )
        return model_targets
            
    def train(self, num_epochs: int) -> None:
        self._model.train()
        self._model.to(self._device)

        total_steps: int = 0
        with self._experiment.train():
            self._evaluate_model(epoch=0)
            for epoch in range(1, num_epochs + 1):
                self._model.save(path=os.path.join(CURRENT_DIR, "..", "saved_models", f"maskrcnn_epoch_{epoch}"))
                self._experiment.log_current_epoch(epoch)
                for i, (images, targets) in enumerate(self._train_dataloader):
                    total_steps += 1

                    images = list(image.to(self._device) for image in images)
                    model_targets = self._create_model_targets(targets=targets)
                    
                    # Forward pass
                    loss_dict = self._model(images, model_targets)
                    losses = sum(loss for loss in loss_dict.values())  # tensor(12.2093)

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

                # Save model after each epoch
                self._model.save(path=os.path.join(CURRENT_DIR, "..", "saved_models", f"maskrcnn_epoch_{epoch}.pth"))
                
                # Evaluate model after each epoch
                self._evaluate_model(epoch=epoch)

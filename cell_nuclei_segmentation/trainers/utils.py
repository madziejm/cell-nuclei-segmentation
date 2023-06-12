import cv2
import torch
import numpy as np


def _denoise_mask(mask, kernel_size=5, blur_size=3):
    # Convert the mask to a numpy array and ensure type is uint8
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(mask_np, (blur_size, blur_size), 0)

    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Use cv2.morphologyEx function to perform opening operation
    denoised_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)

    return denoised_mask

def mask_to_bboxes_and_training_masks(mask: torch.Tensor):
    bboxes = []
    binary_masks = []
    
    masks_ids = np.unique(mask)[1:]
    for mask_id in masks_ids:
        mask_denoised = _denoise_mask(mask == mask_id)
        pos = np.where(mask_denoised == 1)
        
        if len(pos[0]) >= 4:  # assume mask has to be at least 4 pixels
            binary_mask = (mask_denoised == 1)
            binary_masks.append(
                torch.tensor(binary_mask, device=mask.device).float()
            )
            
            # Calculate the minimum and maximum coordinates
            xmin = pos[1].min()
            xmax = pos[1].max()
            ymin = pos[0].min()
            ymax = pos[0].max()

            # Append the bounding box to the list
            bboxes.append([xmin, ymin, xmax, ymax])

    bboxes = torch.tensor(bboxes, device=mask.device, dtype=torch.int32)
    binary_masks = torch.stack(binary_masks)
    return bboxes, binary_masks

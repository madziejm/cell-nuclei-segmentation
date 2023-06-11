from __future__ import annotations
from typing import Any
import torch
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MaskRCNN:
    def __init__(self) -> None:
        self._model = maskrcnn_resnet50_fpn(pretrained=False)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self._model(*args, **kwargs)
    
    def eval(self) -> None:
        self._model.eval()

    def cuda(self) -> MaskRCNN:
        return self._model.cuda()
    
    def train(self) -> None:
        self._model.train()

    def to(self, device: str) -> MaskRCNN:
        return self._model.to(device)
    
    def parameters(self) -> Any:
        return self._model.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def save(self, name: str) -> None:
        torch.save(self._model.state_dict(), os.path.join(CURRENT_DIR, '..', 'saved_models', f'{name}.pth'))

    

if __name__ == "__main__":
    model = MaskRCNN()
    model.eval()

    # Load your image
    image = Image.open(os.path.join(CURRENT_DIR, '..', 'dataset', 'rawimages', "Ganglioneuroblastoma_0.tif")).convert("L")
    # Convert PIL image to tensor
    image_tensor = F.to_tensor(image)
    # Add an extra dimension at the beginning of the tensor, which represents the batch size
    image_tensor = image_tensor.unsqueeze(0)

    # Check for GPU availability and if available, move the model and input tensor to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        image_tensor = image_tensor.cuda()

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
        print(prediction)

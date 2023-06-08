import pytest
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from cell_nuclei_segmentation.dataloader import CellNucleiDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def dataset() -> CellNucleiDataset:
    return CellNucleiDataset.create(
        images_path=os.path.join(CURRENT_DIR, '..', 'dataset', 'rawimages'),
        groundtruth_path=os.path.join(CURRENT_DIR, '..', 'dataset', 'groundtruth'),
        target_size=(1024, 1360),  # biggest image in the dataset
        transform=ToTensor(),
    )

def test_dataset_length(dataset: CellNucleiDataset):
    assert len(dataset) == 10

def test_dataset_item(dataset: CellNucleiDataset):
    image, mask = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (1024, 1360)
    assert mask.shape == (1024, 1360)


def test_dataloader(dataset: CellNucleiDataset):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for images, masks in dataloader:
        assert images.shape == (4, 1024, 1360)
        assert masks.shape == (4, 1024, 1360)
        break

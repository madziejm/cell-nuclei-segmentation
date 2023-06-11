from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

from typing import List, Tuple, Dict, Union, Any
import os
import torch
import yaml
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from models.maskrcnn import MaskRCNN
from trainers.trainer import Trainer
from dataloader.dataloader import CellNucleiDataset


def main() -> None:
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    print("Starting training with config:\n", config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    experiment = Experiment(
        api_key = "FN4Av6jEhDTqxcLYbHwkGMJeO",
        project_name = "cell-nuclei-segmentation",
        workspace="thefebrin"
    )
    experiment.add_tag(config["cometml_tag"])
    experiment.set_name(config["cometml_name"])
    experiment.add_tag(config["cometml_name"])
    experiment.log_parameters(config)

    dataset = CellNucleiDataset.create(
        images_path=os.path.join('dataset', 'rawimages'),
        groundtruth_path=os.path.join('dataset', 'groundtruth'),
        target_size=(
            config["target_size_x"], config["target_size_y"]
        ),  # biggest image in the dataset
        transform=ToTensor(),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    
    model=MaskRCNN()
  
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
    )

    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=config["learning_rate"], 
    #     momentum=0.9,
    #     weight_decay=0.0005,
    # )

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        dataloader=dataloader,
        experiment=experiment,
    )

    trainer.train(num_epochs=config["num_epochs"])


if __name__ == "__main__":
    main()
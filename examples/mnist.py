import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.layer import Layer, euclidian_distance, cosine_distance
import os
from pathlib import Path


class TransformFlat:
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, input_data):
        data = self.to_tensor(input_data).flatten()
        return data


training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=TransformFlat()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=TransformFlat()
)


@hydra.main(config_path="../config", config_name="mnist", version_base="1.3")
def run(cfg: DictConfig):
    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=cfg.pin_memory
    )
    # print(f"hydra.run.dir", hydra.run.dir)
    print(f"Current working directory : {os.getcwd()}")
    # print(f"Orig working directory    : {get_original_cwd()}")
    layer = Layer(
        num_classes=10,
        distance_metric=euclidian_distance,
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
    )

    layer.epoch_loop(
        dataloader=train_dataloader,
    )
    num_neighbors = len(layer.neighbors)

    print("neighbors.device", layer.neighbors.device)
    print("neighbor_class.device", layer.neighbor_class.device)

    directory = Path(os.getcwd())
    torch.save(layer.neighbors, str(directory / "neighbors.pt"))
    torch.save(layer.neighbor_class, str(directory / "neighbor_classes.pt"))

    train_result = layer.test_loop(
        dataloader=train_dataloader,
    )
    print("train_result", train_result)

    test_result = layer.test_loop(
        dataloader=test_dataloader,
    )

    print("test_result", test_result)
    print("neighbors in model", num_neighbors)


if __name__ == "__main__":
    run()

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.layer import Layer, euclidian_distance, cosine_distance
from deep_nearest_neighbor.networks import Network
import os
from pathlib import Path
from hydra.utils import get_original_cwd, to_absolute_path


class TransformFlat:
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, input_data):
        data = self.to_tensor(input_data).flatten()
        return data


def run_single_layer(cfg: DictConfig):
    mnist_data_path = str(Path(get_original_cwd()) / "data")

    training_data = datasets.MNIST(
        root=mnist_data_path,
        train=True,
        download=True,
        transform=TransformFlat(),
    )

    test_data = datasets.MNIST(
        root=mnist_data_path, train=False, download=True, transform=TransformFlat()
    )

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
        max_neighbors=cfg.max_neighbors,
    )

    layer.epoch_loop(
        dataloader=train_dataloader,
    )
    num_neighbors = len(layer.neighbors)

    layer.save()

    train_result = layer.test_loop(
        dataloader=train_dataloader,
    )
    print("train_result", train_result)

    test_result = layer.test_loop(
        dataloader=test_dataloader,
    )

    print("test_result", test_result)
    print("neighbors in model", num_neighbors)


def run_network(cfg: DictConfig):
    mnist_data_path = str(Path(get_original_cwd()) / "data")

    training_data = datasets.MNIST(
        root=mnist_data_path,
        train=True,
        download=True,
        transform=TransformFlat(),
    )

    test_data = datasets.MNIST(
        root=mnist_data_path, train=False, download=True, transform=TransformFlat()
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=cfg.pin_memory
    )

    network = Network(
        dataloader=train_dataloader,
        num_classes=10,
        distance_metric=euclidian_distance,
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
        max_neighbors=cfg.max_neighbors,
    )

    network.train()

    print("")
    print(network.layer(1).num_neighbors, network.layer(1).num_features)

    train_result = network.test_loop(
        dataloader=train_dataloader,
    )
    print("")
    print("train_result", train_result)

    test_result = network.test_loop(
        dataloader=test_dataloader,
    )
    print("")
    print("test_result", test_result)


@hydra.main(config_path="../config", config_name="mnist", version_base="1.3")
def run(cfg: DictConfig):
    if cfg.train_network is False:
        run_single_layer(cfg=cfg)
    else:
        run_network(cfg=cfg)


if __name__ == "__main__":
    run()

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.layer import Layer
from deep_nearest_neighbor.metrics import (
    CosineDistance,
    InfluenceCone,
    EuclidianDistance,
    EuclidianPyramidDistance,
)
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


def choose_dataset(cfg: DictConfig):
    mnist_data_path = str(Path(get_original_cwd()) / "data")

    num_classes = None
    if cfg.data == "mnist":
        num_classes = 10
        training_data = datasets.MNIST(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.MNIST(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )
    elif cfg.data == "cifar100":
        num_classes = 100
        training_data = datasets.CIFAR100(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.CIFAR100(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )
    elif cfg.data == "cifar10":
        num_classes = 10
        training_data = datasets.CIFAR10(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.CIFAR10(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )

    return training_data, test_data, num_classes


def choose_metric(cfg: DictConfig):
    if cfg.kernel_type == "influence_cone":
        distance_metric = InfluenceCone(
            epsilon=cfg.epsilon, exponent=cfg.exponent, factor=cfg.influence_cone_factor
        )
    elif cfg.kernel_type == "euclidian_pyramid":
        distance_metric = EuclidianPyramidDistance(
            epsilon=cfg.epsilon, exponent=cfg.exponent, scales=cfg.scales
        )
    elif cfg.kernel_type == "cosine":
        distance_metric = CosineDistance(epsilon=cfg.epsilon, exponent=cfg.exponent)
    else:
        distance_metric = EuclidianDistance(epsilon=cfg.epsilon, exponent=cfg.exponent)

    return distance_metric


def run_single_layer(cfg: DictConfig):
    training_data, test_data, num_classes = choose_dataset(cfg=cfg)
    distance_metric = choose_metric(cfg)

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
        num_classes=num_classes,
        # distance_metric=InfluenceCone(epsilon=1e-6, exponent=2, factor=4),
        distance_metric=distance_metric,
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
        max_neighbors=cfg.max_neighbors,
        max_count=cfg.max_count,
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
    training_data, test_data, num_classes = choose_dataset(cfg=cfg)
    distance_metric = choose_metric(cfg)

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
        num_classes=num_classes,
        distance_metric=distance_metric,
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
        max_neighbors=cfg.max_neighbors,
        num_layers=cfg.num_layers
        # max_count=cfg.max_count,
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


@hydra.main(
    config_path="../config",
    config_name="invariant_image_classification",
    version_base="1.3",
)
def run(cfg: DictConfig):
    if cfg.train_network is False:
        run_single_layer(cfg=cfg)
    else:
        run_network(cfg=cfg)


if __name__ == "__main__":
    run()

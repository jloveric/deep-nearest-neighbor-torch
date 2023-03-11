import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.single_image_dataset import (
    ImageDataset,
    image_to_dataset,
    image,
)
from deep_nearest_neighbor.layer import (
    RegressionLayer,
    EuclidianDistance,
    cosine_distance,
)
from deep_nearest_neighbor.networks import Network
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class TransformFlat:
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, input_data):
        data = self.to_tensor(input_data).flatten()
        return data


filename = "images/mountains.jpg"
training_data = ImageDataset(filename=filename)


def run_single_layer(cfg: DictConfig):
    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )

    # print(f"hydra.run.dir", hydra.run.dir)
    print(f"Current working directory : {os.getcwd()}")
    # print(f"Orig working directory    : {get_original_cwd()}")
    layer = RegressionLayer(
        distance_metric=EuclidianDistance(epsilon=cfg.epsilon, exponent=cfg.exponent),
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
        max_neighbors=cfg.max_neighbors,
        tolerance=cfg.tolerance,
    )

    layer.epoch_loop(
        dataloader=train_dataloader,
    )
    num_neighbors = len(layer.neighbors)

    print("neighbors.device", layer.neighbors.device)
    print("neighbor_class.device", layer.neighbor_value.device)

    directory = Path(os.getcwd())
    torch.save(layer.neighbors, str(directory / "neighbors.pt"))
    torch.save(layer.neighbor_value, str(directory / "neighbor_classes.pt"))

    train_result = layer.test_loop(
        dataloader=train_dataloader,
    )

    print("train_result", train_result)
    print("neighbors in model", num_neighbors)

    # Use this to get the xy
    img = image.imread(filename)
    shape = img.shape
    result = image_to_dataset(filename=filename, device=cfg.device)
    # shape = result[0].shape

    xy = result[1].to(cfg.device)

    # predict the result
    # layer.to("cpu")

    rgb = []
    for i in range(0, len(xy), cfg.batch_size):
        rgb += [layer(xy[i : (i + cfg.batch_size)])]

    rgb = torch.stack(rgb, dim=0)

    print("rgb.shape", rgb.shape)
    rgb = rgb.reshape(shape[0], shape[1], 3).cpu().numpy()

    # img = mpimg.imread('your_image.png')
    centers = layer._neighbors.to("cpu")
    colors = layer._neighbor_value.to("cpu")
    print("center.shape", centers.shape)
    # plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.scatter(centers[:, 1], -centers[:, 0], s=1, c=colors)
    plt.axis("off")
    plt.axis("equal")
    # plt.figure(2)
    plt.subplot(1, 2, 2)

    imgplot = plt.imshow(rgb)
    plt.axis("off")

    plt.show()


@hydra.main(
    config_path="../config", config_name="image_interpolation", version_base="1.3"
)
def run(cfg: DictConfig):
    run_single_layer(cfg=cfg)


if __name__ == "__main__":
    run()

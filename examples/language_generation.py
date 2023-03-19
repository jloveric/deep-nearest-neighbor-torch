import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.layer import Layer
from deep_nearest_neighbor.metrics import choose_metric
from deep_nearest_neighbor.networks import Network
from deep_nearest_neighbor.utils import generate
import os
from pathlib import Path
from language_interpolation.single_text_dataset import (
    SingleTextDataset,
    generate_dataset,
)
from typing import Callable, Tuple, Any, List
from torch import Tensor
from hydra.utils import get_original_cwd, to_absolute_path
from language_interpolation.utils import (
    create_gutenberg_cache,
)


class TextDataset(SingleTextDataset):
    def __init__(
        self,
        filenames: List[str] = None,
        gutenberg_ids: List[int] = None,
        text: str = None,
        features: int = 10,
        targets: int = 1,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Any, Any]
        ] = generate_dataset,
        num_workers: int = 0,
        add_channel_dimension: bool = False,
        transforms: Callable[[Tensor], Tensor] = None,
    ):
        super().__init__(
            filenames=filenames,
            gutenberg_ids=gutenberg_ids,
            text=text,
            features=features,
            targets=targets,
            max_size=max_size,
            dataset_generator=dataset_generator,
            num_workers=num_workers,
            add_channel_dimension=add_channel_dimension,
            transforms=transforms,
        )

    def __getitem__(self, idx) -> Tensor:
        index = self.valid_ids[idx]
        if torch.is_tensor(index):
            index = index.tolist()

        inputs = self.inputs[index].clone()
        if self.transforms is not None:
            inputs = self.transforms(inputs)
        return inputs.float(), self.output[index]


def get_dataloaders(cfg: DictConfig):
    create_gutenberg_cache(parent_directory=hydra.utils.get_original_cwd())
    training_data = TextDataset(
        gutenberg_ids=cfg.gutenberg_train,
        features=cfg.num_features,
        targets=cfg.num_targets,
        num_workers=0,
        transforms=None,
    )

    test_data = TextDataset(
        gutenberg_ids=cfg.gutenberg_test,
        features=cfg.num_features,
        targets=cfg.num_targets,
        num_workers=0,
        transforms=None,
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )

    return train_dataloader, test_dataloader


def run_network(cfg: DictConfig):
    if cfg.train is True:
        train_dataloader, test_dataloader = get_dataloaders(cfg)

        # print(f"hydra.run.dir", hydra.run.dir)
        print(f"Current working directory : {os.getcwd()}")
        print(f"Orig working directory    : {get_original_cwd()}")

        network = Network(
            dataloader=train_dataloader,
            num_classes=128,
            distance_metric=choose_metric(cfg=cfg),
            device=cfg.device,
            target_accuracy=cfg.target_accuracy,
            max_neighbors=cfg.max_neighbors,
            num_layers=cfg.num_layers
            # max_count=cfg.max_count,
        )

        network.train()

        network.save()

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

    else:
        # layer = Layer(
        #    num_classes=128,
        #    distance_metric=choose_metric(cfg),
        #    device=cfg.device,
        #    target_accuracy=cfg.target_accuracy,
        #    max_neighbors=cfg.max_neighbors,
        # )
        network = Network(
            num_classes=128,
            distance_metric=choose_metric(cfg),
            device=cfg.device,
            target_accuracy=cfg.target_accuracy,
            max_neighbors=cfg.max_neighbors,
        )

        network.load(cfg.directory)

        result = generate(
            model=network, start_text=cfg.text_prompt, length=cfg.text_prediction_length
        )
        print("result", result)


@hydra.main(
    config_path="../config", config_name="language_interpolation", version_base="1.3"
)
def run(cfg: DictConfig):
    run_network(cfg=cfg)


if __name__ == "__main__":
    run()

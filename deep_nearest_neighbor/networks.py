import torch
from torch import Tensor
from deep_nearest_neighbor.layer import Layer, euclidian_distance, Predictor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from pathlib import Path
from typing import Optional


class ForwardLoader:
    """
    Loader that propagates the batch through each layer to the desired index.  Using this
    loader we don't have to save the output dataset for each layer, but it potentially takes
    longer.  However, since this is a one shot or couple shot method, this shouldn't be too
    bad.
    """

    def __init__(
        self,
        network: "Network",
        dataloader: DataLoader,
        layer_index: int,
        device: str = "cuda",
        splits: int = 1,
    ):
        self._network = network
        self._loader_iter = iter(dataloader)
        self._layer_index = layer_index
        self._device = device
        self._splits = splits

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(self._loader_iter)
        x = x.to(self._device)
        y = y.to(self._device)
        # xp = x
        for layer_index in range(self._layer_index):
            x = self._network._layer_list[layer_index].featurize(x, splits=self._splits)
        return x, y


class Network:
    def __init__(
        self,
        num_classes: int,
        dataloader: Optional[DataLoader] = None,
        distance_metric=None,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = 1000,
        num_layers: int = 2,
        splits: int = 1,
    ):
        self._layer_list = []
        for layer_index in range(num_layers):
            layer = Layer(
                num_classes=num_classes,
                distance_metric=distance_metric,
                device=device,
                target_accuracy=target_accuracy,
                max_neighbors=max_neighbors,
            )
            self._layer_list.append(layer)

        self._device = device
        self.dataloader = dataloader
        self._max_neighbors = max_neighbors
        self._splits = splits

    @property
    def num_features(self) -> int:
        return self._layer_list[0].neighbors.shape[1]

    def save(self, directory: str = None):
        if directory is None:
            directory = os.getcwd()

        for index, layer in enumerate(self._layer_list):
            torch.save(layer.neighbors, str(Path(directory) / f"neighbors_{index}.pt"))
            torch.save(
                layer.neighbor_value,
                str(Path(directory) / f"neighbor_value_{index}.pt"),
            )

    def load(self, directory: str = None):
        if directory is None:
            directory = os.getcwd()

        for index, layer in enumerate(self._layer_list):
            layer.neighbors = torch.load(str(Path(directory) / f"neighbors_{index}.pt"))
            layer.neighbor_value = torch.load(
                str(Path(directory) / f"neighbor_value_{index}.pt")
            )

    def layer(self, i: int) -> Layer:
        return self._layer_list[i]

    def forward(self, x: Tensor, to_index: int):
        out = x.to(self._device)

        for layer_index in range(to_index + 1):
            # featurize each internal layer though
            out = self._layer_list[layer_index].featurize(out, splits=self._splits)
            print("out.shape", out.shape)

        return out

    def __call__(self, x: Tensor):
        return self.forward(x, len(self._layer_list) - 1)

    def predict(
        self,
        distances: Tensor,
        target_classification: Tensor,
        style: Predictor = Predictor.Interp,
    ) -> Tensor:
        # Predictions depend on the distances from the last layer.
        return self._layer_list[-1].predict(
            distances=distances,
            target_classification=target_classification,
            style=style,
        )

    def incorrect_predictions(
        self,
        distances: Tensor,
        target_classification: Tensor,
        sample_classification: Tensor,
    ) -> Tensor:
        return self._layer_list[-1].incorrect_predictions(
            distances=distances,
            target_classification=target_classification,
            sample_classification=sample_classification,
        )

    def train(self):
        for count, layer in enumerate(self._layer_list):
            print(f"training layer {count}")
            dataloader = ForwardLoader(
                network=self,
                dataloader=self.dataloader,
                layer_index=count,
                device=self._device,
                splits=self._splits,
            )
            layer.epoch_loop(dataloader=dataloader)

    def test_loop(self, dataloader, only_final: bool = True):
        results = []
        if only_final == True:
            forward_loader = ForwardLoader(
                network=self,
                dataloader=dataloader,
                layer_index=len(self._layer_list),
                device=self._device,
                splits=self._splits,
            )
            layer = self._layer_list[-1]
            print("layer.size", layer.num_neighbors, layer.num_features)
            result = layer.test_loop(dataloader=forward_loader)
            results.append(result)
        else:
            for count, layer in enumerate(self._layer_list):
                print(f"test layer {count}")
                forward_loader = ForwardLoader(
                    network=self,
                    dataloader=dataloader,
                    layer_index=count,
                    device=self._device,
                    splits=self._splits,
                )
                result = layer.test_loop(dataloader=forward_loader)
                results.append(result)

        return results

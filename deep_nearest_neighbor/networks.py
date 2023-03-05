import torch
from torch import Tensor
from deep_nearest_neighbor.layer import Layer, euclidian_distance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ForwardLoader:
    """
    Loader that propagates the batch through each layer to the desired index.  Using this
    loader we don't have to save the output dataset for each layer, but it potentially takes
    longer.  However, since this is a one shot or couple shot method, this shouldn't be too
    bad.
    """

    def __init__(self, network: "Network", dataloader: DataLoader, layer_index: int):
        self._network = network
        self._loader_iter = iter(dataloader)
        self._layer_index = layer_index

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(self._loader_iter)
        for layer in range(self._layer_index):
            x = layer(x)
        return x


class Network:
    def __init__(
        self,
        dataloader: DataLoader,
        num_classes: int,
        distance_metric=euclidian_distance,
        device: str = "cuda",
        target_accuracy: float = 0.9,
    ):
        self.layer1 = Layer(
            num_classes=num_classes,
            distance_metric=euclidian_distance,
            device=device,
            target_accuracy=target_accuracy,
        )
        self.layer2 = Layer(
            num_classes=num_classes,
            distance_metric=euclidian_distance,
            device=device,
            target_accuracy=target_accuracy,
        )

        self._layer_list = [self.layer1, self.layer2]
        self.dataloader = dataloader

    def forward(self, x: Tensor, to_index: int):
        out = x
        for layer_index in range(to_index + 1):
            out = self._layer_list[layer_index](out)

        return out

    def predict(self):
        pass

    def train(self):
        for count, layer in enumerate(self._layer_list):
            dataloader = ForwardLoader(
                network=self, dataloader=self.dataloader, layer_index=count
            )
            layer.epoch_loop(dataloader=dataloader)

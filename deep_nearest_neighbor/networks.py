import torch
from torch import Tensor
from deep_nearest_neighbor.layer import Layer, euclidian_distance, Predictor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
    ):
        self._network = network
        self._loader_iter = iter(dataloader)
        self._layer_index = layer_index
        self._device = device

    def __iter__(self):
        return self
        """
        for data in self._loader_iter:
            x, y = data
            x = x.to(self._device)
            y = y.to(self._device)
            for layer_index in range(self._layer_index):
                print("loading layer", layer_index)
                x = self._network._layer_list[layer_index](x)
            yield x, y
        """

    def __next__(self):
        x, y = next(self._loader_iter)
        x = x.to(self._device)
        y = y.to(self._device)
        for layer_index in range(self._layer_index):
            # print("loading layer", layer_index)
            x = self._network._layer_list[layer_index](x)
        return x, y


class Network:
    def __init__(
        self,
        dataloader: DataLoader,
        num_classes: int,
        distance_metric=euclidian_distance,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = 1000,
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
        self._device = device
        self._layer_list = [self.layer1, self.layer2]
        self.dataloader = dataloader
        self._max_neighbors = max_neighbors

    def forward(self, x: Tensor, to_index: int):
        out = x.to(self._device)
        for layer_index in range(to_index + 1):
            out = self._layer_list[layer_index](out)

        return out

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
            )
            layer.epoch_loop(dataloader=dataloader, max_neighbors=self._max_neighbors)

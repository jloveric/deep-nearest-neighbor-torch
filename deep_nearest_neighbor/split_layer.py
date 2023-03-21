import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import NamedTuple, Tuple, Union
import time
from torch import Tensor
from enum import Enum
from pathlib import Path
import os
from deep_nearest_neighbor.metrics import euclidian_distance
from deep_nearest_neighbor.layer import Layer, CommonMixin


class Results(NamedTuple):
    error: float
    accuracy: float
    incorrect: int
    total: int


class Layer(CommonMixin):
    def __init__(
        self,
        num_classes: int,
        distance_metric=None,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = float("inf"),
        max_count: int = 3,
        num_splits: int = 2,
    ):
        self._split = [
            Layer(
                num_classes=num_classes,
                distance_metric=distance_metric,
                device=device,
                target_accuracy=target_accuracy,
                max_neighbors=max_neighbors,
                max_count=max_count,
            )
            for _ in range(num_splits)
        ]

    def predict(
        self,
        distances: Tensor,
        target_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param distances: inverse distances between samples and all neighbors
        :param target_value: classification of each neighbor
        :param style: style of prediction (nearest neighbor, all sum)
        :returns: predicted classification for each sample
        """
        
        return predictions, probabilities

    def incorrect_predictions(
        self,
        distances: Tensor,
        target_value: Tensor,
        sample_classification: Tensor,
    ) -> Tensor:
        """
        Compute the sample classifications that did not match the predicted
        classifications.  Return the indices that were computed incorrectly
        :param distance: Distances from each of the neighbors
        :param target_value: The classification for each of those neighbors
        :param sample_classification: The classification for each of the samples
        """
        

    def train_loop(
        self,
        samples,
        sample_class,
        target_accuracy: float = 0.9,
    ):
        return self._neighbors, self._neighbor_value

    def test_wrong(
        self,
        neighbors: Tensor,
        neighbor_value: Tensor,
        samples: Tensor,
        sample_class: Tensor,
    ) -> int:
        """
        Return the number of elements that are wrong in samples
        """

        return 

    def test_loop(self, dataloader: DataLoader) -> Results:
        

        return Results(
            error=wrong / datapoints,
            accuracy=(datapoints - wrong) / datapoints,
            incorrect=wrong,
            total=datapoints,
        )

    def epoch_loop(self, dataloader: DataLoader) -> Tensor:
        
        return self._neighbors, self._neighbor_value

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import NamedTuple
import time
from torch import Tensor
from enum import Enum


# class syntax
class Predictor(Enum):
    Nearest = 0
    Interp = 1


class DistanceMetric(Enum):
    Euclidian = 0
    Cosine = 1


class Results(NamedTuple):
    error: float
    accuracy: float
    incorrect: int
    total: int


def euclidian_distance(
    keys: Tensor, values: Tensor, epsilon: float = 1e-3, exponent: float = 4.0
) -> Tensor:
    """
    Compute the inverse distance from each of the neighbors and store that
    in the returning vector.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """
    if False:  # lower memory
        distances = []
        for value in values:
            delta = keys - value
            dist = 1 / torch.pow((torch.linalg.norm(delta, dim=1) + epsilon), 8)
            distances.append(dist)

        res = torch.stack(distances)
        return res
    else:
        delta = values.unsqueeze(1) - keys
        distance = 1 / torch.pow((torch.linalg.norm(delta, dim=2) + epsilon), exponent)
        # print("distance", distance)
        # distance = torch.nan_to_num(
        #    torch.nn.functional.normalize(distance, dim=1), nan=0.0, posinf=1.0
        # )
        # print("distance", distance)
        # distance = 1 / torch.pow((torch.linalg.norm(delta, dim=2) + epsilon), 8)

        return distance


def cosine_distance(
    keys: Tensor, values: Tensor, epsilon: float = 1e-6, exponent: float = 2.0
) -> Tensor:
    """
    Compute the inverse distance from each of the neighbors and store that
    in the returning vector.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    distances = (
        torch.nn.functional.normalize(values) @ torch.nn.functional.normalize(keys).t()
    )
    distances = 1 / torch.pow(1 - torch.abs(distances) + epsilon, exponent)
    return distances


class Layer:
    def __init__(
        self,
        num_classes: int,
        distance_metric=euclidian_distance,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = float("inf"),
    ):
        self._distance_metric = distance_metric
        self._neighbors: torch.Tensor = None
        self._neighbor_class: torch.Tensor = None
        self._device = device
        self._target_accuracy = target_accuracy
        self._num_classes = num_classes
        self._max_neighbors = max_neighbors

    @property
    def neighbors(self) -> Tensor:
        return self._neighbors

    @property
    def neighbor_class(self) -> Tensor:
        return self._neighbor_class

    def predict(
        self,
        distances: Tensor,
        target_classification: Tensor,
        style: Predictor = Predictor.Interp,
    ) -> Tensor:
        """
        :param distances: inverse distances between samples and all neighbors
        :param target_classification: classification of each neighbor
        :param style: style of prediction (nearest neighbor, all sum)
        :returns: predicted classification for each sample
        """
        if style == Predictor.Nearest:
            nearest_neighbor = torch.argmax(distances, dim=1)
            predicted_classification = target_classification[nearest_neighbor]
        elif style == Predictor.Interp:
            predicted_classification = 0
            predicted_sum = 0
            for i in range(self._num_classes):
                indexes = (target_classification == i).nonzero().squeeze()

                # TODO: When numel is one 1 I lose a dimension and stuff breaks. Figure out
                # a better approach here.
                if indexes.numel() == 1:
                    indexes = indexes.unsqueeze(0)

                if indexes.numel() > 0:
                    this_sum = torch.sum(distances[:, indexes], dim=1)

                    predicted_sum = torch.where(
                        predicted_sum > this_sum, predicted_sum, this_sum
                    )
                    predicted_classification = torch.where(
                        predicted_sum > this_sum, predicted_classification, i
                    )
        return predicted_classification

    def incorrect_predictions(
        self,
        distances: Tensor,
        target_classification: Tensor,
        sample_classification: Tensor,
    ) -> Tensor:
        """
        Compute the sample classifications that did not match the predicted
        classifications.  Return the indices that were computed incorrectly
        :param distance: Distances from each of the neighbors
        :param target_classification: The classification for each of those neighbors
        :param sample_classification: The classification for each of the samples
        """
        # nearest_neighbor = torch.argmin(distances, dim=1)
        # predicted_classification = target_classification[nearest_neighbor]
        predicted_classification = self.predict(
            distances=distances, target_classification=target_classification
        )

        all = predicted_classification == sample_classification
        wrong_indices = torch.logical_not(all).nonzero().squeeze()
        if wrong_indices.numel() == 1:
            wrong_indices = wrong_indices.unsqueeze(dim=0)

        return wrong_indices

    def extend_neighbors(
        self,
        new_keys: Tensor,
        new_class: Tensor,
    ) -> None:
        self._neighbors = torch.cat([self._neighbors, new_keys], dim=0)
        self._neighbor_class = torch.cat([self._neighbor_class, new_class], dim=0)

    def train_loop(
        self,
        samples,
        sample_class,
        target_accuracy: float = 0.9,
    ):
        result = 0
        count = 0
        while (result < target_accuracy) and (
            len(self._neighbors) <= self._max_neighbors
        ):
            # print("result", result, "count", count)
            distances = self._distance_metric(keys=self._neighbors, values=samples)

            wrong_indices = self.incorrect_predictions(
                distances=distances,
                target_classification=self._neighbor_class,
                sample_classification=sample_class,
            )

            if wrong_indices.numel() > 0:
                new_keys = samples[wrong_indices]
                new_class = sample_class[wrong_indices]

                self.extend_neighbors(new_keys, new_class)

            final_distances = self._distance_metric(
                keys=self._neighbors, values=samples
            )
            final_predictions = self.predict(
                distances=final_distances, target_classification=self._neighbor_class
            )
            how_good = final_predictions == sample_class

            result = torch.sum(how_good) / how_good.shape[0]
            count += 1

        return self._neighbors, self._neighbor_class

    def test_wrong(
        self,
        neighbors: Tensor,
        neighbor_class: Tensor,
        samples: Tensor,
        sample_class: Tensor,
    ) -> int:
        """
        Return the number of elements that are wrong in samples
        """

        distances = self._distance_metric(keys=neighbors, values=samples)

        wrong_indices = self.incorrect_predictions(
            distances=distances,
            target_classification=neighbor_class,
            sample_classification=sample_class,
        )

        return wrong_indices.numel()

    def test_loop(self, dataloader: DataLoader) -> Results:
        wrong = 0
        datapoints = 0

        t_start = time.perf_counter()
        for data in tqdm(iter(dataloader)):
            x, y = data
            x = x.to(self._device)
            y = y.to(self._device)

            datapoints += len(x)
            wrong += self.test_wrong(
                neighbors=self._neighbors,
                neighbor_class=self._neighbor_class,
                samples=x,
                sample_class=y,
            )

        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")

        return Results(
            error=wrong / datapoints,
            accuracy=(datapoints - wrong) / datapoints,
            incorrect=wrong,
            total=datapoints,
        )

    def epoch_loop(self, dataloader: DataLoader) -> Tensor:
        data_iter = iter(dataloader)

        if self._neighbors == None:
            # Set the neighbors to the first batch
            self._neighbors, self._neighbor_class = next(data_iter)
            # print(
            #    "creating first batch",
            #    self._neighbors.shape,
            #    self._neighbor_class.shape,
            # )

        self._neighbors = self._neighbors.to(self._device)
        self._neighbor_class = self._neighbor_class.to(self._device)
        t_start = time.perf_counter()
        for count, data in enumerate(pbar := tqdm(data_iter)):
            pbar.set_postfix({"neighbors": len(self._neighbors)})
            # print("count", count)
            x, y = data
            x = x.to(self._device)
            y = y.to(self._device)

            # print("x", x, "y", y)
            self._neighbors, self._neighbor_class = self.train_loop(
                samples=x,
                sample_class=y,
                target_accuracy=self._target_accuracy,
            )
        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")
        print(f"Network neighbors {len(self._neighbors)}")

        return self._neighbors, self._neighbor_class

    def __call__(self, x):
        """
        This does not make a prediction, just gives distances to each neighbor, so
        it's a new set of features similar to what happens in a neural network.
        """
        return self._distance_metric(self._neighbors, x)


class RegressionLayer:
    def __init__(
        self,
        distance_metric=euclidian_distance,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = float("inf"),
        tolerance: float = 0.05,
    ):
        self._distance_metric = distance_metric
        self._neighbors: torch.Tensor = None
        self._neighbor_value: torch.Tensor = None
        self._device = device
        self._target_accuracy = target_accuracy
        self._max_neighbors = max_neighbors
        self._tolerance = tolerance

    @property
    def neighbors(self) -> Tensor:
        return self._neighbors

    @property
    def neighbor_value(self) -> Tensor:
        return self._neighbor_value

    def predict(
        self,
        distances: Tensor,
        target_value: Tensor,
    ) -> Tensor:
        """
        :param distances: inverse distances between samples and all neighbors
        :param target_classification: classification of each neighbor
        :param style: style of prediction (nearest neighbor, all sum)
        :returns: predicted classification for each sample
        """

        norm_distances = torch.nn.functional.normalize(
            distances, p=1.0
        )  # should sum to 1
        interpolated_value = torch.matmul(norm_distances, target_value)

        return interpolated_value

    def incorrect_predictions(
        self,
        distances: Tensor,
        target_value: Tensor,
        sample_value: Tensor,
    ) -> Tensor:
        """
        Compute the sample classifications that did not match the predicted
        classifications.  Return the indices that were computed incorrectly
        :param distance: Distances from each of the neighbors
        :param target_classification: The classification for each of those neighbors
        :param sample_classification: The classification for each of the samples
        """

        predicted_values = self.predict(distances=distances, target_value=target_value)

        all = torch.abs(predicted_values - sample_value)
        wrong_indices = (
            torch.where(all < self._tolerance, True, False).nonzero().squeeze()
        )
        if wrong_indices.numel() == 1:
            wrong_indices = wrong_indices.unsqueeze(dim=0)

        return wrong_indices

    def extend_neighbors(
        self,
        new_keys: Tensor,
        new_class: Tensor,
    ) -> None:
        self._neighbors = torch.cat([self._neighbors, new_keys], dim=0)
        self._neighbor_class = torch.cat([self._neighbor_class, new_class], dim=0)

    def train_loop(
        self,
        samples,
        sample_values,
        target_accuracy: float = 0.9,
    ):
        result = 0
        count = 0
        while (result < target_accuracy) and (
            len(self._neighbors) <= self._max_neighbors
        ):
            distances = self._distance_metric(keys=self._neighbors, values=samples)

            wrong_indices = self.incorrect_predictions(
                distances=distances,
                target_value=self._neighbor_value,
                sample_value=sample_values,
            )

            if wrong_indices.numel() > 0:
                new_keys = samples[wrong_indices]
                new_values = sample_values[wrong_indices]

                self.extend_neighbors(new_keys, new_values)

            final_distances = self._distance_metric(
                keys=self._neighbors, values=samples
            )
            final_predictions = self.predict(
                distances=final_distances, target_values=self._neighbor_value
            )
            how_good = torch.abs(final_predictions - sample_values)

            result = torch.sum(how_good) / how_good.shape[0]
            count += 1

        return self._neighbors, self._neighbor_class

    def test_wrong(
        self,
        neighbors: Tensor,
        neighbor_class: Tensor,
        samples: Tensor,
        sample_class: Tensor,
    ) -> int:
        """
        Return the number of elements that are wrong in samples
        """

        distances = self._distance_metric(keys=neighbors, values=samples)

        wrong_indices = self.incorrect_predictions(
            distances=distances,
            target_classification=neighbor_class,
            sample_classification=sample_class,
        )

        return wrong_indices.numel()

    def test_loop(self, dataloader: DataLoader) -> Results:
        wrong = 0
        datapoints = 0

        t_start = time.perf_counter()
        for data in tqdm(iter(dataloader)):
            x, y = data
            x = x.to(self._device)
            y = y.to(self._device)

            datapoints += len(x)
            wrong += self.test_wrong(
                neighbors=self._neighbors,
                neighbor_class=self._neighbor_class,
                samples=x,
                sample_class=y,
            )

        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")

        return Results(
            error=wrong / datapoints,
            accuracy=(datapoints - wrong) / datapoints,
            incorrect=wrong,
            total=datapoints,
        )

    def epoch_loop(self, dataloader: DataLoader) -> Tensor:
        data_iter = iter(dataloader)

        if self._neighbors == None:
            # Set the neighbors to the first batch
            self._neighbors, self._neighbor_class = next(data_iter)
            # print(
            #    "creating first batch",
            #    self._neighbors.shape,
            #    self._neighbor_class.shape,
            # )

        self._neighbors = self._neighbors.to(self._device)
        self._neighbor_class = self._neighbor_class.to(self._device)
        t_start = time.perf_counter()
        for count, data in enumerate(pbar := tqdm(data_iter)):
            pbar.set_postfix({"neighbors": len(self._neighbors)})
            # print("count", count)
            x, y = data
            x = x.to(self._device)
            y = y.to(self._device)

            # print("x", x, "y", y)
            self._neighbors, self._neighbor_class = self.train_loop(
                samples=x,
                sample_class=y,
                target_accuracy=self._target_accuracy,
            )
        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")
        print(f"Network neighbors {len(self._neighbors)}")

        return self._neighbors, self._neighbor_class

    def __call__(self, x):
        """
        This does not make a prediction, just gives distances to each neighbor, so
        it's a new set of features similar to what happens in a neural network.
        """
        return self._distance_metric(self._neighbors, x)

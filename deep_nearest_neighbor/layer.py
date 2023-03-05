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


class Results(NamedTuple):
    error: float
    accuracy: float
    incorrect: int
    total: int


def layer(keys: Tensor, values: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Compute the inverse distance from each of the neighbors and store that
    in the returning vector.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """
    distances = []
    for value in values:
        delta = keys - value
        dist = 1 / (torch.linalg.norm(delta, dim=1) + epsilon)
        distances.append(dist)

    res = torch.stack(distances)
    return res


def predict(
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
        for i in range(10):  # TODO: don't want to hard code number of classes
            indexes = (target_classification == i).nonzero()
            this_sum = torch.sum(distances[indexes], dim=1)
            predicted_sum = torch.where(
                predicted_sum > this_sum, predicted_sum, this_sum
            )
            predicted_classification = torch.where(
                predicted_sum > this_sum, predicted_classification, i
            )

    return predicted_classification


def incorrect_predictions(
    distances: Tensor, target_classification: Tensor, sample_classification: Tensor
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
    predicted_classification = predict(
        distances=distances, target_classification=target_classification
    )

    all = predicted_classification == sample_classification
    wrong_indices = torch.logical_not(all).nonzero().squeeze()
    if wrong_indices.numel() == 1:
        wrong_indices = wrong_indices.unsqueeze(dim=0)

    return wrong_indices


def extend_neighbors(
    keys: Tensor,
    key_class: Tensor,
    new_keys: Tensor,
    new_class: Tensor,
) -> Tensor:
    ext_keys = torch.cat([keys, new_keys], dim=0)
    ext_class = torch.cat([key_class, new_class], dim=0)
    return ext_keys, ext_class


def train_loop(
    neighbors, neighbor_class, samples, sample_class, target_accuracy: float = 0.9
):
    result = 0

    while result < target_accuracy:
        distances = layer(keys=neighbors, values=samples)

        wrong_indices = incorrect_predictions(
            distances=distances,
            target_classification=neighbor_class,
            sample_classification=sample_class,
        )

        if wrong_indices.numel() > 0:
            new_keys = samples[wrong_indices]
            new_class = sample_class[wrong_indices]

            neighbors, neighbor_class = extend_neighbors(
                neighbors, neighbor_class, new_keys, new_class
            )

        final_distances = layer(keys=neighbors, values=samples)
        final_predictions = predict(
            distances=final_distances, target_classification=neighbor_class
        )

        how_good = final_predictions == sample_class
        result = torch.sum(how_good) / how_good.shape[0]

    return neighbors, neighbor_class


def test_wrong(
    neighbors: Tensor,
    neighbor_class: Tensor,
    samples: Tensor,
    sample_class: Tensor,
) -> int:
    """
    Return the number of elements that are wrong in samples
    """

    distances = layer(keys=neighbors, values=samples)

    wrong_indices = incorrect_predictions(
        distances=distances,
        target_classification=neighbor_class,
        sample_classification=sample_class,
    )

    return wrong_indices.numel()


def test_loop(
    neighbors, neighbor_class, dataloader: DataLoader, device: str = "cpu"
) -> Results:
    wrong = 0
    datapoints = 0

    t_start = time.perf_counter()
    for data in tqdm(iter(dataloader)):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        datapoints += len(x)
        wrong += test_wrong(
            neighbors=neighbors,
            neighbor_class=neighbor_class,
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


def epoch_loop(
    dataloader: DataLoader, target_accuracy=0.9, device: str = "cpu"
) -> Tensor:
    data_iter = iter(dataloader)

    neighbors, neighbor_class = next(data_iter)
    neighbors = neighbors.to(device)
    neighbor_class = neighbor_class.to(device)
    t_start = time.perf_counter()
    for data in tqdm(data_iter):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        neighbors, neighbor_class = train_loop(
            neighbors=neighbors,
            neighbor_class=neighbor_class,
            samples=x,
            sample_class=y,
            target_accuracy=target_accuracy,
        )
    t_total = time.perf_counter() - t_start
    print(f"Epoch_loop time {t_total}")

    return neighbors, neighbor_class

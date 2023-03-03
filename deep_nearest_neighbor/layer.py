import torch
from torch.utils.data import DataLoader


def layer(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    distances = []
    for value in values:
        delta = keys - value
        dist = torch.linalg.norm(delta, dim=1)
        distances.append(dist)

    res = torch.stack(distances)
    return res


def predict(distances, target_classification):
    nearest_neighbor = torch.argmin(distances, dim=1)
    predicted_classification = target_classification[nearest_neighbor]
    return predicted_classification


def incorrect_predictions(
    distances, target_classification, sample_classification
) -> torch.Tensor:
    """
    Compute the sample classifications that did not match the predicted
    classifications.  Return the indices that were computed incorrectly
    :param distance: Distances from each of the neighbors
    :param target_classification: The classification for each of those neighbors
    :param sample_classification: The classification for each of the samples
    """
    nearest_neighbor = torch.argmin(distances, dim=1)
    predicted_classification = target_classification[nearest_neighbor]

    all = predicted_classification == sample_classification
    wrong_indices = torch.logical_not(all).nonzero().squeeze()

    return wrong_indices


def extend_neighbors(
    keys: torch.Tensor,
    key_class: torch.Tensor,
    new_keys: torch.Tensor,
    new_class: torch.Tensor,
) -> torch.Tensor:
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


def ecoch_loop(dataloader: DataLoader, target_accuracy=0.9):
    data_iter = iter(dataloader)

    neighbors, neighbor_class = next(data_iter)

    for data in data_iter:
        x, y = data
        
        neighbors, neighbor_class = train_loop(
            neighbors=neighbors,
            neighbor_class=neighbor_class,
            sample=x,
            sample_class=y,
            target_accuracy=target_accuracy,
        )

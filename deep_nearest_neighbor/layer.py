import torch


def layer(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    distances = []
    for value in values:
        delta = keys - value
        dist = torch.linalg.norm(delta, dim=1)
        distances.append(dist)

    res = torch.stack(distances)
    return res


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


def extend_keys(keys: torch.Tensor, new_keys: torch.Tensor) -> torch.Tensor:
    keys = torch.cat([keys, new_keys],dim=0)
    return keys

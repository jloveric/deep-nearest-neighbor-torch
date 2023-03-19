import torch
from torch import Tensor


def influence_cone(
    keys: Tensor,
    values: Tensor,
    epsilon: float = 1e-3,
    exponent: float = 4.0,
    factor: float = 2.0,
) -> Tensor:
    """
    Compute the inverse distance from each of the neighbors and store that
    in the returning vector.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    delta = values.unsqueeze(1) - keys
    distance = torch.linalg.norm(delta, dim=2) + epsilon

    # Get the minimum distances
    min_dist, _ = torch.min(distance, dim=1)
    min_dist = min_dist.view(-1, 1)
    # print('min_dist', min_dist.shape)
    # print('distance', distance.shape)
    cone = torch.clamp(-distance + factor * min_dist, min=0)

    return cone


class InfluenceCone:
    def __init__(self, epsilon: float = 1e-3, exponent: float = 2, factor: float = 2):
        self._epsilon = epsilon
        self._exponent = exponent
        self._factor = factor

    def __call__(self, keys: Tensor, values: Tensor):
        return influence_cone(
            keys=keys,
            values=values,
            epsilon=self._epsilon,
            exponent=self._exponent,
            factor=self._factor,
        )


def euclidian_pyramid_distance(
    keys: Tensor,
    values: Tensor,
    epsilon: float = 1e-3,
    exponent: float = -4.0,
    scales: int = 4,
) -> Tensor:
    """
    Compute distances based off of a pyramid, first at full scale, then half
    then quarter... This is a 1 dimensional pyramid, not the typical 2d pyramid
    used in image problems.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    delta = values.unsqueeze(1) - keys

    distances = 0
    new_delta = delta

    for i in range(scales):
        distances += torch.pow(
            (torch.linalg.norm(new_delta, dim=2) + epsilon), exponent
        )
        new_delta = new_delta[:, :, ::2]

    return distances / scales


def euclidian_distance(
    keys: Tensor, values: Tensor, epsilon: float = 1e-3, exponent: float = -4.0
) -> Tensor:
    """
    Compute the inverse distance from each of the neighbors and store that
    in the returning vector.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    delta = values.unsqueeze(1) - keys
    distance = torch.pow((torch.linalg.norm(delta, dim=2) + epsilon), exponent)
    # print("distance", distance)
    # distance = torch.nan_to_num(
    #    torch.nn.functional.normalize(distance, dim=1), nan=0.0, posinf=1.0
    # )
    # print("distance", distance)
    # distance = 1 / torch.pow((torch.linalg.norm(delta, dim=2) + epsilon), 8)

    return distance


def cosine_distance(
    keys: Tensor, values: Tensor, epsilon: float = 1e-6, exponent: float = -2.0
) -> Tensor:
    """
    Compute the cosine distance.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    distances = (
        torch.nn.functional.normalize(values) @ torch.nn.functional.normalize(keys).t()
    )
    distances = torch.pow(1 - torch.abs(distances) + epsilon, exponent)
    return distances


def dot_product(
    keys: Tensor, values: Tensor, epsilon: float = 1e-6, exponent: float = -2.0
) -> Tensor:
    """
    Compute the dot product distance.
    :param keys: tensor containing all neighbors
    :param values: tensor containing all samples
    :param epsilon: factor so the inverse doesn't become infinite
    :return: inverse distance between keys and values
    """

    distances = values @ keys.t()
    distances = torch.pow(1 - torch.abs(distances) + epsilon, exponent)
    return distances


class EuclidianDistance:
    def __init__(self, epsilon: float = 1e-3, exponent: float = -2):
        self._epsilon = epsilon
        self._exponent = exponent

    def __call__(self, keys: Tensor, values: Tensor):
        return euclidian_distance(
            keys=keys, values=values, epsilon=self._epsilon, exponent=self._exponent
        )


class EuclidianPyramidDistance:
    def __init__(self, epsilon: float = 1e-3, exponent: float = -2, scales: int = 4):
        self._epsilon = epsilon
        self._exponent = exponent
        self._scales = scales

    def __call__(self, keys: Tensor, values: Tensor):
        return euclidian_pyramid_distance(
            keys=keys,
            values=values,
            epsilon=self._epsilon,
            exponent=self._exponent,
            scales=self._scales,
        )


class CosineDistance:
    def __init__(self, epsilon: float = 1e-3, exponent: float = -2):
        self._epsilon = epsilon
        self._exponent = exponent

    def __call__(self, keys: Tensor, values: Tensor):
        return cosine_distance(
            keys=keys, values=values, epsilon=self._epsilon, exponent=self._exponent
        )

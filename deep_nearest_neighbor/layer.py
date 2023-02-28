import torch


def layer(keys: torch.Tensor, values: torch.Tensor, metric=lambda x,y : torch.dot(x,y)):

    dist = metric(keys, values)
    return dist            



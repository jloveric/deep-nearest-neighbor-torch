import torch


def layer(keys: torch.Tensor, values: torch.Tensor):
    
    distances = []
    for value in values :
        delta = keys-value
        dist = torch.linalg.norm(delta,dim=1)
        distances.append(dist)


    res = torch.stack(distances)
    return res


def new_keys(distances, categories) :
    predictions = torch.argmin(distances,dim=1)
    print('predictions', predictions.shape, categories.shape)
    res = (predictions==categories)

    
    print('res', res)
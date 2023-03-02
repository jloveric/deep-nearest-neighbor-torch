import torch


def layer(keys: torch.Tensor, values: torch.Tensor):
    
    distances = []
    for value in values :
        delta = keys-value
        dist = torch.linalg.norm(delta,dim=1)
        distances.append(dist)


    res = torch.stack(distances)
    return res


def new_keys(distances, target_classification, sample_classification ) :
    nearest_neighbor = torch.argmin(distances,dim=1)
    predicted_classification = target_classification[nearest_neighbor]

    print('predictions', predicted_classification)
    res = (predicted_classification==sample_classification)

    
    print('res', res)
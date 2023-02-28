from deep_nearest_neighbor.layer import layer
import torch

def test_layer() :
    keys = torch.rand(5,10)
    values = torch.rand(20,10)

    ans = layer(keys=keys, values=values)
    print('ans', ans)

    return ans

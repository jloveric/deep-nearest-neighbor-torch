from deep_nearest_neighbor.layer import layer, new_keys
import torch


def test_layer():
    keys = torch.rand(5, 10)
    key_classification = torch.randint(low=0, high=5, size=(keys.shape[0],))

    values = torch.rand(20, 10)
    value_classification = torch.randint(low=0, high=5, size=(values.shape[0],))

    ans = layer(keys=keys, values=values)
    print("ans", ans)

    final = new_keys(
        distances=ans,
        target_classification=key_classification,
        sample_classification=value_classification,
    )
    print("final", final)

    return ans

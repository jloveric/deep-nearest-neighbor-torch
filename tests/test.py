from deep_nearest_neighbor.layer import (
    layer,
    incorrect_predictions,
    extend_neighbors,
    predict,
)
import torch


def test_layer():
    keys = torch.rand(5, 10)
    print("keys.shape", keys.shape)

    key_classification = torch.randint(low=0, high=2, size=(keys.shape[0],))

    values = torch.rand(20, 10)
    value_classification = torch.randint(low=0, high=2, size=(values.shape[0],))

    ans = layer(keys=keys, values=values)

    wrong_indices = incorrect_predictions(
        distances=ans,
        target_classification=key_classification,
        sample_classification=value_classification,
    )

    new_keys = values[wrong_indices]
    new_classification = value_classification[wrong_indices]
    print("new_keys.shape", new_keys.shape)

    keys, key_class = extend_neighbors(
        keys, key_classification, new_keys, new_classification
    )

    print("keys.shape", keys.shape)
    assert keys.shape[1] == 10
    assert keys.shape[0] >= 5

    final_distances = layer(keys=keys, values=values)
    final_predictions = predict(distances=final_distances, target_classification=key_class)

    result = final_predictions == value_classification
    print('result', torch.sum(result)/result.shape[0])

import torch
import random


def generate(model, start_text, length, device: str = "cuda") -> str:
    num_features = model.num_features
    text = start_text
    for i in range(length):
        feature_set = text[-num_features:]
        x = torch.tensor([[ord(val) for val in feature_set]]).float().to(device)
        # print("x", x)
        p, probabilities = model(x)

        # actual = random.choices(ascii, probabilities.flatten().tolist())

        p = chr(p[0].item())
        text += p

    return text

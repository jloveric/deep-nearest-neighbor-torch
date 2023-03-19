import torch
import random


def generate(model, start_text, length, device: str = "cuda") -> str:
    num_features = model.num_features
    text = start_text
    ascii = list(range(128))
    for i in range(length):
        feature_set = text[-num_features:]
        x = torch.tensor([[ord(val) for val in feature_set]]).float().to(device)
        # print("x", x)
        output = model(x)
        # p = torch.argmax(output, dim=1)

        p = random.choices(ascii, output.flatten().tolist())

        # print("p", p)
        character = chr(p[0])
        text += character

    return text

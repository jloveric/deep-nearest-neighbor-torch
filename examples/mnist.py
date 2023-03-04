import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from deep_nearest_neighbor.layer import train_loop, epoch_loop

class TransformFlat :

    def __init__(self) :
        self.to_tensor = ToTensor()
    
    def __call__(self, input_data) :
        data = self.to_tensor(input_data).flatten()
        return data

training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=TransformFlat()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=TransformFlat()
)


def collate_fn(batch):
    return batch
    print('batch', len(batch))
    print('batch[]',batch[0])
    x, y = batch
    return x.resize(x.shape[0], -1), y


train_dataloader = DataLoader(
    training_data, batch_size=64, shuffle=True,
)
test_dataloader = DataLoader(
    test_data, batch_size=64, shuffle=True,
)


@hydra.main(config_path="../config", config_name="mnist", version_base="1.3")
def run(cfg: DictConfig):
    print("inside here", flush=True)
    epoch_loop(dataloader=train_dataloader, target_accuracy=cfg.target_accuracy)


if __name__ == "__main__":
    run()

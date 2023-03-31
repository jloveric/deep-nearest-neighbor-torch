from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import get_original_cwd


class TransformFlat:
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, input_data):
        data = self.to_tensor(input_data).flatten()
        return data


def choose_dataset(cfg: DictConfig):
    mnist_data_path = str(Path(get_original_cwd()) / "data")

    num_classes = None
    if cfg.data == "mnist":
        num_classes = 10
        training_data = datasets.MNIST(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.MNIST(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )
    elif cfg.data == "cifar100":
        num_classes = 100
        training_data = datasets.CIFAR100(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.CIFAR100(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )
    elif cfg.data == "cifar10":
        num_classes = 10
        training_data = datasets.CIFAR10(
            root=mnist_data_path,
            train=True,
            download=True,
            transform=TransformFlat(),
        )

        test_data = datasets.CIFAR10(
            root=mnist_data_path, train=False, download=True, transform=TransformFlat()
        )

    return training_data, test_data, num_classes

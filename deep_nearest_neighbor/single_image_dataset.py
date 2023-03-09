# from PIL import Image
from matplotlib import image
import torch
from torch import Tensor
import numpy as np
import math
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def image_to_dataset(filename: str, device="cpu"):
    """
    Read in an image file and return the flattened position input
    flattened output and torch array of the original image.def image_to_dataset(filename: str, peano: str = False, rotations: int = 1):

    Args :
        filename : image filename.
    Returns :
        flattened image [width*heigh, rgb], flattened position vectory
        [width*height, 2] and torch tensor of original image.
    """
    img = image.imread(filename)

    torch_image = torch.from_numpy(np.array(img)).float()
    shape = torch_image.shape
    pixels = shape[0] * shape[1]

    x = torch.arange(torch_image.shape[0], dtype=torch.float)
    y = torch.arange(torch_image.shape[1], dtype=torch.float)
    mesh = torch.meshgrid(x, y, indexing="ij")
    print('mesh', mesh)

    return (
        torch_image.reshape(pixels, -1),
        torch.cat([mesh[0].unsqueeze(2), mesh[1].unsqueeze(2)], dim=2).reshape(
            pixels, -1
        ),
        torch_image,
    )


class ImageDataset(Dataset):
    def __init__(self, filename: str):
        self.output, self.input, self.image = image_to_dataset(filename)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]

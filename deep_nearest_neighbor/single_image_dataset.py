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


def image_to_dataset(
    filename: str, peano: str = False, rotations: int = 1, device="cpu"
):
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

    torch_image = torch.from_numpy(np.array(img))

    return torch_image_flat, torch_position, torch_image


class ImageDataset(Dataset):
    def __init__(self, filenames: List[str], rotations: int = 1):
        self.output, self.input, self.image = image_to_dataset(
            filenames[0], rotations=rotations
        )

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]
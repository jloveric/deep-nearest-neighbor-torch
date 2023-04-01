from pytorch_lightning import LightningModule
from deep_nearest_neighbor.metrics import CosineDistance

import torch.nn as nn
import torch.nn.functional as F
import torch


class DeepNearestNeighborLayer(LightningModule):
    """
    Learn the transform that leads to the best nearest neighbor
    approximation where the approximation metric is
    a distance metric such as euclidean or cosine with
    some sort of inverse distance weighting.
    After the transform has been computed, it can be used
    to generate "neighbors" to which the distance metric
    is applied.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_classes: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features)
        self._centers = torch.tensor([])
        self._values = torch.tensor([])
        self._transformed_centers = torch.tensor([])
        self._transformed_values = torch.tensor([])
        self._distance_metric = CosineDistance(epsilon=1e-2, exponent=-2)
        self._num_classes = num_classes
        self._device = device

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def add_keys(self, x):
        keys = self.l1(x)

    def training_step(self, batch, batch_idx):
        print("batch", batch[0])
        x, y = batch[0]

        size = x.shape[0] // 2
        xa = x[:size, ...]
        xb = x[size:, ...]
        ya = y[:size, ...]
        yb = y[size:, ...]

        txa = self.l1(xa)
        txb = self.l1(xb)

        # calculate the nearest neighbors a->b and b->a. We
        # reverse the role of keys and values to get the most
        # out of the data.
        ans_a = self._distance_metric(keys=txa, values=txb)
        ans_b = self._distance_metric(keys=txb, values=txa)

        probabilities_a = self.predict(distances=ans_a, target_value=ya)
        probabilities_b = self.predict(distances=ans_b, target_value=yb)

        loss = F.cross_entropy(probabilities_a, ya) + F.cross_entropy(
            probabilities_b, yb
        )
        return loss

    def predict(
        self, distances: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        probabilities = torch.zeros(
            distances.shape[0], self._num_classes, device=self._device
        )

        for i in range(self._num_classes):
            indexes = (target_value.flatten() == i).nonzero().squeeze()

            if indexes.numel() == 1:
                indexes = indexes.unsqueeze(0)
            # print("indexes.shape", indexes.shape, i)
            if indexes.numel() > 0:
                this_sum = torch.sum(distances[:, indexes], dim=1)
                probabilities[:, i] = this_sum
            else:
                probabilities[:, i] = 0

        probabilities = probabilities / torch.linalg.norm(
            probabilities, ord=1, dim=1
        ).view(-1, 1)

        return probabilities

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
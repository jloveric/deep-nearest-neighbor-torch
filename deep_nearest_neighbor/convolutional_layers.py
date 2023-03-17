from deep_nearest_neighbor.layer import CommonMixin, Predictor, Results
import torch
from torch import Tensor
from typing import Tuple
from tqdm import tqdm
import time
from torch.utils.data import DataLoader


class ConvolutionalLayer2d(CommonMixin):
    def __init__(
        self,
        num_classes: int,
        distance_metric=None,
        device: str = "cuda",
        target_accuracy: float = 0.9,
        max_neighbors: int = float("inf"),
        max_count: int = 3,
        kernel_size: int = 3,
    ):
        self._distance_metric = distance_metric
        self._neighbors: torch.Tensor = None
        self._neighbor_value: torch.Tensor = None
        self._device = device
        self._target_accuracy = target_accuracy
        self._num_classes = num_classes
        self._max_neighbors = max_neighbors
        self._max_count = max_count
        self._kernel_size = kernel_size

    def predict(
        self,
        distances: Tensor,
        target_value: Tensor,
        style: Predictor = Predictor.Interp,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param distances: inverse distances between samples and all neighbors
        :param target_value: classification of each neighbor
        :param style: style of prediction (nearest neighbor, all sum)
        :returns: predicted classification for each sample
        """

        probabilities = torch.zeros(distances.shape[0], self._num_classes)

        if style == Predictor.Nearest:
            nearest_neighbor = torch.argmax(distances, dim=1)
            predicted_classification = target_value[nearest_neighbor]
            probabilities[:, predicted_classification] = 1.0
        elif style == Predictor.Interp:
            predicted_classification = 0
            predicted_sum = 0

            # TODO: Figure out how to do this without the for loop
            for i in range(self._num_classes):
                indexes = (target_value.flatten() == i).nonzero().squeeze()
                # print("target_value", target_value)

                # TODO: When numel is one 1 I lose a dimension and stuff breaks. Figure out
                # a better approach here.

                # None of the indexes match
                # if indexes.numel() == 0:
                #    continue
                # print("indexes.shape", indexes.shape, i, indexes)

                if indexes.numel() == 1:
                    indexes = indexes.unsqueeze(0)
                # print("indexes.shape", indexes.shape, i)
                if indexes.numel() > 0:
                    this_sum = torch.sum(distances[:, indexes], dim=1)

                    predicted_sum = torch.where(
                        predicted_sum > this_sum, predicted_sum, this_sum
                    )

                    predicted_classification = torch.where(
                        predicted_sum > this_sum, predicted_classification, i
                    )
                    probabilities[:, i] = this_sum
                else:
                    probabilities[:, i] = 0

                probabilities = probabilities / torch.linalg.norm(
                    probabilities, ord=1, dim=1
                ).view(-1, 1)

        return predicted_classification, probabilities

    def incorrect_predictions(
        self,
        distances: Tensor,
        target_value: Tensor,
        sample_classification: Tensor,
    ) -> Tensor:
        """
        Compute the sample classifications that did not match the predicted
        classifications.  Return the indices that were computed incorrectly
        :param distance: Distances from each of the neighbors
        :param target_value: The classification for each of those neighbors
        :param sample_classification: The classification for each of the samples
        """
        # nearest_neighbor = torch.argmin(distances, dim=1)
        # predicted_classification = target_value[nearest_neighbor]
        predicted_classification, probabilities = self.predict(
            distances=distances, target_value=target_value
        )

        all = predicted_classification == sample_classification
        wrong_indices = torch.logical_not(all).nonzero().squeeze()
        if wrong_indices.numel() == 1:
            wrong_indices = wrong_indices.unsqueeze(dim=0)

        return wrong_indices

    def train_loop(
        self,
        samples,
        sample_class,
        target_accuracy: float = 0.9,
    ):
        result = 0
        count = 0
        while (
            (result < target_accuracy)
            and (len(self._neighbors) <= self._max_neighbors)
            and (count < self._max_count)
        ):
            # print("result", result, "count", count)
            distances = self._distance_metric(keys=self._neighbors, values=samples)

            wrong_indices = self.incorrect_predictions(
                distances=distances,
                target_value=self._neighbor_value,
                sample_classification=sample_class,
            )

            if wrong_indices.numel() > 0:
                new_keys = samples[wrong_indices]
                new_class = sample_class[wrong_indices]

                self.extend_neighbors(new_keys, new_class)

            final_distances = self._distance_metric(
                keys=self._neighbors, values=samples
            )
            final_predictions, probabilities = self.predict(
                distances=final_distances, target_value=self._neighbor_value
            )
            how_good = final_predictions == sample_class

            result = torch.sum(how_good) / how_good.shape[0]
            count += 1

        return self._neighbors, self._neighbor_value

    def test_wrong(
        self,
        neighbors: Tensor,
        neighbor_value: Tensor,
        samples: Tensor,
        sample_class: Tensor,
    ) -> int:
        """
        Return the number of elements that are wrong in samples
        """

        distances = self._distance_metric(keys=neighbors, values=samples)

        wrong_indices = self.incorrect_predictions(
            distances=distances,
            target_value=neighbor_value,
            sample_classification=sample_class,
        )

        return wrong_indices.numel()

    def test_loop(self, dataloader: DataLoader) -> Results:
        wrong = 0
        datapoints = 0

        t_start = time.perf_counter()
        for data in tqdm(iter(dataloader)):
            x, y = data
            x = x.to(self._device)
            y = y.flatten().to(self._device)

            datapoints += len(x)
            wrong += self.test_wrong(
                neighbors=self._neighbors,
                neighbor_value=self._neighbor_value,
                samples=x,
                sample_class=y,
            )

        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")

        return Results(
            error=wrong / datapoints,
            accuracy=(datapoints - wrong) / datapoints,
            incorrect=wrong,
            total=datapoints,
        )

    def epoch_loop(self, dataloader: DataLoader) -> Tensor:
        data_iter = iter(dataloader)

        if self._neighbors == None:
            # Set the neighbors to the first batch
            x, y = next(data_iter)
            self._neighbors = x
            self._neighbor_value = y.flatten()
            # print(
            #    "creating first batch",
            #    self._neighbors.shape,
            #    self._neighbor_value.shape,
            # )

        self._neighbors = self._neighbors.to(self._device)
        self._neighbor_value = self._neighbor_value.to(self._device)
        t_start = time.perf_counter()
        for count, data in enumerate(pbar := tqdm(data_iter)):
            pbar.set_postfix({"neighbors": len(self._neighbors)})
            x, y = data

            # convert [N, C, X, Y] to [N, C*kernel_size*kernel_size, (x-kernel_size+1)*(y-kernel_size+1)]
            x = torch.nn.functional.unfold(
                x.to(self._device), kernel_size=self._kernel_size
            )
            y = y.to(self._device).flatten()

            self._neighbors, self._neighbor_value = self.train_loop(
                samples=x,
                sample_class=y,
                target_accuracy=self._target_accuracy,
            )
        t_total = time.perf_counter() - t_start
        print(f"Epoch_loop time {t_total}")
        print(f"Network neighbors {len(self._neighbors)}")

        return self._neighbors, self._neighbor_value

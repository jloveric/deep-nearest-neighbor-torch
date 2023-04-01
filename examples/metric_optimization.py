from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from deep_nearest_neighbor.layer import Layer
from deep_nearest_neighbor.metrics import choose_metric
from deep_nearest_neighbor.networks import Network
from deep_nearest_neighbor.image_data_utils import choose_dataset
import os

def run_single_layer(cfg: DictConfig):
    training_data, test_data, num_classes = choose_dataset(cfg=cfg)
    distance_metric = choose_metric(cfg)

    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=cfg.pin_memory
    )

    # print(f"hydra.run.dir", hydra.run.dir)
    print(f"Current working directory : {os.getcwd()}")
    # print(f"Orig working directory    : {get_original_cwd()}")
    layer = Layer(
        num_classes=num_classes,
        # distance_metric=InfluenceCone(epsilon=1e-6, exponent=2, factor=4),
        distance_metric=distance_metric,
        device=cfg.device,
        target_accuracy=cfg.target_accuracy,
        max_neighbors=cfg.max_neighbors,
        max_count=cfg.max_count,
    )

    layer.epoch_loop(
        dataloader=train_dataloader,
    )
    num_neighbors = len(layer.neighbors)

    layer.save()

    train_result = layer.test_loop(
        dataloader=train_dataloader,
    )
    print("train_result", train_result)

    test_result = layer.test_loop(
        dataloader=test_dataloader,
    )

    print("test_result", test_result)
    print("neighbors in model", num_neighbors)

@hydra.main(
    config_path="../config",
    config_name="metric_optimization",
    version_base="1.3",
)
def run(cfg: DictConfig):
    run_single_layer(cfg=cfg)
    


if __name__ == "__main__":
    run()
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from deep_nearest_neighbor.learned_transform import DeepNearestNeighborLayer
from deep_nearest_neighbor.metrics import choose_metric
from deep_nearest_neighbor.networks import Network
from deep_nearest_neighbor.image_data_utils import choose_dataset
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch.nn as nn
from high_order_layers_torch.networks import HighOrderMLP, LowOrderMLP
import torch


def run_single_layer(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(filename="{epoch:03d}", monitor="loss")

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

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=[checkpoint_callback],
    )

    if cfg.mlp.style == "low-order":
        transform_network = LowOrderMLP(
            in_width=cfg.mlp.input.width,
            out_width=cfg.mlp.output.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_width=cfg.mlp.hidden.width,
            non_linearity=torch.nn.ReLU(),
        )
    elif cfg.mlp.style == "high-order":
        transform_network = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            in_width=cfg.mlp.input.width,
            out_width=cfg.mlp.output.width,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            in_segments=cfg.mlp.input.segments,
            out_segments=cfg.mlp.output.segments,
            hidden_segments=cfg.mlp.hidden.segments,
        )

    model = DeepNearestNeighborLayer(
        transform_network=transform_network,
        in_features=cfg.in_features,
        out_features=cfg.out_features,
        num_classes=num_classes,
        device=cfg.device,
    )
    trainer.fit(model, train_dataloaders=[train_dataloader])
    print("testing")
    # trainer.test(model)

    print("finished testing")
    print("best check_point", trainer.checkpoint_callback.best_model_path)


@hydra.main(
    config_path="../config",
    config_name="transform_optimization",
    version_base="1.3",
)
def run(cfg: DictConfig):
    run_single_layer(cfg=cfg)


if __name__ == "__main__":
    run()

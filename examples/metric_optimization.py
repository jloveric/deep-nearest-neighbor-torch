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


def run_single_layer(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(filename="{epoch:03d}", monitor="train_loss")

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

    model = DeepNearestNeighborLayer(
        in_features=784, out_features=32, num_classes=num_classes, device=cfg.device
    )
    trainer.fit(model)
    print("testing")
    # trainer.test(model)

    print("finished testing")
    print("best check_point", trainer.checkpoint_callback.best_model_path)


@hydra.main(
    config_path="../config",
    config_name="metric_optimization",
    version_base="1.3",
)
def run(cfg: DictConfig):
    run_single_layer(cfg=cfg)


if __name__ == "__main__":
    run()

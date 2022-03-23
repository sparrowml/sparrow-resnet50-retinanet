from operator import itemgetter
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sparrow_tracky import MODA

from .config import Config
from .dataset import RetinaNetDataset, Holdout
from .model import RetinaNet
from .utils import batch_moda


class RetinaNetTrainer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = RetinaNet(n_classes=Config.n_classes)
        self.learning_rate = Config.learning_rate
        self.train_dataset = RetinaNetDataset(Holdout.train)
        self.dev_dataset = RetinaNetDataset(Holdout.dev)
        self.test_dataset = RetinaNetDataset(Holdout.test)
        self.batch_size = Config.batch_size
        self.n_workers = Config.n_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            collate_fn=lambda x: x,
        )

    def training_step(self, batch: list[dict[str, torch.Tensor]], _) -> torch.Tensor:
        images = list(map(itemgetter("image"), batch))
        loss = self.model(images, batch)
        cls_loss, box_loss = itemgetter("classification", "bbox_regression")(loss)
        total_loss = cls_loss + box_loss
        logger_kwargs = dict(on_step=True, prog_bar=True, logger=True)
        self.log("box_loss", box_loss, **logger_kwargs)
        self.log("class_loss", cls_loss, **logger_kwargs)
        self.log("total_loss", total_loss, **logger_kwargs)
        return total_loss

    def validation_step(self, batch: list[dict[str, torch.Tensor]], _) -> MODA:
        images = list(map(itemgetter("image"), batch))
        results = self.model(images)
        moda = batch_moda(results, batch)
        self.log("dev_moda", moda.value, prog_bar=True, on_epoch=True)
        return moda

    def validation_epoch_end(self, outputs: list[MODA]) -> float:
        moda = MODA()
        for moda_batch in outputs:
            moda += moda_batch
        return moda.value

    def test_step(self, batch: list[dict[str, torch.Tensor]], _) -> MODA:
        images = list(map(itemgetter("image"), batch))
        results = self.model(images)
        moda = batch_moda(results, batch, self.n_classes)
        self.log("test_moda", moda.value, prog_bar=True, on_epoch=True)
        return moda

    def test_epoch_end(self, outputs: list[MODA]) -> float:
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)


def train_model(
    pretrained_model_path: Union[Path, str] = str(Config.pretrained_model_path),
    skip_classes: bool = False,
    max_epochs: int = Config.max_epochs,
    max_steps: Optional[int] = -1,  # For testing
) -> None:
    early_stop = EarlyStopping("dev_moda", mode="max")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        gpus=Config.gpus,
        callbacks=[early_stop],
    )
    lightning = RetinaNetTrainer()
    if pretrained_model_path is not None:
        lightning.model.load(pretrained_model_path, skip_classes=skip_classes)
    trainer.fit(lightning)


def save_checkpoint(checkpoint_path: str) -> None:
    lightning = RetinaNetTrainer()
    lightning = lightning.load_from_checkpoint(checkpoint_path)
    torch.save(lightning.model.state_dict(), Config.trained_model_path)

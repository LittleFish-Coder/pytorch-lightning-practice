# imports
import torch
import pytorch_lightning as pl
from model import NN
from dataset import MNISTDataModule
import config


if __name__ == "__main__":
    # initialize
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )
    # data module
    data_module = MNISTDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    # trainer
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
    )
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)

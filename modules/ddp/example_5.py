"""
Unlike your manual script, you don't use torchrun from the command line. You simply run `python example_4.py`
Lightning will see the devices=4 and strategy="ddp" flags and internally trigger the multi-process launch

Using lighting greatly simplifies the code, by removing boilerplates, such as:
- logging (it logs only on rank 0 and gives handlers to alternative loggers. by default it uses TensorBoard)
- checkpointing (it saves only on rank 0 and gives handlers to alternative checkpointing strategies - early stopping, best k models, etc)
- data loading (it automatically shards the data across processes)
- distributed data parallel strategies is automatically handled (GLoo for CPUs, nccl for GPUs).
- mixed precision training (you just set precision=16,"bf16","16-mixed","16-true",... and it handles the rest)
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import lightning as L

class SampleDataset(Dataset):
    def __init__(self, size):
        super(SampleDataset, self).__init__()
        self.data = torch.randn(size, 10)
        self.target = self.data.sum(dim=1, keepdim=True) + torch.randn(size, 1) * 0.1 # simulated target with noise
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.target[idx]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

class LitDataModule(L.LightningDataModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def setup(self, stage):
        # Split your data here
        if stage == "fit" or stage is None:
            self.train_set = SampleDataset(size=int(self.size * 0.8))
            self.val_set = SampleDataset(size=int(self.size * 0.2))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, num_workers=4, persistent_workers=True)

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        # Built-in modules (Lightning handles 'Rank 0' logic for you)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.mse_loss(y_hat, y)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True) # WandB will pick this up automatically
            return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

if __name__ == "__main__":
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    # lightning provices various types of logging integrations (tensorboard(default), wandb, etc.)
    wandb_logger = WandbLogger(project="learning", name="abdelmanan-abdelrahman03-komrades")

    # Saves the "best" model based on validation loss, not just the last one.
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    # Stops training automatically if the model stops improving (saves time/money).
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)

    model = LitModel()

    datamodule = LitDataModule(size=1000)

    # Note: Use strategy="auto" or strategy="ddp" for 2+ devices. 
    # For 1 device, "auto" is safest.
    trainer = L.Trainer(
        precision=32,
        accelerator="auto", 
        strategy="ddp",      # Uses 'gloo' for CPU and 'nccl' for GPU automatically
        devices=8,           # Number of CPU processes (like nproc-per-node)
        max_epochs=10,  # The model will see the dataset 10 times
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)

"""
Unlike your manual script, you don't use torchrun from the command line. You simply run `python example_4.py`
Lightning will see the devices=4 and strategy="ddp" flags and internally trigger the multi-process launch
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import lightning as L

class SampleDataset(Dataset):
    def __init__(self, size):
        super(SampleDataset, self).__init__()
        self.data = torch.randn(size, 10)
        self.target = torch.randn(size, 1)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.target[idx]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        # Built-in modules (Lightning handles 'Rank 0' logic for you)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

if __name__ == "__main__":
    # lightning provices various types of logging integrations (tensorboard(default), wandb, etc.)
    from lightning.pytorch.loggers import WandbLogger
    wandb_logger = WandbLogger(project="learning", name="abdelmanan-abdelrahman03-komrades")

    model = LitModel()

    dataset = SampleDataset(1000)
    # NOTICE: No DistributedSampler! Lightning adds it automatically.
    train_loader = DataLoader(dataset, batch_size=32, num_workers=4)

    # This is where you set your DDP strategy
    trainer = L.Trainer(
        precision=32,  # Could be "16-mixed", "bf16-mixed", etc.
        accelerator="gpu",
        devices=1,           # Number of CPU processes (like nproc-per-node)
        strategy="ddp",      # Uses 'gloo' for CPU and 'nccl' for GPU automatically
        max_epochs=10,  # The model will see the dataset 10 times
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(model, train_loader)

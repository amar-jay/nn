import pytorch_lightning as L
import torch
from dataset import get_dataloader
from model import CNNClassifierModel, CNNClassifierConfig
from optimizers import adam
import torch.nn.functional as F
from torchmetrics.functional import accuracy
# print(pytorch_lightning.cuda)
from pytorch_lightning.callbacks import ModelCheckpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
print("\n\n")
# model = CNNClassifierModel()
# model.to(device)
# print("num of parameters: ", model.num_parameters())
# res = model(torch.randn((64, 1, 28,28)))
# print("result", res.shape, torch.randn((64, 1, 28,28)).dtype)
# train_loader, _ = get_dataloader()
# for inputs, labels in train_loader:
#     inputs, labels = inputs.to(device), labels.to(device)
#     logit = model(inputs)
#     print(logit.shape)

#     break

class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        config = CNNClassifierConfig()
        self.model = CNNClassifierModel(config)
    def forward(self, x):
        return self.model(x) # forward step -> meant for inference only



    def training_step(self, batch, batch_idx):
        x, label = batch
        logits = self(x)
        loss = F.cross_entropy(logits, label)
        acc = accuracy(logits, label, task="multiclass", num_classes=10)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        logits = self(x)
        loss = F.cross_entropy(logits, label)
        acc = accuracy(logits, label, task="multiclass", num_classes=10)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, label = batch
        logits = self(x)
        loss = F.cross_entropy(logits, label)
        acc = accuracy(logits, label, task="multiclass", num_classes=10)
        self.log('test_loss', loss)
        self.log('test_acc', acc)



    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        optimizer = adam.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',   # Metric to monitor
    filename='checkpoint',  # Filename for the best checkpoint
    save_top_k=1,          # Save only the best model
    mode='min'             # Save the model with minimum validation loss
)

if __name__ == "__main__":
    _type = input("train / inference, yes if training (y/N): ")
    if _type == "y":
        model = CNNClassifierModel()
        lit_model = LitClassifier(model)
        trainer = L.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
        train, val, test = get_dataloader()
        trainer.fit(lit_model, train_dataloaders=train, val_dataloaders=val)
        
        trainer.test(model, test)

    else:
        version = int(input("(NOTE: saves only the best version)\nversion no: "))
        model = LitClassifier.load_from_checkpoint(f"lightning_logs/version_{version}/checkpoints/checkpoint.ckpt")
        model.eval()
        _, _, mnist_data = get_dataloader()

        # Perform inference with the loaded model
        total_predictions = 0
        for inputs, label in mnist_data:
            inputs = inputs.to(device)
            inputs, label = inputs[0], label[0]  # Add batch dimension
            with torch.no_grad():
                prediction = model(inputs)
                predicted_class = torch.argmax(prediction, dim=1)
                total_predictions += 1 if predicted_class.item() == label else 0

            print(f"Predicted class: {predicted_class.item()} | actual: {label}")

        print(f"Total accuracy of predictions: {total_predictions * 100 / len(mnist_data):.3f}%")

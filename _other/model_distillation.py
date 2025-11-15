import os
import numpy as np
import pandas as pd
from torch import nn
import torch


dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic-Dataset.csv')

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please ensure the Titanic-Dataset.csv file is present in the data directory.")


# get from csv file
dataset = pd.read_csv(dataset_path)
"""
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""
# survived is the target column
target_columns = ['Survived', 'Sex']
target = dataset[target_columns]
dataset = dataset.drop(columns=[*target_columns, 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'])

# change all to float
dataset = dataset[dataset.columns].astype(float)

# remoce nan values
dataset = dataset.dropna()
target = target.loc[dataset.index]

# unique target values assign string to int
target["Sex"] = target["Sex"] == "male"
target = target.astype(float)
dataset = dataset.loc[target.index]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise EnvironmentError("GPU not available. Please run this code on a machine with a CUDA-compatible GPU.")

class Model(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=10):
        super(Model, self).__init__()
        size = input_size
        self.layers = []
        for _ in range(num_layers):
            layer = nn.Linear(size, 2*size, device=device)
            nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
            self.layers.append(layer)
            #self.layers.append(nn.BatchNorm1d(2*size, device=device))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.Dropout(0.1))
            size *= 2

        # classification head
        self.fc_head = nn.Linear(size, num_classes, device=device)

    def forward(self, x):
        x = x.to(device)
        for layer in self.layers:
            x = layer(x)
        return self.fc_head(x)

print("Dataset shape after cleaning:", dataset.shape, "Target shape after cleaning:", target.shape)
print(dataset.info())
print(target.info())
print("-"*80)
dataset = dataset.reset_index(drop=True)
target = target.reset_index(drop=True)

#dataset_t = torch.tensor(, dtype=torch.float32, device=device)
dset = torch.utils.data.TensorDataset(
    torch.tensor(dataset.values, dtype=torch.float32, device=device),
    torch.tensor(target["Survived"].values, dtype=torch.long, device=device)   # <-- MUST be long for CE
)

dataloader = torch.utils.data.DataLoader(dataset=dset, batch_size=32, shuffle=True)

teacher_model = Model(input_size=5, num_classes=2, num_layers=4).to(device)
student_model = Model(input_size=5, num_classes=2, num_layers=1).to(device)

# Test forward pass
for x, y in dataloader:
    print(x.shape, y.shape)
    print(teacher_model(x).shape)  # (batch, 2)
    print(student_model(x).shape)
    break


if __name__ == "__main__":
    from tqdm import trange
    from matplotlib import pyplot as plt
    print("Model distillation teacher model training ...")

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean') # For distillation later

    # TRAIN TEACHER
    losses = []
    running_loss = 0
    for epoch in trange(50):
        teacher_model.train()

        for x, y in dataloader:
            logits = teacher_model(x)    # shape (batch, 2)

            # y MUST be shape (batch,) with int class labels {0,1}
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            running_loss += loss.item()

    print(f"Teacher Loss: {running_loss/len(dataloader):.4f}")
    plt.plot(losses)
    plt.show()

    # EVALUATE TEACHER
    # using the training data itself for simplicity
    teacher_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            logits = teacher_model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Teacher Model Accuracy: {correct/total:.4f}")

    torch.save(teacher_model.state_dict(), "teacher_model.pth")


    # -----------------------
    # Student Distillation
    # -----------------------
    optimizer_student = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    temperature = 5.0
    alpha = 0.7

    student_losses = []

    student_model.train()
    total_loss = 0
    for epoch in trange(5):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer_student.zero_grad()
            student_logits = student_model(x)
            with torch.no_grad():
                teacher_logits = teacher_model(x)

            # Soft targets
            soft_teacher = torch.softmax(teacher_logits / temperature, dim=1)
            soft_student = torch.log_softmax(student_logits / temperature, dim=1)
            loss_kd = criterion_kd(soft_student, soft_teacher) * (temperature ** 2)

            # Hard labels
            loss_ce = criterion(student_logits, y)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce
            loss.backward()
            optimizer_student.step()
            total_loss += loss.item()
        student_losses.append(total_loss / len(dset))
        total_loss = 0
    print(f"Student Loss={total_loss / len(dset):.4f}")

    # EVALUATE STUDENT
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = student_model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Student Accuracy: {correct/total:.4f}")
    torch.save(student_model.state_dict(), "student_model.pth")



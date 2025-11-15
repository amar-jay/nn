from torch import nn
import torch
import torchvision

# -----------------------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dset = torchvision.datasets.CIFAR10("../data/", train=True, download=True, transform=transform)
test_dset = torchvision.datasets.CIFAR10("../data/", train=False, download=True, transform=transform)

print(train_dset.data.shape)  # (50000, 32, 32, 3)
print(train_dset.classes)  # (50000,)
print(test_dset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise EnvironmentError("GPU not available. Please run this code on a machine with a CUDA-compatible GPU.")

class Model(nn.Module):
    def __init__(self, num_classes=10, input_channels=3,
                 num_layers=6, base_channels=32, batch_norm=True, device=device):
        super().__init__()

        layers = []
        in_ch = input_channels
        ch = base_channels

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())

            # downsample every 2 layers
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2,2))

            in_ch = ch
            ch *= 2

        self.feature_extractor = nn.Sequential(*layers)

        # figure out flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 32, 32)
            out = self.feature_extractor(dummy)
            flat_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, num_classes, device=device)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

print("-"*80)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dset, batch_size=4, shuffle=True)

teacher_model = Model(num_classes=10, num_layers=6).to(device)
student_model = Model(num_classes=10, num_layers=2).to(device)

# Test forward pass
for x, y in train_dataloader:
    print(x.shape, y.shape)
    x = x.to(device) 
    y = y.to(device)
    print(teacher_model(x).shape)  # (batch, 2)
    print(student_model(x).shape)
    break

if __name__ == "__main__":
    import os
    from tqdm import trange, tqdm
    from matplotlib import pyplot as plt

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean') # For distillation later

    if not os.path.exists("teacher_model.pth"):
        print("Model distillation teacher model training ...")
        # TRAIN TEACHER
        losses = []
        teacher_model.train()
        for epoch in range(10):
            total_loss = 0
            for x, y in tqdm(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = teacher_model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / len(train_dset))
            print(f"Epoch {epoch+1}: Teacher Loss={total_loss/len(train_dset):.4f}")

        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        # EVALUATE TEACHER
        # using the training data itself for simplicity
        teacher_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                logits = teacher_model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Teacher Model Accuracy: {correct/total:.4f}")

        torch.save(teacher_model.state_dict(), "teacher_model.pth")
    else:
        model_state = torch.load("teacher_model.pth", map_location=device)
        teacher_model.load_state_dict(model_state)

        teacher_model.eval()
        print("Evaluating loaded teacher model ...")
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                logits = teacher_model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Teacher Model Accuracy: {correct/total:.4f}")
        """

    # -----------------------
    # Student Distillation
    # -----------------------
    optimizer_student = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
    temperature = 5.0
    alpha = 0.7

    student_losses = []

    print("Training student model with knowledge distillation ...")
    student_model.train()
    total_loss = 0
    for epoch in range(5):
        for x, y in tqdm(train_dataloader):
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
        student_losses.append(total_loss / len(train_dset))
        total_loss = 0
    print(f"Student Loss={total_loss / len(train_dset):.4f}")

    print("Evaluating student model ...")
    # EVALUATE STUDENT
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            logits = student_model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Student Accuracy (Train): {correct/total:.4f}")
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            logits = student_model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Student Accuracy (Test): {correct/total:.4f}")

    torch.save(student_model.state_dict(), "student_model.pth")



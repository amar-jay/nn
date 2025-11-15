import torch
from dataset import CalculationDataset, collate_arithmetic, ALL_OPS
from models import MLP
import os
import torch.optim as optim
from tqdm import tqdm


# test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("checkpoints", exist_ok=True)

def train_mlp(batch_size=64, num_epochs=3):
    model = MLP(step_size=64)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    dataset = CalculationDataset("data/int_data.csv", filtered_ops=["add", "sub"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_arithmetic)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_arithmetic)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y[:, :1], reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        av_loss = total_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {av_loss:.4f}")
        model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                total_acc += (torch.abs(pred - y[:, :1]) < 1e-1).sum().item()
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {total_acc/len(test_dataloader.dataset):.4f}")
        model.train()

    checkpoint_path = f"checkpoints/{model._get_name()}_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return total_loss / len(train_dataloader.dataset)


@torch.no_grad()
def eval_mlp():
    model = MLP(step_size=64)
    model.load_state_dict(torch.load(f"checkpoints/{model._get_name()}_checkpoint.pth"))
    model = model.to(device)
    dataset = CalculationDataset("data/int_data.csv", filtered_ops=["add", "sub"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=collate_arithmetic)

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y[:, :1], reduction='mean')
            total_loss += loss.item() * x.size(0)
            total_acc += (torch.abs(pred - y[:, :1]) < 1).sum().item()
            total_mse += torch.nn.functional.mse_loss(pred, y[:, :1], reduction='none').mean().item()
    loss, acc, mse = total_loss / len(test_dataloader.dataset), total_acc / len(test_dataloader.dataset), total_mse / len(test_dataloader.dataset)
    print(f"Test Loss after loading checkpoint: {loss}")
    print(f"Test Accuracy after loading checkpoint: {acc}")
    print(f"Test MSE after loading checkpoint: {mse}")
    return

if __name__ == "__main__":
    import os

    # create optimizer AFTER model is created so it references the correct params

    # print(f"Epoch {num_epochs}, Train Loss: {train_loss:.4f}")
    # save model checkpoint
    # train_linear_regression()
    # train_mlp()
    # eval_mlp()
    # eval_linear_regression()

    # tryout an example 2 + 6
    model = MLP()
    print(model)
    # model.load_state_dict(torch.load("checkpoints/MLP_checkpoint.pth"))
    # model = model.to(device)
    # model.eval()
    # a = torch.tensor([[5.0, 11.0, 0.0]], dtype=torch.float32).to(device)
    # with torch.no_grad():
    #     pred = model(a)
    # print(f"Model prediction for {ALL_OPS[int(a[:, 2].item())]}({a[:, 0].item()} , {a[:, 1].item()}): {pred.cpu().numpy()}")
    # for x, y in train_dataloader:
    #     x = x.to(device)
    #     y = y.to(device)
    #     print(f"x: {x}, y: {y}, pred: {pred}, loss: {loss}")
    #     break
    # with torch.no_grad():
    #     pred = model(a)
    # print(f"Model prediction for exp(2): {pred.cpu().numpy()}")
from simple_dataset import Data
from simple_model import LinearRegressionModel
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader





def train(optim, f, **kwargs):
    model = LinearRegressionModel(1,1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = optim(model.parameters(), **kwargs)
    epoch = 0
    # training loop
    print("Training...")
    loss = torch.tensor(0)
    while True:
        for x, y in dataloader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch += 1
        if loss is not None and epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss {loss.item()}')
        if loss.item() < 1e-3:
            break


    # inference
    print("Inference...")
    mean = torch.tensor([0.00])
    accuracy = 0
    for x in torch.linspace(0, 9, 10):
        y = -3 * x + 1 + 0.1
        x, y = torch.tensor([x]), torch.tensor([y])
        y_hat = model(x)
        mean += torch.abs(y_hat - y)
        if torch.abs(y_hat - y) < 0.2:
            accuracy += 1
        print(f'Prediction: {y_hat.item():8.3f} Actual: {y.item():8.3f}')

    print(f'mean difference: {(mean*100/10).item():.3f}%\taccuracy: {accuracy*10}%\tnumber of epochs: {epoch}\toptimizer: {optimizer.__class__.__name__}')
    f.write(f'{optimizer.__class__.__name__:20},{accuracy*10:8.3f},{epoch:6},{mean.item():8.3f}\n')
    print("\n\n")

if __name__ == "__main__":
    from sgd import StochasticGradientDescent
    from sgd_mom import SGDWithMomentum
    from rms import RMSProp
    from adam import Adam

    with open('metrics.log', 'w', newline='') as f:
        for i in range(1):
            dataset = Data()
            train(StochasticGradientDescent, f)# Vanilla Stochastic Gradient Descent Optimizer
            train(SGDWithMomentum,f, momentum=0.9) # Stochastic Gradient Descent with moementum Optimizer
            train(SGDWithMomentum, f, momentum=0.9, nestrov=True) # Nestrov Stochastic Gradient Descent Optimizer
            train(RMSProp,f) # Root Mean Square optimizer
            train(Adam,f) # Adam Optimizer


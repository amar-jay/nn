import math
import torch
from torch import nn

INPUT_SIZE = 3 # a, b, op
OUTPUT_SIZE = 1 # result, valid

# simple MLP
class MLP(nn.Module):
    def __init__(self,  step_size=64, num_layers=3, input_size=INPUT_SIZE,output_size=OUTPUT_SIZE):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, step_size))
        # expand
        halfway = num_layers // 2
        for i in range(halfway):
            self.layers.append(nn.Linear(step_size * (2 ** i), step_size * (2 ** (i + 1))))
        # contract
        for i in range(halfway, 0, -1):
            self.layers.append(nn.Linear(step_size * (2 ** i), step_size * (2 ** (i - 1))))
        self.layers.append(nn.Linear(step_size, output_size))

    def forward(self, x):
        # Apply ReLU on intermediate layers, but do NOT apply ReLU on the final
        # layer so the network can produce negative outputs (required for
        # regression targets that can be negative).
        for i, layer in enumerate(self.layers):
            if i < (len(self.layers) - 1):
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x

# Fourier Neural Network
# siren-style network
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # first layer: U(-1/in, 1/in) scaled by 1/omega?
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                # hidden layers: U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
                bound = torch.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class FNN(nn.Module):
    def __init__(self, step_size=64, num_layers=4,input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        super(FNN, self).__init__()
        self.omega_0 = 30.0
        self.layers = nn.ModuleList()
        first_layer = nn.Linear(input_size, step_size)
        print(first_layer.weight.shape)
        torch.nn.init.uniform_(first_layer.weight)
        # first_layer.weight.uniform_(-1 / first_layer.in_features, 1 / first_layer.in_features)
        self.layers.append(first_layer)

        halfway = (num_layers - 1) // 2
        for i in range(halfway):
            hidden_layer = nn.Linear(step_size * (2 ** i), step_size * (2 ** (i + 1)))
            bound = math.sqrt(6 / hidden_layer.in_features) / self.omega_0
            torch.nn.init.uniform_(hidden_layer.weight, -bound, bound)
            self.layers.append(hidden_layer)
        # contract
        for i in range(halfway, 0, -1):
            self.layers.append(nn.Linear(step_size * (2 ** i), step_size * (2 ** (i - 1))))
        self.layers.append(nn.Linear(step_size, output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = torch.sin(self.omega_0 * layer(x))
        return x
# GANs
# symbolic transformer
# graph neural network
# neural arithmetic logic unit (NALU)
# neural ODE
if __name__ == "__main__":
    model = FNN()
    print(model(torch.randn(1, 3)))
import torch

# Example usage
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat

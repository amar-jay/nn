from typing import Any
import torch
import torch.nn as nn

## My implementation of ConvTranspose2d was wrong. Initially I thought it was the direct implementation of a deconvolution operation. 
## However I was wrong. The GPT implementation seems to be the right one. Bcs it closely aligns to the pytorch implementation but not entirely.

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size)) 
        self.bias = nn.Parameter(torch.randn(out_channels + 1))

    def forward(self, x):
        B, T, H, W = x.shape
        H_out = (H-1)*self.stride
        W_out = (W-1)*self.stride

        if T != self.in_channels:
            raise IndexError()

        output = torch.zeros(B, self.out_channels, H_out, W_out)
        for b in range(B):
            for t in range(self.out_channels):
                for h in range(H-1):
                    for w in range(W-1):
                        h_start, h_end = h, h + self.kernel_size 
                        w_start, w_end = w, w + self.kernel_size 
                        z = x[b, t, h, w].view(-1, 1, 1) * self.kernel[t] + self.bias[t]
                        output[b, :, h_start:h_end, w_start:w_end] += z

        return output

class GPTConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(GPTConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape

        # Calculate output dimensions
        out_height = (in_height - 1) * self.stride[0]  + self.kernel_size[0]
        out_width = (in_width - 1) * self.stride[1] + self.kernel_size[1]

        # Create output tensor
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=x.device)

        # Perform transposed convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(in_height):
                        for j in range(in_width):
                            out_i = i * self.stride[0]
                            out_j = j * self.stride[1]
                            output[b, oc, out_i:out_i+self.kernel_size[0], out_j:out_j+self.kernel_size[1]] += \
                                x[b, ic, i, j] * self.weight[ic, oc]

        output += self.bias.view(1, -1, 1, 1)

        return output

if __name__ == "__main__":
    input_image = torch.randn(1, 16, 64, 64)

    params: dict[str, Any] = dict(
        in_channels=16,
        out_channels=3,
        kernel_size=2,
        stride=2,
    )
    layer = GPTConvTranspose2d(**params)
    weight = nn.Parameter(torch.randn_like(layer.weight))
    bias = nn.Parameter(torch.randn_like(layer.bias))
    layer.weight = weight
    layer.bias = bias
    output2 = layer(input_image)
    print(output2.shape)

    layer = nn.ConvTranspose2d(**params)
    layer.weight = weight
    layer.bias = bias
    output = layer(input_image)
    print(output.shape)


    print(output == output2)
    assert (output - output2 < 1e-5).all().item()

from typing_extensions import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    """
    in convolutional neural network the weight is the kernel of the convolution operation with an optional bias.
    $output = input * kernel + bias$
    kernel is a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
    $$
    output_size =  \frac {input_size + 2 * padding - kernel_size} {stride} + 1
    $$


    ## purpose
    Typically used for feature extraction and downsampling in neural networks, 
    such as in convolutional neural networks (CNNs) and autoencoders.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        print(f"{out_channels=}, {in_channels=}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
    def forward(self, x):
        B, _, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.zeros(B, self.out_channels, H_out, W_out)

        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)


        for b in range(B):
            for t in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                        w_start, w_end = w * self.stride, w * self.stride + self.kernel_size
                        patch = x[b, :, h_start:h_end, w_start:w_end]
                        output[b, t, h, w] = (patch * self.kernel[t]).sum() + self.bias[t]
        return output


if __name__ == "__main__":
    input_image = torch.randn(1, 3, 64, 64)

    params: dict[str, Any] = dict(
        in_channels=3,
        out_channels=16,
        kernel_size=2,
        stride=2,
    )
    layer = Conv2d(**params)
    weight =  nn.Parameter(torch.randn_like(layer.kernel))
    bias =  nn.Parameter(torch.randn_like(layer.bias))

    layer.kernel = weight
    layer.bias = bias 
    output = layer(input_image)
    print(output.shape)

    layer = nn.Conv2d(**params)
    layer.weight = weight
    layer.bias = bias 
    output2 = layer(input_image)
    print(output2.shape)

    assert (output - output2 < 1e-5).all().item()

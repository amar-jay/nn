import torch
from torch import nn 
from torch._prims_common import Tensor
import torch.nn.functional as F


# Group Quantization and Normalization is a way to scale up model for large scale deployment
# It does so by partitioning and parrallelizing the matrix multiplication operation through a process called group quantization
# on what dimension are we to split the input, batch, right??

class BitLinear(nn.Linear):
    """BitLinear is a linear layer with bit-wise operations
    It is a drop-in replacement for the standard linear layer
    that is less computationally expensive but competitively accurate.
    its based on the paper.

        https://arxiv.org/pdf/2310.11453.pdf

    BitLinear : input -> Quantization(absmax) -> BitWeight product -> Dequantization + Absmax quants
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bit: int = 8,
        n_groups: int = 1,
        device = None,
        dtype = None
    ):
        super(BitLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.bit = bit
        self.n_groups = n_groups
        self.ln = nn.LayerNorm(in_features)
        self.eps = 1e-05 # the eps usually used is 1e-05
        #self.alpha = torch.sum(self.weight) / (self.weight.size(0) * self.weight.size(0))
        self.alpha = torch.mean(self.weight)
        self.beta = self.weight.abs().mean()

        # using buffers to store weight. Im still not convinced that buffers is 
        # useful in this case
        # self.register_buffer(
        #     "binarized_weight", self._ste(self.weight - self.alpha)
        # )
        
        self.binarized_weight = self._ste(self.weight - self.alpha)

        self.Qb = 2 ** (bit - 1) 
        
        # to reduce memory overhead
        del self.weight

    def forward(self, input):
        input = self.ln(input)
        B, T, C = input.size()

        x_groups = torch.split(input, input.size(1) // self.n_groups, dim=1)


        # normalize, quantize, transform, dequantize
        # gamma = torch.stack([torch.max(x) for x in x_groups], dim=0) # gammas for each group. This may not be the best way to do it
        gamma = torch.max(x_groups.view(self.n_groups, -1), dim=1).values
        # self.gamma = torch.norm(x, p=torch.inf) # torch.max() will do too I guess??
        
        #x_hat = torch.stack([self._quantize(x_groups[i], gamma[i]) for i in range(self.n_groups)]).view(B, T, C)
        # let broadcasting do the work
        x_hat = self._quantize(x_groups, gamma).view(B, T, C)

        y = F.linear(x_hat, self.binarized_weight, self.bias)
        y = self._dequantize(y, gamma)
        return y

    def _dequantize(self, x, gamma):
        return x * self.beta * gamma / self.Qb

    def _quantize(self, x, gamma):
        """
        Quantizes input group 

        x     -> input group 
        gamma -> gamma of the input group
        """
        v = x * (self.Qb / gamma)
        w = -self.Qb + self.eps
        x = self.Qb - self.eps
        return torch.clip(v, w, x) 

    def _ste(self, x: Tensor):
        """ Straight through estimator (STE) function

        sign(x) -> -1 if x >= 0 else 1

        """
        return (torch.sign(x) - x).detach() + x


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, bit={}, n_groups={}'.format(
            self.in_features,
            self.out_features,
            self.bias,
            self.bit,
            self.n_groups
        )

if __name__ == "__main__":
    x = BitLinear(5,5, bias=False)
    print(x)
    e = x(torch.rand((10, 5,5)))
    print(e)

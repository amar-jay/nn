import torch
from typing import Any


class StochasticGradientDescent(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr) 
        super(StochasticGradientDescent, self).__init__(params, defaults) 
  
    def step(self, closure=None)->Any: 
        if closure is not None: 
            with torch.enable_grad(): 
                closure()

        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue
                p.data.add_(p.grad, alpha=-group['lr'])




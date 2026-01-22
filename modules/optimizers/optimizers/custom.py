from typing import Any
import torch

# https://www.geeksforgeeks.org/custom-optimizers-in-pytorch/
class CustomOptimizer(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr) 
        super().__init__(params, defaults) 
  
    def step(self, closure=None)->Any: 
        if closure is not None: 
            with torch.enable_grad(): 
                closure()

        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue
                #p.data -= group['lr']*p.grad.data
                p.data.add_(-p['lr'], p.grad)
  


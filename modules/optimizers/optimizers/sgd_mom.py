import torch
from typing import Any

class SGDWithMomentum(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3, momentum=1.0, dampening=0, weight_decay=0, nestrov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nestrov=nestrov)
        if momentum == 0:
            raise Exception("momentum must be greater than 0")
        if nestrov:
            self.__class__.__name__ = "SGDWithNestrov"

        super(SGDWithMomentum, self).__init__(params, defaults) 
        self.initialized = []
        self.b_t = dict()
  
    def step(self, closure=None)->Any: 
        loss = None
        if closure is not None: 
            with torch.enable_grad(): 
                loss = closure()

        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue

                # weight decay
                if group['weight_decay'] != 0:
                    p.grad += group['weight_decay'] * p.data

                if group['momentum'] != 0:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        # buf = momentum * buf + (1-dampening) * grad
                        self.state[p]['momentum_buffer'].mul_(group['momentum']).add_(p.grad.data.mul_(1-group['dampening']))
                if group['nestrov']:
                    p.data.add_(self.state[p]['momentum_buffer'], alpha=-group['lr'])

                p.data.add_(self.state[p]['momentum_buffer'], alpha=-group['lr'])
            return loss

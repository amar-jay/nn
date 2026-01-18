"""
command to run:
torchrun --standalone --nnodes=1 --nproc_per_node=4 example_2.py

Explanation:
This demonstrates how to run Distributed Data Parallel (DDP) training.
Gloo backend is much flexible, it allows training on CPU as well as GPU. Also, allows multi-node training on a single GPU.
Unlike Gloo backend, NCCL backend only works with GPUs and requires one GPU per process. this is one gpu per process.
NCCL is built for NVLink communication, it is very fast and efficient, but also more restrictive in terms on the one-GPU-per-process prerequisite.
Gloo however is optimized for CPU/Ethernet communication. It is more flexible (handles the one-GPU-per-process error gracefuly) but generally slower than NCCL.

Gloo runs the All-reduce synchronization algorithm via the RAM, whilst NCCL does so via the VRAM of the GPUs itself.


The all-reduce process:
1. Forward Pass: Each rank takes a different batch of data and calculates its own output.

2. Backward Pass: Each rank calculates its own local gradients based on its specific batch.

3. The All-Reduce (Magic Step): Before the optimizer moves the weights, all processes communicate. They sum up (or average) their gradients.
    - Rank 0 sends its gradients to others.
    - Rank 1 sends its gradients to others.
    - By the end of this communication, every rank has the exact same averaged gradient.

4. Optimizer Step: Because every rank now has the same gradient, they all update their local weights in the exact same way. They stay perfectly synchronized.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# 1. Simple Dataset to demonstrate data splitting
class MyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.target = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def setup():
    # 'gloo' is the backend for CPU-based communication
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    
    # Get environment variables set by torchrun
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Process started: Rank {rank} of {world_size}")

    # 2. Create model and move to CPU (since we are using gloo)
    model = nn.Linear(10, 1)
    # Wrap model in DDP. For CPU, device_ids remains None.
    model = DDP(model)

    # 3. Setup Distributed Sampler
    # This is CRITICAL. It ensures Rank 0 gets data [0,2,4...] and Rank 1 gets [1,3,5...]
    dataset = MyDataset(size=100)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 4. Training Loop
    for epoch in range(2):
        # This is necessary to make shuffling work across epochs
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # During optimizer.step(), DDP automatically averages the 
            # gradients across all processes!
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="gloo" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each process gets its own device
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        #torch.cuda.set_device(local_rank)
        #device = torch.device(f"cuda:{local_rank}")

        # force all process to use GPU 0
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:0")

    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}/{world_size}] running on {device}")

    # Tiny model
    model = torch.nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Fake data
    x = torch.randn(32, 10).to(device)
    y = torch.randn(32, 1).to(device)

    for step in range(3):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step {step}, loss {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

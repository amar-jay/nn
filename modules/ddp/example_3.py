"""
How to Run This

Unlike your manual script, you don't use torchrun from the command line. You simply run `python example_3.py`
Lightning will see the devices=4 and strategy="ddp" flags and internally trigger the multi-process launch

"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

def setup(backend="gloo"):
    # we are using gloo backend for learning, but in practive we might use nccl
    dist.init_process_group(backend=backend)

def get_device(is_cuda_available=False):
    # Each process gets its own device
    if is_cuda_available:
        # when using NCCL, LOCAL_RANK env variable is set by torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # force all process to use GPU 0, This only works for Gloo backend not NCCL
        # torch.cuda.set_device(0)
        # device = torch.device(f"cuda:0")
        # rank = dist.get_rank() % torch.cuda.device_count() # local rank
    else:
        device = torch.device("cpu")
        rank = dist.get_rank() # in CPU mode, use global rank
    return device, rank

def cleanup():
    dist.destroy_process_group()

class SampleDataset(Dataset):
    def __init__(self, size):
        super(SampleDataset, self).__init__()
        self.data = torch.randn(size, 10)
        self.target = torch.randn(size, 1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def main(backend):
    setup(backend)
    device, rank = get_device(is_cuda_available=(backend=="nccl"))

    world_size = dist.get_world_size()
    print(f"Global Rank: {rank}, World Size: {world_size}")
    
    # Create model, dataset, dataloader
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    
    dataset = SampleDataset(1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, 
        rank=rank,
        shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        loss = None
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = ((output - target) ** 2).mean()
            loss.backward() # All Hidden All-Reduce happens here!
            optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}: Training loss is {loss.item() if loss is not None else 'N/A'}")
            print(f"Average loss across all ranks is synchronized automatically as {loss.item() / len(dataloader) if loss is not None else 0.0:.3f}")
            checkpoint = {
                'model_state_dict': model.module.state_dict(), # Note the .module!
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, "model_checkpoint.pt")
            print("Checkpoint saved by Rank 0.")

    # 3. SYNCHRONIZATION: The Barrier
    # If Rank 1-3 need that file immediately after it's saved, 
    # you must wait for Rank 0 to finish writing it.
    dist.barrier() 
    print("training complete on all ranks.")
    cleanup()
if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--backend", type=str, default="gloo", required=False)
    args = arg_parser.parse_args()
    print("Using backend:", args.backend)
    main(backend=args.backend)

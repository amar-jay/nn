Hmm! With the increasing complexity, I think I will stick to Pytorch Lightning. It meets all my needs and does all the job I currently need as well as the ease of model checkpoint, Tensorboard and others. And btw, I'm an avid user of Lightning AI

### Resources I read on Distributed training

- (Pytorch DDP)[https://pytorch.org/tutorials/intermediate/ddp_tutorial.html]
- (Pytorch lightning docs)[]
- (large-scale transformer training - TP)[https://pytorch.org/tutorials/intermediate/TP_tutorial.html]
- (FairScale Docs)[https://fairscale.readthedocs.io/en/latest/]
- (On why Pytorch lightning is cool-even in 2020)[https://medium.com/pytorch/pytorch-lightning-1-1-model-parallelism-training-and-more-logging-options-7d1e47db7b0b]

---

### on model parallelism

Basically, there are three levels of parallelism in order of increasing size and architectural complexity:

- Distributed Data Parallel - if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
- Fully-sharded Distributed Data Parallel - when your model cannot fit on one GPU.
- Tensor Parallel - when the above isn't enough, read more on [fairscale docs](https://fairscale.readthedocs.io/en/latest/getting_started.html)

- Pipeline Parallel

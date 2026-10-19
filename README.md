# Neural Network Learning

## Setup

Use Python 3.11 or newer with [uv](https://docs.astral.sh/uv/).
Run these commands from the repository root:

```bash
git submodule update --init --recursive
uv sync --locked
```

The root `pyproject.toml` and `uv.lock` manage one shared environment for this
repository's scripts, including the toy RL Policy Lab experiments. The lockfile
pins dependency versions; `.venv/` and caches are ignored by Git.

```bash
# Run a Python example
uv run python modules/layers/convolution/conv2d.py

# Optional notebook tools
uv sync --locked --group notebooks
uv run --group notebooks jupyter lab

# Optional Hugging Face RL training dependencies
uv sync --locked --group llm-rl
```

Submodules retain their upstream dependency definitions; the root environment
does not combine every external model project's requirements. The legacy
`_other/torrent_to_GDrive.ipynb` helper requires Google Colab and its own setup.

## Topics

```
- [x] MLP
- [x] Token Embedding

- [x] Batch Normalization
- [x] Layer Normalization

- [x] Logistic Regression - Theory
- [x] Gradient Descent/Autograd - Theory

- [x] Byte Pair Encoding

- [x] Rotary Positional Encoding

- [x] Quanitization
- [x] LoRA Finetuning
- [-] QLoRA
- [x] Direct Preference Optimization (DPO)

- [x] Attention
- [x] Decoder Transformer (GPT)
- [x] Encoder Transformer (BERT - NER/Sentiment Analysis)
- [x] Full Transformer / Cross Attention
- [x] T5

- [x] Convolution / CNN
- [x] ResNet
- [x] YOLO

- [x] RNN
- [x] GRU
- [x] LSTM

- [x] VAE
- [ ] GAN
- [x] Diffusion Model
- [x] Rectified Flow
- [x] Autoencoders

- [x] KV Cache
- [x] Parallelization

- [x] Vision Transformer
- [x] UNet
- [ ] VLA

- [x] Scaling laws - Theory

- [x] Reinforcement Learning
  - [x] Finite Markov Decision Processes / Policy Evaluation
  - [x] Q-Learning, SARSA, Expected SARSA
  - [x] Deep Q-Networks (DQN)
  - [x] Advantage Actor-Critic (A2C)
  - [x] GRPO / RLVR / RLOO (rl_policy_lab)

- [x] Distributed Data Processing
- [x] Profiling
- [x] Flash attention
- [ ] MOE

- [x] karparthy/llama.c
- [x] Flux
- [x] VGGT
```

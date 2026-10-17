"""From-scratch GRPO on the countdown game (outcome supervision).

Loop per iteration:
  1. sample G completions per prompt from the old policy
  2. score with the verifiable reward, normalize advantages *within* each group
  3. clipped-surrogate update + KL penalty against a frozen reference policy
"""

import copy
import random

import torch
import torch.nn.functional as F
from torch import nn

from .countdown import generate_puzzle, reward

CHARS = "0123456789+-*/() :="
BOS, EOS, PAD = 0, 1, 2
stoi = {"<bos>": BOS, "<eos>": EOS, "<pad>": PAD, **{c: i + 3 for i, c in enumerate(CHARS)}}
VOCAB = len(stoi)
itos = {i: c for c, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids if i >= 3)


def prompt_for(numbers: list[int], target: int) -> str:
    return f"{' '.join(map(str, numbers))}:{target}="


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4, ctx: int = 128):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


@torch.no_grad()
def sample(net: TinyGPT, prompt_ids: list[int], n: int, max_new: int = 24) -> list[list[int]]:
    device = next(net.parameters()).device
    seqs = []
    for _ in range(n):
        ids = list(prompt_ids)
        for _ in range(max_new):
            inp = torch.tensor([ids[-64:]], device=device)
            probs = F.softmax(net(inp)[0, -1], dim=-1)
            tok = int(torch.multinomial(probs, 1))
            if tok == EOS:
                break
            ids.append(tok)
        seqs.append(ids)
    return seqs


def random_valid_expr(numbers: list[int], rng: random.Random) -> str:
    a, b = rng.sample(numbers, 2)
    e = f"{a}{rng.choice(['+', '-', '*'])}{b}"
    return f"({e})" if rng.random() < 0.3 else e


def pretrain_format(net: TinyGPT, rng: random.Random, steps: int = 400, batch: int = 32):
    """SFT warm-start: teach expression *format* so GRPO has signal to amplify.

    Pure RL from a random policy almost never emits a parseable expression
    (reward stays 0.000) - the same reason real pipelines do SFT before RL.
    """
    device = next(net.parameters()).device
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    net.train()
    for _ in range(steps):
        seqs, plens = [], []
        for _ in range(batch):
            numbers, target = generate_puzzle(rng, 2)
            pids = encode(prompt_for(numbers, target))
            eids = encode(random_valid_expr(numbers, rng)) + [EOS]
            seqs.append(pids + eids)
            plens.append(len(pids))
        maxlen = max(len(s) for s in seqs)
        idx = torch.full((len(seqs), maxlen), PAD, device=device, dtype=torch.long)
        for i, s in enumerate(seqs):
            idx[i, : len(s)] = torch.tensor(s, device=device)
        logits = net(idx)[:, :-1]
        labels = idx[:, 1:]
        pos = torch.arange(idx.shape[1] - 1, device=device)[None]
        plens_t = torch.tensor(plens, device=device)[:, None]
        lens_t = torch.tensor([len(s) for s in seqs], device=device)[:, None]
        mask = (pos >= plens_t - 1) & (pos < lens_t - 1)
        loss = F.cross_entropy(logits[mask], labels[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"sft done, final loss {float(loss.detach()):.4f}", flush=True)


def completion_logps(net: TinyGPT, seqs: list[list[int]], plens: list[int]):
    """Per-token logprobs of the completion part of each sequence. Returns padded tensor + mask."""
    device = next(net.parameters()).device
    maxlen = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), maxlen), PAD, device=device, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s, device=device)
    logp = F.log_softmax(net(idx)[:, :-1], dim=-1).gather(-1, idx[:, 1:, None]).squeeze(-1)
    pos = torch.arange(idx.shape[1] - 1, device=device)[None]
    plens_t = torch.tensor(plens, device=device)[:, None]
    lens_t = torch.tensor([len(s) for s in seqs], device=device)[:, None]
    mask = (pos >= plens_t - 1) & (pos < lens_t - 1)  # token t predicted from prefix <: completion
    return logp, mask.float()


def grpo_step(net, old_net, ref_net, opt, batch, eps=0.2, beta=0.02, epochs=2):
    seqs, plens, rewards = batch
    with torch.no_grad():
        old_logp, mask = completion_logps(old_net, seqs, plens)
        ref_logp, _ = completion_logps(ref_net, seqs, plens)
    n_tok = mask.sum().clamp_min(1)
    for _ in range(epochs):
        new_logp, _ = completion_logps(net, seqs, plens)
        ratio = torch.exp(new_logp - old_logp)
        adv = rewards[:, None] * mask
        surr = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
        kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1  # k3 estimator
        loss = -(surr * mask).sum() / n_tok + beta * (kl * mask).sum() / n_tok
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def train(iters=300, prompts_per_iter=16, group=12, seed=0, n_numbers=2, sft_steps=400):
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TinyGPT().to(device)
    pretrain_format(net, rng, steps=sft_steps)
    ref_net = copy.deepcopy(net).eval()
    for p in ref_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    for it in range(1, iters + 1):
        old_net = copy.deepcopy(net).eval()
        puzzles = [generate_puzzle(rng, n_numbers) for _ in range(prompts_per_iter)]
        seqs, plens, rws = [], [], []
        for numbers, target in puzzles:
            pids = encode(prompt_for(numbers, target))
            rollout = sample(old_net, pids, group)
            for s in rollout:
                seqs.append(s)
                plens.append(len(pids))
                rws.append(reward(decode(s[len(pids) :]), numbers, target))
        # group-normalized advantages (the heart of GRPO: no value network)
        rws_t = torch.tensor(rws, device=device)
        adv = torch.zeros_like(rws_t)
        for i in range(prompts_per_iter):
            g = rws_t[i * group : (i + 1) * group]
            adv[i * group : (i + 1) * group] = (g - g.mean()) / (g.std() + 1e-4)
        loss = grpo_step(net, old_net, ref_net, opt, (seqs, plens, adv))
        if it % 25 == 0 or it == 1:
            exact = sum(1 for r in rws if r >= 1.0) / len(rws)
            print(f"iter {it:3d} loss {loss:+.4f} mean_r {rws_t.mean():.3f} exact {exact:.2f}", flush=True)

    # show one greedy decode per fresh puzzle
    net.eval()
    with torch.no_grad():
        for _ in range(3):
            numbers, target = generate_puzzle(rng, n_numbers)
            ids = encode(prompt_for(numbers, target))
            for _ in range(24):
                inp = torch.tensor([ids[-64:]], device=device)
                nxt = int(net(inp)[0, -1].argmax())
                if nxt == EOS:
                    break
                ids.append(nxt)
            print(f"{numbers} -> {target} | {decode(ids[len(encode(prompt_for(numbers, target))) :])}")


if __name__ == "__main__":
    train()

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

from .countdown import generate_puzzle, reward, encode, decode, prompt_for, random_valid_expr, VOCAB, PAD, EOS
from .rl import METRIC_NAMES, select_experiment

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
        return x + self.mlp(self.ln2(x)) # preLN


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
def sample(net: TinyGPT, prompt_ids: list[int], n: int, max_new: int = 48) -> list[list[int]]:
    """Sample n answers together, removing finished rows from the active batch."""
    if n < 0 or max_new < 0:
        raise ValueError("n and max_new must be nonnegative")
    if n == 0:
        return []
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    device = next(net.parameters()).device
    plen = len(prompt_ids)
    if hasattr(net, "pos") and plen + max_new > net.pos.num_embeddings:
        raise ValueError("prompt plus max_new exceeds the model context capacity")
    ids = torch.full((n, plen + max_new), PAD, device=device, dtype=torch.long)
    ids[:, :plen] = torch.tensor(prompt_ids, device=device)
    lengths = torch.full((n,), plen + max_new, device=device, dtype=torch.long)
    active = torch.arange(n, device=device)
    for step in range(max_new):
        end = plen + step
        inp = ids[active, :end]
        probs = F.softmax(net(inp)[:, -1], dim=-1)
        tokens = torch.multinomial(probs, 1).squeeze(-1)
        ids[active, end] = tokens  # Retain EOS for the RL loss.
        finished = tokens == EOS
        lengths[active[finished]] = end + 1
        active = active[~finished]
        if active.numel() == 0:
            break
    rows = ids.cpu().tolist()
    return [row[:length] for row, length in zip(rows, lengths.cpu().tolist())]



def pretrain_format(net: TinyGPT, rng: random.Random, steps: int = 400, batch: int = 64, n_numbers: int = 4):
    """SFT warm-start: teach expression *format* so GRPO has signal to amplify.

    Pure RL from a random policy almost never emits a parseable expression
    (reward stays 0.000) - the same reason real pipelines do SFT before RL.
    Keep all puzzle numbers in the prompt, but uniformly choose an expression
    operand count from 2 through n_numbers to demonstrate optional subsets.
    """
    device = next(net.parameters()).device
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    net.train()
    for step in range(steps):
        seqs, plens = [], []
        for _ in range(batch):
            numbers, target = generate_puzzle(rng, n_numbers)
            pids = encode(prompt_for(numbers, target))
            operand_count = rng.randint(2, len(numbers))
            operands = rng.sample(numbers, operand_count)
            eids = encode(random_valid_expr(operands, rng)) + [EOS]
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
        if step % 50 == 0 or step == steps - 1:
            print(f"sft step {step:3d} loss {float(loss.detach()):.4f}", flush=True)
    print(f"sft done, final loss {float(loss.detach()):.4f}", flush=True)


def train(iters=300, prompts_per_iter=64, group=12, seed=0, n_numbers=4,
          sft_steps=400, sft_batch=64, device="cuda", max_new=48,
          rl_step="grpo"):
    """Train on fresh puzzles each iteration; larger batches cover more puzzles."""
    update_step, reward_fn = select_experiment(rl_step)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training requested, but CUDA is unavailable. Check that "
            "nvidia-smi works and the NVIDIA driver is running. "
            "For explicit CPU training, call train(device='cpu')."
        )
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    net = TinyGPT().to(device)
    print(f"Training on {device}" +
          (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)
    print(f"SFT: {sft_steps * sft_batch:,} examples; RL: "
          f"{prompts_per_iter} puzzles x {group} answers per iteration "
          f"({iters * prompts_per_iter:,} generated puzzles total)", flush=True)
    print(f"Using RL step: {update_step.__name__} | reward: {reward_fn.__name__}")
    pretrain_format(net, rng, steps=sft_steps, batch=sft_batch, n_numbers=n_numbers)
    ref_net = copy.deepcopy(net).eval()
    for p in ref_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    metric_history = [[] for _ in METRIC_NAMES]

    for it in range(1, iters + 1):
        old_net = copy.deepcopy(net).eval()
        puzzles = [generate_puzzle(rng, n_numbers) for _ in range(prompts_per_iter)]
        seqs, plens, rws = [], [], []
        for numbers, target in puzzles:
            pids = encode(prompt_for(numbers, target))
            rollout = sample(old_net, pids, group, max_new=max_new)
            for s in rollout:
                seqs.append(s)
                plens.append(len(pids))
                rws.append(reward_fn(decode(s[len(pids) :]), numbers, target))
        rws_t = torch.tensor(rws, device=device)
        loss, metrics = update_step(net, old_net, ref_net, opt, (seqs, plens, rws_t), group=group, pad_tok=PAD, device=device)
        for history, values in zip(metric_history, metrics):
            history.extend(values)
        if it % 25 == 0 or it == 1:
            exact = sum(1 for r in rws if r >= 1.0) / len(rws)
            successful_groups = (rws_t.reshape(-1, group) >= 1).any(dim=1).float().mean()
            print(f"iter {it:3d} loss {loss:+.4f} mean_r {rws_t.mean():.3f} "
                  f"exact {exact:.2f} groups_with_exact {successful_groups:.2f}", flush=True)
    torch.save(net.state_dict(), "grpo_toy.pt")

    # show one greedy decode per fresh puzzle
    net.eval()
    with torch.no_grad():
        for _ in range(3):
            numbers, target = generate_puzzle(rng, n_numbers)
            ids = encode(prompt_for(numbers, target))
            for _ in range(max_new):
                inp = torch.tensor([ids], device=device)
                nxt = int(net(inp)[0, -1].argmax())
                if nxt == EOS:
                    break
                ids.append(nxt)
            print(f"{numbers} -> {target} | {decode(ids[len(encode(prompt_for(numbers, target))) :])}")
    return metric_history

def infer(count=100, model_path="grpo_toy.pt", seed=12345, n_numbers=4,
          max_new=48, device=None, show=20):
    """Reproducible greedy evaluation; return detailed results and summary.

    A separate puzzle RNG makes checkpoint comparisons use identical puzzles.
    ``show`` limits printed examples, not the number evaluated. ``output``
    optionally saves the full report as JSON.
    """
    import json
    from collections import Counter
    from pathlib import Path
    from .countdown import safe_eval

    if count < 1 or n_numbers < 2 or max_new < 1 or show < 0:
        raise ValueError("count/max_new must be positive, n_numbers >= 2, show >= 0")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = TinyGPT().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    rng = random.Random(seed)
    results = []
    labels = {1.0: "exact", 0.3: "wrong_result", 0.1: "disallowed_numbers", 0.0: "invalid_expression"}
    print(f"Checkpoint: {model_path} | device: {device} | greedy | seed: {seed}")
    with torch.no_grad():
        for index in range(count):
            numbers, target = generate_puzzle(rng, n_numbers)
            prompt_ids = encode(prompt_for(numbers, target))
            if len(prompt_ids) + max_new > net.pos.num_embeddings:
                raise ValueError("prompt plus max_new exceeds the model context capacity")
            ids = prompt_ids.copy()
            stopped = False
            for _ in range(max_new):
                inp = torch.tensor([ids], device=device)
                token = int(net(inp)[0, -1].argmax())
                ids.append(token)
                if token == EOS:
                    stopped = True
                    break
            completion = decode(ids[len(prompt_ids):])
            score = reward(completion, numbers, target)
            value = safe_eval(completion)
            result = dict(numbers=numbers, target=target, expression=completion,
                          value=value, reward=score, status=labels[score],
                          generated_tokens=len(ids) - len(prompt_ids),
                          stopped_on_eos=stopped)
            results.append(result)
            if index < show:
                ending = "EOS" if stopped else "LENGTH LIMIT"
                print(f"{index + 1:4d} | {numbers} -> {target} | {completion!r} "
                      f"= {value} | {labels[score]} | {ending}")
    counts = Counter(row["status"] for row in results)
    summary = dict(
        count=count, exact_rate=counts["exact"] / count,
        mean_reward=sum(row["reward"] for row in results) / count,
        status_counts={label: counts[label] for label in labels.values()},
        eos_rate=sum(row["stopped_on_eos"] for row in results) / count,
        mean_generated_tokens=sum(row["generated_tokens"] for row in results) / count,
    )
    report = dict(checkpoint=str(model_path), seed=seed, n_numbers=n_numbers,
                  max_new=max_new, decoding="greedy", summary=summary, results=results)
    print(f"\nExact: {summary['exact_rate']:.1%} ({counts['exact']}/{count}) | "
          f"mean reward: {summary['mean_reward']:.3f}")
    print("Outcomes:", summary["status_counts"])
    print(f"Stopped on EOS: {summary['eos_rate']:.1%} | "
          f"mean generated tokens (including EOS): {summary['mean_generated_tokens']:.2f}")
    return report

if __name__ == "__main__":
    train()

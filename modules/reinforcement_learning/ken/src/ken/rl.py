import torch
import torch.nn.functional as F
# from .grpo_toy import PAD

def _completion_logps(net: torch.nn.Module, seqs: list[list[int]], plens: list[int], pad_tok: int=2):
    """Per-token logprobs of the completion part of each sequence. Returns padded tensor + mask."""
    device = next(net.parameters()).device
    maxlen = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), maxlen), pad_tok, device=device, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s, device=device)
    logp = F.log_softmax(net(idx)[:, :-1], dim=-1).gather(-1, idx[:, 1:, None]).squeeze(-1)
    pos = torch.arange(idx.shape[1] - 1, device=device)[None]
    plens_t = torch.tensor(plens, device=device)[:, None]
    lens_t = torch.tensor([len(s) for s in seqs], device=device)[:, None]
    mask = (pos >= plens_t - 1) & (pos < lens_t - 1)  # token t predicted from prefix <: completion
    return logp, mask.float()



# --------------- TRYOUTS ---------------

def grpo_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2, eps=0.2, beta=0.02, epochs=2, device="cpu"):
    seqs, plens, rwd_t = batch
    # group-normalized advantages (the heart of GRPO: no value network)
    adv = torch.zeros_like(rwd_t, device=device)
    for i in range(len(rwd_t) // group):
        group_rewards = rwd_t[i * group : (i + 1) * group]
        adv[i * group : (i + 1) * group] = (
            group_rewards - group_rewards.mean()
        ) / (group_rewards.std() + 1e-4)
    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)
    n_tok = mask.sum().clamp_min(1) # DAPO-like normalization
    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens)
        ratio = torch.exp(new_logp - old_logp)
        masked_adv = adv[:, None] * mask
        surr = torch.min(ratio* masked_adv, ratio.clamp(1 - eps, 1 + eps) * masked_adv)
        kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1  # k3 estimator
        loss = -(surr * mask).sum() / n_tok + beta * (kl * mask).sum() / n_tok 
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


# The fundamental problem that GRPO or DAPO make different assumptions about is credit assignment.
# GRPO: Every response is equally important.
# DAPO: Every token is equally important.
# Length-dependent weighting: more or less like an interpolation between the two.
def ldw_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
             eps=0.2, beta=0.02, epochs=2, device=None, alpha=0.5):
    """Update using length-dependent weighting of the clipped surrogate and KL.

    ``batch`` contains sequences, prompt lengths, and raw rewards, ordered in
    contiguous groups for each prompt. A response of length L receives token
    weight L**(-alpha), normalized by sum(L**(1-alpha)). Thus alpha=0
    averages tokens, alpha=1 averages response means, and alpha=0.5
    interpolates. Lengths count only completion tokens present in ``seqs``.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seqs, plens, rwd_t = batch
    if not seqs or len(seqs) != len(plens) or len(seqs) != len(rwd_t):
        raise ValueError("batch must contain equally many sequences, prompt lengths, and rewards")
    if group < 2 or len(seqs) % group:
        raise ValueError("batch must contain complete groups of at least two responses")
    if any(not 1 <= plen < len(seq) for seq, plen in zip(seqs, plens)):
        raise ValueError("each sequence needs a nonempty prompt and completion")
    model_device = next(net.parameters()).device
    if device is not None and torch.empty(0, device=device).device != model_device:
        raise ValueError("device must match the model device")
    rewards = torch.as_tensor(rwd_t, device=model_device, dtype=torch.float32)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    grouped = rewards.reshape(-1, group)
    adv = ((grouped - grouped.mean(dim=1, keepdim=True)) /
           (grouped.std(dim=1, keepdim=True) + 1e-4)).reshape(-1)

    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)
    lengths = mask.sum(dim=1)
    weights = lengths.pow(-alpha)
    normalizer = lengths.pow(1 - alpha).sum()
    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens, pad_tok=pad_tok)
        ratio = torch.exp(new_logp - old_logp)
        token_adv = adv[:, None]
        surr = torch.min(ratio * token_adv,
                         ratio.clamp(1 - eps, 1 + eps) * token_adv)
        d = ref_logp - new_logp
        kl = d.exp() - d - 1
        # Apply the same length weighting to both terms of the token objective.
        per_response_loss = ((-surr + beta * kl) * mask).sum(dim=1)
        loss = (weights * per_response_loss).sum() / normalizer
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())

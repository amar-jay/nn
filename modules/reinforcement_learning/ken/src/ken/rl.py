import torch
import torch.nn.functional as F
from ken.countdown import exact_reward, reward

def _completion_logps(net: torch.nn.Module, seqs: list[list[int]], plens: list[int], pad_tok: int=2,
                      return_distribution=False):
    """Per-token logprobs of the completion part of each sequence. Returns padded tensor + mask."""
    device = next(net.parameters()).device
    maxlen = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), maxlen), pad_tok, device=device, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s, device=device)
    distribution = F.log_softmax(net(idx)[:, :-1], dim=-1)
    logp = distribution.gather(-1, idx[:, 1:, None]).squeeze(-1)
    pos = torch.arange(idx.shape[1] - 1, device=device)[None]
    plens_t = torch.tensor(plens, device=device)[:, None]
    lens_t = torch.tensor([len(s) for s in seqs], device=device)[:, None]
    mask = (pos >= plens_t - 1) & (pos < lens_t - 1)  # token t predicted from prefix <: completion
    if return_distribution:
        return logp, mask.float(), distribution
    return logp, mask.float()



METRIC_NAMES = (
    "ratio", "active_clip", "surrogate", "KL",
    "masked_ratio", "masked_adv_ratio", "masked_surrogate", "masked_KL",
    "exact_rate", "equal_reward_groups", "weighted_KL", "clip_fraction",
)


def _record_metrics(history, ratio, adv, surr, kl, mask, rewards, group, eps,
                    weights=None, clip_surrogate=True):
    """Record pre-step batch metrics; masked means exclude prompt and padding."""
    with torch.no_grad():
        count = mask.sum().clamp_min(1)
        def completion_mean(value):
            return (value * mask).sum() / count
        if weights is None:
            weights = torch.ones_like(mask[:, 0])
        weighted_mask = mask * weights[:, None]
        grouped = rewards.reshape(-1, group)
        active_clip = (((ratio > 1 + eps) & (adv > 0)) |
                       ((ratio < 1 - eps) & (adv < 0)))
        if not clip_surrogate:
            active_clip = torch.zeros_like(active_clip)
        values = (
            ratio.mean(), completion_mean(active_clip.float()), surr.mean(), kl.mean(),
            completion_mean(ratio), completion_mean(ratio * adv),
            completion_mean(surr), completion_mean(kl),
            (rewards >= 1.0).float().mean(),
            (grouped.max(dim=1).values == grouped.min(dim=1).values).float().mean(),
            (kl * weighted_mask).sum() / weighted_mask.sum().clamp_min(1),
            completion_mean(((ratio < 1 - eps) | (ratio > 1 + eps)).float()),
        )
        for series, value in zip(history, values):
            series.append(value.item())


# --------------- TRYOUTS ---------------

def grpo_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2, eps=0.2, beta=0.02, epochs=2, device="cpu", clip_surrogate=True):
    seqs, plens, rwd_t = batch

    # group-normalized advantages (the heart of GRPO: no value network)
    adv = torch.zeros_like(rwd_t, device=device)
    for i in range(len(rwd_t) // group):
        group_rewards = rwd_t[i * group : (i + 1) * group]
        adv[i * group : (i + 1) * group] = (
            group_rewards - group_rewards.mean()
        ) / (group_rewards.std() + 1e-4)

    # Frozen policy probabilities
    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)

    metric_history = [[] for _ in METRIC_NAMES]

    n_tok = mask.sum().clamp_min(1) # DAPO-like normalization
    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens, pad_tok=pad_tok)
        ratio = torch.exp(new_logp - old_logp)

        masked_adv = adv[:, None] * mask
        raw_surr = ratio * adv[:, None]
        unmasked_surr = (torch.min(raw_surr, ratio.clamp(1 - eps, 1 + eps) * adv[:, None])
                         if clip_surrogate else raw_surr)
        surr = unmasked_surr * mask
        # With beta=0, retain reference drift as a diagnostic only.
        kl_logp = new_logp if beta != 0 else new_logp.detach()
        d = ref_logp - kl_logp
        kl = d.exp() - d - 1

        loss = -(surr * mask).sum() / n_tok
        if beta != 0:
            loss = loss + beta * (kl * mask).sum() / n_tok

        opt.zero_grad()
        loss.backward()
        opt.step()

        _record_metrics(metric_history, ratio, adv[:, None], unmasked_surr, kl,
                        mask, rwd_t, group, eps, clip_surrogate=clip_surrogate)


    return float(loss.detach()), tuple(metric_history)


# The fundamental problem that GRPO or DAPO make different assumptions about is credit assignment.
# GRPO: Every response is equally important.
# DAPO: Every token is equally important.
# Length-dependent weighting: more or less like an interpolation between the two.

def ldw_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2, eps=0.2, beta=0.02, epochs=2, device="cpu",
    alpha=0.5,
):
    """Update using length-dependent weighting of the clipped surrogate and KL.
    A response of length L receives token
    weight L**(-alpha), normalized by sum(L**(1-alpha)). Thus alpha=0
    averages tokens, alpha=1 averages response means, and alpha=0.5
    interpolates. Lengths count only completion tokens present in ``seqs``.
    """
    assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"
    seqs, plens, rwd_t = batch

    # Group-relative advantages
    grouped = rwd_t.reshape(-1, group)
    adv = (grouped - grouped.mean(dim=1, keepdim=True)
    ) / (grouped.std(dim=1, keepdim=True,unbiased=False,) + 1e-4)
    adv = adv.reshape(-1, 1)

    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)

    # Length-dependent weights
    n = mask.sum(dim=1)
    valid = n > 0
    n = n.clamp_min(1)
    weights = n.pow(-alpha) * valid
    norm = (n * weights).sum().clamp_min(1)

    # for visualization
    metric_history = [[] for _ in METRIC_NAMES]

    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens, pad_tok=pad_tok)
        ratio = torch.exp(new_logp - old_logp)

        surr = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
        kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1  # k3 estimator

        tok_loss = -surr + beta * kl # Per-token objective
        per_resp = (tok_loss * mask).sum(dim=1) # Length-dependent reduction

        loss = (weights * per_resp).sum() / norm

        opt.zero_grad()
        loss.backward()
        opt.step()
        _record_metrics(metric_history, ratio, adv, surr, kl, mask,
                        rwd_t, group, eps, weights=weights)


    return float(loss.detach()), tuple(metric_history)


def rlvr_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
              eps=0.2, epochs=2, device="cpu"):
    """No-reference-penalty ablation of grpo_step, using verifier rewards.

    RLVR describes the reward source, not a distinct optimization algorithm.
    This experiment changes only beta to zero: group advantages, clipping,
    token normalization, and sampling stay the same. Reference KL is still
    logged for comparison but has no effect on gradients. Reward shaping
    remains whatever the caller's verifier supplies.
    """
    return grpo_step(net, old_net, ref_net, opt, batch, group=group,
                     pad_tok=pad_tok, eps=eps, beta=0.0, epochs=epochs,
                     device=device)


def rlvr_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
                 eps=0.2, beta=0.01, epochs=2, device="cpu"):
    """Exact-reward experiment with the reference KL penalty retained.

    Matches rlvr_step except for beta; select_experiment pairs both with
    exact_reward so their difference isolates reference regularization.
    """
    return grpo_step(net, old_net, ref_net, opt, batch, group=group,
                     pad_tok=pad_tok, eps=eps, beta=beta, epochs=epochs,
                     device=device)


def unclip_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
               eps=0.2, beta=0.02, epochs=2, device="cpu"):
    """Unclipped r*A ablation of rlvr_kl_step, with exact rewards and KL.

    eps only defines the diagnostic out-of-bounds fraction in this variant;
    it does not restrict the policy surrogate.
    """
    return grpo_step(net, old_net, ref_net, opt, batch, group=group,
                     pad_tok=pad_tok, eps=eps, beta=beta, epochs=epochs,
                     device=device, clip_surrogate=False)


def rloo_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
                 eps=0.2, beta=0.02, epochs=2, device="cpu"):
    """RLOO-style advantage ablation with clipping, token averaging and KL.

    A_i = R_i - mean(R_j for j != i), with no standard-deviation scaling.
    The paired verifier is exact-only. This retains the existing PPO-style
    update rather than implementing a separate full RLOO trainer.
    """
    seqs, plens, rewards = batch
    if group < 2 or rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group:
        raise ValueError("rewards must contain complete groups of at least two answers")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    # Each answer is compared only with the OTHER answers for its prompt.
    grouped = rewards.reshape(-1, group)
    others_mean = (grouped.sum(dim=1, keepdim=True) - grouped) / (group - 1)
    adv = (grouped - others_mean).reshape(-1, 1)

    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)
    n_tok = mask.sum().clamp_min(1)
    metric_history = [[] for _ in METRIC_NAMES]

    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens, pad_tok=pad_tok)
        ratio = (new_logp - old_logp).exp()
        surr = torch.min(ratio * adv,
                         ratio.clamp(1 - eps, 1 + eps) * adv)
        kl_logp = new_logp if beta != 0 else new_logp.detach()
        d = ref_logp - kl_logp
        kl = d.exp() - d - 1
        loss = -(surr * mask).sum() / n_tok
        if beta != 0:
            loss = loss + beta * (kl * mask).sum() / n_tok

        opt.zero_grad()
        loss.backward()
        opt.step()
        _record_metrics(metric_history, ratio, adv, surr, kl, mask,
                        rewards, group, eps)

    return float(loss.detach()), tuple(metric_history)


def rms_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
                eps=0.2, beta=0.02, epochs=2, device="cpu", rms_eps=1e-6, detach_rms=False):
    """Unclipped -mean(r*A / RMS(r*A)) plus reference KL.

    RMS is computed over completion tokens. Set detach_rms=True to treat
    its denominator as a fixed scale during backpropagation.
    Advantages use the same group mean/std as rlvr_kl. eps is only the
    diagnostic ratio threshold; rms_eps stabilizes the normalization.
    """
    seqs, plens, rewards = batch
    if group < 2 or rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group:
        raise ValueError("rewards must contain complete groups of at least two answers")
    if epochs < 1 or rms_eps <= 0:
        raise ValueError("epochs and rms_eps must be positive")
    grouped = rewards.reshape(-1, group)
    adv = ((grouped - grouped.mean(dim=1, keepdim=True))
        / (grouped.std(dim=1, keepdim=True, correction=0) + 1e-4)).reshape(-1, 1).detach()
    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        ref_logp, _ = _completion_logps(ref_net, seqs, plens, pad_tok=pad_tok)
    valid = mask.bool()
    if not valid.any():
        raise ValueError("batch must contain at least one completion token")
    history = [[] for _ in METRIC_NAMES]
    for _ in range(epochs):
        new_logp, _ = _completion_logps(net, seqs, plens, pad_tok=pad_tok)
        ratio = (new_logp - old_logp).exp()
        z = ratio * adv
        z_valid = z[valid]
        rms = (z_valid.square().mean() + rms_eps).sqrt()
        if detach_rms:
            rms = rms.detach()
        normalized = z / rms
        policy_loss = -z_valid.mean() / rms
        kl_logp = new_logp if beta != 0 else new_logp.detach()
        d = ref_logp - kl_logp
        kl = d.exp() - d - 1
        loss = policy_loss
        if beta != 0:
            loss = loss + beta * kl[valid].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Surrogate metrics show the normalized objective, adv_ratio the raw r*A.
        _record_metrics(history, ratio, adv, normalized, kl, mask,
                        rewards, group, eps, clip_surrogate=False)
    return float(loss.detach()), tuple(history)


def rms_ref_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
                eps=0.2, beta=0.02, epochs=2, device="cpu", rms_eps=1e-6, detach_rms=False):
    """Unclipped RMS policy objective plus exact KL(reference || current).

    RMS is computed over completion tokens. Set detach_rms=True to treat
    its denominator as a fixed scale during backpropagation.
    Advantages use the same group mean/std as rlvr_kl. eps is only the
    diagnostic ratio threshold; rms_eps stabilizes the normalization.
    """
    seqs, plens, rewards = batch
    if group < 2 or rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group:
        raise ValueError("rewards must contain complete groups of at least two answers")
    if epochs < 1 or rms_eps <= 0:
        raise ValueError("epochs and rms_eps must be positive")
    grouped = rewards.reshape(-1, group)
    adv = ((grouped - grouped.mean(dim=1, keepdim=True))
        / (grouped.std(dim=1, keepdim=True, correction=0) + 1e-4)).reshape(-1, 1).detach()
    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        _, _, ref_distribution = _completion_logps(
            ref_net, seqs, plens, pad_tok=pad_tok, return_distribution=True)
    valid = mask.bool()
    if not valid.any():
        raise ValueError("batch must contain at least one completion token")
    history = [[] for _ in METRIC_NAMES]
    for _ in range(epochs):
        new_logp, _, new_distribution = _completion_logps(
            net, seqs, plens, pad_tok=pad_tok, return_distribution=True)
        ratio = (new_logp - old_logp).exp()
        z = ratio * adv
        z_valid = z[valid]
        rms = (z_valid.square().mean() + rms_eps).sqrt()
        if detach_rms:
            rms = rms.detach()
        normalized = z / rms
        policy_loss = -z_valid.mean() / rms
        kl = (ref_distribution.exp() * (ref_distribution - new_distribution)).sum(dim=-1)
        loss = policy_loss
        if beta != 0:
            loss = loss + beta * kl[valid].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Surrogate metrics show the normalized objective, adv_ratio the raw r*A.
        _record_metrics(history, ratio, adv, normalized, kl, mask,
                        rewards, group, eps, clip_surrogate=False)
    return float(loss.detach()), tuple(history)


def grpo_ref_kl_step(net, old_net, ref_net, opt, batch, group=12, pad_tok=2,
                     eps=0.2, beta=0.02, epochs=2, device="cpu"):
    """Clipped group-relative policy update with exact KL(reference || current).

    Sums over the whole vocabulary at each stored completion prefix. The
    prefixes still come from old-policy rollouts; exact refers to the token
    distribution sum, not an expectation over all possible trajectories.
    Paired with exact_reward to compare directly against rlvr_kl.
    """
    seqs, plens, rewards = batch
    if group < 2 or rewards.ndim != 1 or not rewards.numel() or rewards.numel() % group:
        raise ValueError("rewards must contain complete groups of at least two answers")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    grouped = rewards.reshape(-1, group)
    adv = ((grouped - grouped.mean(dim=1, keepdim=True)) /
           (grouped.std(dim=1, keepdim=True) + 1e-4)).reshape(-1, 1)
    with torch.no_grad():
        old_logp, mask = _completion_logps(old_net, seqs, plens, pad_tok=pad_tok)
        _, _, ref_distribution = _completion_logps(
            ref_net, seqs, plens, pad_tok=pad_tok, return_distribution=True)
    count = mask.sum().clamp_min(1)
    history = [[] for _ in METRIC_NAMES]
    for _ in range(epochs):
        new_logp, _, new_distribution = _completion_logps(
            net, seqs, plens, pad_tok=pad_tok, return_distribution=True)
        ratio = (new_logp - old_logp).exp()
        surr = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
        # q is frozen reference; p is the current policy. Gradient wrt logits: p-q.
        kl = (ref_distribution.exp() * (ref_distribution - new_distribution)).sum(dim=-1)
        loss = -(surr * mask).sum() / count + beta * (kl * mask).sum() / count
        opt.zero_grad()
        loss.backward()
        opt.step()
        # All KL panels now report the exact vocabulary sum in this direction.
        _record_metrics(history, ratio, adv, surr, kl, mask, rewards, group, eps)
    return float(loss.detach()), tuple(history)


def rl_step(*args, **kwargs):
    return grpo_step(*args, **kwargs)
rl_step.name = grpo_step.__name__



experiments = {
    "grpo": (grpo_step, reward),
    "ldw": (ldw_step, reward),
    "rlvr": (rlvr_step, exact_reward),
    "rlvr_kl": (rlvr_kl_step, exact_reward), # this is basically grpo_step with an exact_reward verifier and beta=0.01
    "unclip_kl": (unclip_kl_step, exact_reward),
    "rloo_kl": (rloo_kl_step, exact_reward),
    "rms_kl": (rms_kl_step, exact_reward),
    "rms_ref_kl": (rms_ref_kl_step, exact_reward),
    "grpo_ref_kl": (grpo_ref_kl_step, exact_reward),
}

def select_experiment(rl_step="grpo"):
    """Select an RL update and its paired verifier reward together."""
    if rl_step not in experiments:
        raise ValueError(f"Unknown RL step: {rl_step}. Choose {', '.join(experiments)}.")
    return experiments[rl_step]

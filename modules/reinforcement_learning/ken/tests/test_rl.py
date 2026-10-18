import pytest
import torch
from torch import nn

from ken import rl


@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_ldw_loss_and_gradient_match_length_weighted_objective(monkeypatch, alpha):
    # Unequal completion lengths distinguish all three normalization schemes.
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    current = nn.Linear(1, 1, bias=False)
    old = nn.Linear(1, 1, bias=False)
    ref = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        current.weight.fill_(0.1)
    calls = []

    def logps(model, seqs, plens, pad_tok=2):
        calls.append(pad_tok)
        if model is current:
            return current.weight.expand(2, 3), mask
        value = 0.0 if model is old else -0.2
        return torch.full((2, 3), value), mask

    monkeypatch.setattr(rl, '_completion_logps', logps)
    rewards = torch.tensor([1., 0.])
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    theta = torch.tensor(0.1, requires_grad=True)
    ratio = theta.exp()
    surrogate = torch.minimum(ratio * advantages,
                              ratio.clamp(0.8, 1.2) * advantages)
    kl = (-0.2 - theta).exp() - (-0.2 - theta) - 1
    # Independently average each response, then weight by length**(1-alpha).
    lengths = torch.tensor([1., 3.])
    aggregate_weights = lengths.pow(1 - alpha)
    expected = ((-surrogate + 0.02 * kl) * aggregate_weights).sum() / aggregate_weights.sum()
    expected.backward()

    actual, metrics = rl.ldw_step(current, old, ref,
                         torch.optim.SGD(current.parameters(), lr=0),
                         ([[3, 4], [3, 4, 5, 6]], [1, 1], rewards),
                         group=2, pad_tok=7, epochs=1, alpha=alpha)
    assert actual == pytest.approx(expected.item())
    assert current.weight.grad.item() == pytest.approx(theta.grad.item(), rel=1e-5, abs=1e-7)
    assert calls == [7, 7, 7]


@pytest.mark.parametrize('alpha', [-0.1, 1.1])
def test_ldw_rejects_invalid_alpha(alpha):
    with pytest.raises(AssertionError, match='alpha'):
        rl.ldw_step(None, None, None, None, None, alpha=alpha)


def test_metrics_use_whole_batch_and_completion_denominator():
    mask = torch.tensor([[1., 0., 0.], [1., 1., 0.]])
    ratio = torch.tensor([[1., 99., 99.], [1.5, 0.5, 99.]])
    kl = torch.tensor([[0.1, 100., 100.], [0.2, 0.3, 100.]])
    adv = torch.tensor([[1.], [-1.]])
    history = [[] for _ in rl.METRIC_NAMES]
    rl._record_metrics(history, ratio, adv, ratio * adv, kl, mask,
                       torch.tensor([1., 0.]), 2, 0.2,
                       weights=torch.tensor([1., 0.5]))
    metrics = dict(zip(rl.METRIC_NAMES, [x[0] for x in history]))
    assert metrics['masked_ratio'] == pytest.approx(1.0)
    assert metrics['masked_KL'] == pytest.approx(0.2)
    assert metrics['weighted_KL'] == pytest.approx(0.175)
    assert metrics['exact_rate'] == 0.5
    assert metrics['equal_reward_groups'] == 0.0
    assert metrics['clip_fraction'] == pytest.approx(2 / 3)


def test_rlvr_update_is_independent_of_reference(monkeypatch):
    mask = torch.tensor([[1., 0.], [1., 1.]])
    batch = ([[3, 4], [3, 4, 5]], [1, 1], torch.tensor([1., 0.]))
    outcomes = []
    for reference_value in [-0.2, 1.0]:
        current = nn.Linear(1, 1, bias=False)
        old, ref = object(), object()
        with torch.no_grad():
            current.weight.fill_(0.1)

        def logps(model, seqs, plens, pad_tok=2):
            if model is current:
                return current.weight.expand(2, 2), mask
            value = 0.0 if model is old else reference_value
            return torch.full((2, 2), value), mask

        monkeypatch.setattr(rl, '_completion_logps', logps)
        loss, histories = rl.rlvr_step(
            current, old, ref, torch.optim.SGD(current.parameters(), lr=0.1),
            batch, group=2, epochs=1,
        )
        outcomes.append((loss, current.weight.item(), histories))
    assert outcomes[0][0] == pytest.approx(outcomes[1][0])
    assert outcomes[0][1] == pytest.approx(outcomes[1][1])
    # Reference metrics still show drift despite having no optimization effect.
    index = rl.METRIC_NAMES.index('weighted_KL')
    assert outcomes[0][2][index] != outcomes[1][2][index]


def test_unclip_kl_uses_unclipped_surrogate_outside_bounds(monkeypatch):
    from ken.countdown import exact_reward
    assert rl.select_experiment('unclip_kl') == (rl.unclip_kl_step, exact_reward)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    old, ref = object(), object()
    mask = torch.ones(2, 1)

    def logps(net, seqs, plens, pad_tok=2):
        if net is model:
            # Positive answer has ratio 2: clipped objective would plateau.
            return model.weight.expand(2, 1) + torch.tensor([[0.69314718], [0.]]), mask
        return torch.zeros(2, 1), mask

    monkeypatch.setattr(rl, '_completion_logps', logps)
    batch = ([[3, 4], [3, 5]], [1, 1], torch.tensor([1., 0.]))
    _, histories = rl.unclip_kl_step(model, old, ref,
                                torch.optim.SGD(model.parameters(), lr=0),
                                batch, group=2, epochs=1, beta=0)
    adv = (batch[2] - batch[2].mean()) / (batch[2].std() + 1e-4)
    expected = -(2 * adv[0] + adv[1]) / 2
    assert model.weight.grad.item() == pytest.approx(expected.item())
    index = rl.METRIC_NAMES.index
    assert histories[index('masked_surrogate')] == histories[index('masked_adv_ratio')]

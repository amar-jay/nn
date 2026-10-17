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
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
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

    actual = rl.ldw_step(current, old, ref,
                         torch.optim.SGD(current.parameters(), lr=0),
                         ([[3, 4], [3, 4, 5, 6]], [1, 1], rewards),
                         group=2, pad_tok=7, epochs=1, alpha=alpha)
    assert actual == pytest.approx(expected.item())
    assert current.weight.grad.item() == pytest.approx(theta.grad.item())
    assert calls == [7, 7, 7]


@pytest.mark.parametrize('alpha', [-0.1, 1.1])
def test_ldw_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match='alpha'):
        rl.ldw_step(None, None, None, None, None, alpha=alpha)

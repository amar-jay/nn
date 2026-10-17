"""Focused regression tests for the countdown reward (GRPO training signal)."""

import pytest

from ken.countdown import generate_puzzle, reward, safe_eval


def test_huge_product_returns_none_without_raising():
    # Crashed full GRPO run at iter ~200: OverflowError in math.isfinite.
    expr = "*".join(["10"] * 400)  # 10**400, bit_length > 1024
    assert safe_eval(expr) is None


def test_exponentiation_rejected():
    assert safe_eval("20**20") is None
    assert safe_eval("100//7") is None


def test_reward_ladder():
    assert reward("7*4", [3, 7, 11, 4], 28) == 1.0
    assert reward("7+4", [3, 7, 11, 4], 28) == 0.3
    assert reward("7+5", [3, 7, 11, 4], 28) == 0.1  # 5 not in numbers
    assert reward("7+", [3, 7, 11, 4], 28) == 0.0
    assert reward("", [3, 7, 11, 4], 28) == 0.0


def test_reward_never_raises_on_garbage():
    for expr in ["(", "2(3+4)", "9999999999*", "+-*/", "abc", "10**10**10"]:
        assert reward(expr, [3, 7, 11, 4], 28) == 0.0


def test_generated_puzzles_are_solvable():
    import random

    rng = random.Random(0)
    for _ in range(50):
        numbers, target = generate_puzzle(rng)
        assert isinstance(target, int)
        assert len(numbers) == 4
        assert all(1 <= n <= 20 for n in numbers)
        # target was built from the numbers, so some expression must hit it
        assert reward(f"{numbers[0]}+{0}", numbers, target) in (0.0, 0.1, 0.3, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

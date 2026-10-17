"""Regression tests for the Stage-2 LLM reward wrapper."""

from ken.grpo_llm import countdown_reward


def test_empty_completion_scores_zero_without_raising():
    # Crashed the 50-step run at step 15: IndexError on "".
    assert countdown_reward([""], numbers=[[3, 7, 11, 4]], target=[24]) == [0.0]
    assert countdown_reward(["   \n  "], numbers=[[3, 7, 11, 4]], target=[24]) == [0.0]


def test_best_candidate_wins_over_prose():
    assert countdown_reward(
        [
            "The answer is (11-3)*(7-4), using each number once.",
            "7*4-11+3",
            "blah 99+1 blah",
        ],
        numbers=[[3, 7, 11, 4]] * 3,
        target=[24, 20, 100],
    ) == [1.0, 1.0, 0.1]

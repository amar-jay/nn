# ken — a small lab for reinforcement-learning experiments

Countdown is the test environment, not the research objective. This project uses a small character-level transformer and a verifiable arithmetic task to study how reward design, advantage estimation, normalization, clipping, and regularization change learning. The aim is to make experiments understandable and affordable on a laptop GPU before introducing a larger model or more complicated task.

The results below are observations from our learning sessions, not benchmark claims. Most variants have only one training seed; apparent rankings need replication.

## Run the current code

```bash
uv sync
uv run ken train --numbers 3 --rl-step grpo_ref_kl
uv run ken infer --numbers 3 --count 1000 --show 10 --seed 0
uv run ken diagnose --numbers 3 --count 100 --samples 12 --seed 12345
```

Training uses CUDA and raises an error if CUDA is unavailable. Inference and diagnostics automatically use CUDA when available, otherwise CPU. The CLI currently trains for **2,100 SFT steps and 600 RL iterations**, with 64 prompts per iteration, 12 answers per prompt, and up to 48 generated tokens. SFT batch size is 64: 134,400 generated warm-up examples.

Training overwrites `grpo_toy.pt`. Preserve a checkpoint manually before another run if you want to compare models. Diagnostics print results and return an in-memory report; they do not write JSON files.

**Configuration details that matter:**

- `--rl-step` selects both the update function and its paired reward function.
- The CLI currently parses training `--seed` but does **not** forward it to `train()`. CLI training therefore uses `train`'s default seed, 0. Evaluation `--seed` is forwarded correctly. Different evaluation seeds are not different training seeds.
- Python `train()` defaults differ from CLI settings. For explicit experiments, pass the full configuration:

```python
from ken.grpo_toy import train

metrics = train(
    n_numbers=3, rl_step="grpo_ref_kl", seed=0,
    sft_steps=2100, iters=600, prompts_per_iter=64, group=12,
)
```

- Coefficients are currently function parameters in `src/ken/rl.py`, not CLI arguments. The current `rlvr_kl_step` default beta is **0.01**; most other KL variants default to **0.02**. Historical runs may have used earlier defaults. Record the coefficient when comparing runs.
- Optional before/after diagnostic calls in `train()` are currently commented out. The standalone `diagnose` command is available.

## Environment and warm-up

A prompt such as `6 20 18:2160=` asks for an expression using the available numbers to reach 2160. The `=` marks where the answer begins. `((6*20)*18)` is a correct completion.

The generator draws numbers from 1 through 20, shuffles them, and constructs a guaranteed-solvable target by combining all numbers left-to-right with `+`, `-`, or `*`. The verifier allows a solution to use **any subset of at least two operands**, respecting the multiplicity of supplied numbers. Unused numbers are allowed. Division is also accepted by the verifier, though the generator uses only the other three operations.

The warm-up teaches expression format and operand copying, not target correctness. It keeps all puzzle numbers in the prompt but uniformly selects an operand count from 2 through the number supplied, then builds a random nested expression. Thus three-number warm-up mixes two- and three-operand answers. These are valid expressions, not necessarily solutions.

The model is a small causal transformer: two layers, four attention heads, width 128, and a 128-token positional capacity. It uses character tokens plus BOS/EOS/PAD. RL rollouts retain EOS, allowing the loss to train the decision to stop. A length-limited rollout does not receive an artificial EOS.

Currently `sample()` batches answers **within one prompt** and drops finished rows from its active batch. Training processes different prompts sequentially. Cross-prompt batching was explored earlier but is not present in this checkout.

## Reward functions

| Outcome | Shaped `reward` | Binary `exact_reward` |
|---|---:|---:|
| Correct value and allowed operands | 1.0 | 1.0 |
| Valid expression, allowed operands, wrong value | 0.3 | 0.0 |
| Parseable arithmetic with invalid operand usage | 0.1 | 0.0 |
| Invalid expression | 0.0 | 0.0 |

These are deterministic verifiers, not learned reward models. Rewards are fixed feedback; detaching their tensors makes that explicit but does not change gradients when they already have no computation history.

Inference reports the shaped mean reward for diagnostic purposes, even when training used exact-only rewards. Use **exact accuracy** to compare solving performance across reward configurations.

## The update and model roles

Three policies have different jobs:

- **Current `net`, p:** the model receiving gradients.
- **`old_net`:** a frozen snapshot refreshed each rollout iteration. It generates the batch and supplies the policy-ratio denominator.
- **Reference `ref_net`, q:** the frozen post-SFT model, kept throughout RL.

Answers are grouped by prompt. The usual advantage is the completion reward minus its group's mean, divided by its group's standard deviation plus a small epsilon. Each completion's advantage is broadcast over its tokens.

For each stored answer token:

```text
ratio = exp(new_logp - old_logp)
surrogate = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

The policy loss is the negative masked average of the surrogate. Clipping caps the incentive for sufficiently large changes in the favorable direction; it does not impose hard bounds on token probabilities. Unfavorable changes remain penalized.

With exact-only rewards, all-failure and all-success groups both have zero advantages. Only mixed groups provide a relative policy signal. A reference penalty may still contribute when advantages are zero.

**Naming caveat:** the function called `grpo_step` uses a batch-wide token average, which is DAPO-like normalization. Original response-averaged GRPO gives each response equal aggregate weight. These function names describe this educational code, not full reproductions of external trainers. RLVR describes verifiable rewards, not a uniquely defined optimizer.

## Available variants

| `--rl-step` | Policy term | Anchor | Reward |
|---|---|---|---|
| `grpo` | Standardized group advantages, clipped surrogate | Sampled reference K3 | Shaped |
| `ldw` | Clipped surrogate, length-dependent averaging | Sampled reference K3 | Shaped |
| `rlvr` | Same token-averaged clipped update | No penalty; reference drift logged | Exact-only |
| `rlvr_kl` | Same token-averaged clipped update | Sampled reference K3 | Exact-only |
| `unclip_kl` | Direct ratio × advantage | Sampled reference K3 | Exact-only |
| `rloo_kl` | Leave-one-out advantages, clipped surrogate | Sampled reference K3 | Exact-only |
| `rms_kl` | Unclipped RMS-normalized ratio × advantage | Sampled reference K3 | Exact-only |
| `grpo_ref_kl` | Standardized group advantages, clipped surrogate | Exact vocabulary KL(q‖p) | Exact-only |
| `rms_ref_kl` | RMS-normalized ratio × advantage | Exact vocabulary KL(q‖p) | Exact-only |

RLOO is implemented with its own update loop. Its advantage is reward minus the mean reward of the **other** answers for the prompt, without standard-deviation normalization. It retains this project's clipped token-level policy update rather than implementing an entire external RLOO trainer.

LDW weights each response's token sum by `length**(-alpha)` and divides by the sum of `length**(1-alpha)`. At alpha=0 it averages tokens; at alpha=1 it averages response means. The default alpha is 0.5. Both the policy and KL terms receive that weighting.

RMS variants define `z = ratio * advantage` and minimize `-mean(z) / sqrt(mean(z**2) + rms_eps)` over completion tokens. `detach_rms=False` differentiates through the denominator; setting it to `True` treats the denominator as a fixed scale for that update. Keep this setting matched in comparisons. RMS variants currently use population standard deviation (`correction=0`) for group advantages; the clipped GRPO-style variants use sample standard deviation. RMS normalization largely cancels uniform advantage rescaling, apart from numerical epsilons.

## KL: value, direction, and gradients

Sampled K3 uses the log probabilities of the same generated token:

```python
d = ref_logp - new_logp
k3 = d.exp() - d - 1
```

For normalized positive distributions, its expectation under current-policy sampling equals KL(p‖q): the added ratio term has expectation 1 and cancels the subtracted 1. In training, stored tokens come from the old policy, so after updates this is a surrogate on old-policy samples.

An unbiased value estimate does not automatically give an unbiased gradient when differentiated through fixed samples. In particular, under current-policy sampling, direct differentiation of this K3 expression has expected logit gradient p−q, matching KL(q‖p)'s logit gradient. Under old-policy sampling that equality need not hold exactly.

The exact-reference variants explicitly calculate:

```python
kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
```

This sums the whole next-token vocabulary at each stored prefix. Its logit gradient is p−q; gradient descent moves in the corrective direction q−p. It is exact over vocabulary entries, **not** over all possible generated trajectories.

Exact-reference KL removed much of the noisy sampled-K3 spiking in our plots. Its raw magnitude is not directly comparable with the other KL direction/estimator, and a larger scalar KL does not necessarily imply a proportionally stronger gradient. Beta needs retuning for each objective.

## Reading the plots and diagnostics

Metrics are batch-wide and evaluated **before** their corresponding optimizer step. Two update epochs per rollout iteration produce 1,200 metric points for a 600-iteration run.

- `masked_*` averages over completion tokens, excluding prompt and padding predictions from both numerator and denominator. Plain metrics include those excluded positions and may look very different.
- `weighted_KL` uses the actual loss weighting; for token-averaged variants it matches masked KL, while LDW can differ.
- `clip_fraction` counts completion-token ratios outside the diagnostic interval. `active_clip` counts the subset where the clipped branch actually changes the surrogate. It is explicitly zero when clipping is disabled.
- `exact_rate` measures sampled training answers, not greedy evaluation accuracy.
- `equal_reward_groups` includes both all-correct and all-wrong groups. It must be interpreted alongside accuracy.
- `groups_with_exact` in terminal logs measures prompt coverage: groups containing at least one success. High answer accuracy within very few successful groups can indicate narrow specialization.

The `surrogate`, `masked_adv_ratio`, and `masked_surrogate` panels are histograms of **batch means across update epochs**, not individual token distributions. Outliers and mixed training stages can obscure their central shape. A large plain-surrogate spike at zero often reflects first epochs: ratio=1 and centered advantages cancel across equally sized padded rows. Completion masking introduces unequal response-length weights, so the same cancellation need not occur in masked metrics. Zero scalar loss does not imply zero gradient.

`diagnose` compares greedy accuracy, individual sampled accuracy, and the fraction of puzzles with at least one correct answer among 12 samples. It also reports allowed-number usage separately from syntax validity, expression diversity, and greedy answers to different solvable targets using identical numbers. Changing output with target is not sufficient evidence of understanding; the expressions must also reach those targets.

## What the experiments showed

The main empirical pattern was an interaction between reward and anchoring:

1. Shaped rewards frequently led to short, valid-but-wrong expressions, earning 0.3 without learning to hit the target.
2. Exact-only rewards removed that shortcut but were extremely sparse on four-number puzzles. Three numbers provided more successful rollouts.
3. Exact-only rewards with a frozen reference penalty generally outperformed the early no-KL runs and preserved syntax better.
4. Mixed-length SFT allowed shorter answers but did not itself solve target conditioning. Under shaped reward, training concentrated on valid two-number expressions.
5. More warm-up improved operand usage and downstream accuracy. Gains were substantial between 800 and 1,600 steps, then smaller by 3,200. Final single-batch SFT losses are noisy; these observations are not evidence of grokking. We adopted 2,100 steps as a practical budget.
6. Clipping activated on a small fraction of tokens, yet removing it changed trajectories. Rare activation is not proof of irrelevance.
7. RMS objectives learned without clipping, but differentiable versus detached RMS did not show a clear accuracy difference in the runs tried.
8. Exact vocabulary reference KL produced smoother diagnostics and competitive accuracy; it did not uniformly improve every policy objective.

Selected **historical** greedy evaluations, not a controlled leaderboard:

| Configuration | Evaluation seed / count | Accuracy |
|---|---|---:|
| Exact-only + KL, 1,600 SFT / 300 RL | 12345 / 100,000 | 31.393% |
| Exact-only + sampled KL, 2,100 SFT / 600 RL | 0 / 1,000 | 33.7% |
| RLOO + sampled KL, 2,100 SFT / 600 RL | 0 / 1,000 | 36.2% |
| Clipped policy + exact reference KL, 2,100 SFT / 600 RL | 0 / 1,000 | 37.1% |
| RMS + sampled KL | 12345 / 1,000 | 33–35% across tested settings |
| RMS + exact reference KL | 0 / 1,000 | 30.8% |

Coefficients, SFT lengths, evaluation sets, and other settings changed during development. Do not rank unmatched rows or treat evaluation-seed changes as training replication. Larger evaluation sets refined an estimate around 31–32% for one checkpoint; they did not cause that model to improve. Runs with the same seed and increasing counts contain nested puzzle sets.

Other ideas were implemented or discussed during the session but are **not currently registered in this checkout**:

- **Reward-weighted CE replacing KL:** reinforces verified successes, including all-success groups. A run reached 19.4% on seed 12345 / 1,000, with roughly 90–95% equal-reward groups. Success concentrated on few puzzles; replacement of the anchor did not work well at the tested setting.
- **RMS + rolling old-policy CE:** uses CE(old,current), equivalent in gradient to KL(old‖current). Its anchor refreshes every iteration and has zero gradient initially when old=current. Early logs showed sharply narrowing success coverage; no final result was established here.
- **Successful-answer replay:** discussed, but not implemented. Unlike frozen-reference KL, it would reinforce solutions discovered during RL.
- **Beta/group-size sweep:** an earlier runner tested G in {2,4,8,12,16,32}, selecting beta on validation data and fitting log(beta*) = log(c) + alpha*log(G). The 126-run default was too expensive for an initial screen. Partial G=2 results favored approximately 0.003–0.01 over 0.02; no scaling exponent was established. The runner/CLI are not in the current checkout.

### Approaches we tried and set aside

We tried **RWCE, rolling CE, the beta/G sweep runner, and cross-prompt batching**, but none became part of the working baseline:

- **RWCE replacing reference KL:** did not work well at the tested setting. Accuracy plateaued, success concentrated on a small subset of puzzles, and validity deteriorated.
- **Rolling old-policy CE:** did not work well in the observed early run. Success coverage narrowed sharply. Refreshing the anchor with each old-policy snapshot did not provide the protection of a fixed SFT reference; a final evaluation was not recorded.
- **Sweep runner:** the full 126-run experiment was impractically slow on the laptop GPU (already over four hours into partial results). It did not establish the proposed beta/G scaling relationship.
- **Cross-prompt batching:** implemented and tested, then set aside. It did not become part of the working setup; we did not establish a measured speedup or accuracy benefit. This was an execution optimization, not evidence against a policy objective.

**Our practical conclusion: retain the frozen reference.** Removing it or replacing it with success imitation or a rolling anchor has not matched our strongest reference-regularized results. For subsequent experiments, frozen-reference regularization stays in the baseline unless its removal is the explicit variable under test. This is a conclusion about the configurations we tested, not proof that every RL method requires a reference model.

## Next experiments and controls

Use a fixed SFT budget, coefficients, rollout budget, and evaluation set when changing one component. Repeat actual training seeds before drawing conclusions from small accuracy differences. The current training-seed forwarding issue needs resolving first.

For a beta/G study, keep completions per iteration and optimizer-update count fixed. With 768 completions, prompts per iteration are 768/G. This controls answer count but necessarily changes distinct prompt count; variable answer lengths also mean token/FLOP budgets are not exactly matched. Start with a small screening grid, then refine promising regions. Boundary optima and tied scores make an inferred power-law slope unreliable.

Remaining weaknesses are primarily operand copying/repetition and operation/order selection, rather than stopping. Measure both before changing another loss. Evaluate target sensitivity with fixed numbers and multiple targets, and distinguish discovery (sampled solution coverage) from concentration (greedy accuracy). Do not infer a hard 40% ceiling from several related objectives sharing the same model and warm-up.

## Files and checks

| File | Purpose |
|---|---|
| `src/ken/countdown.py` | Puzzle generation, character encoding, verifier and rewards |
| `src/ken/grpo_toy.py` | TinyGPT, warm-up, rollout collection, training and greedy inference |
| `src/ken/rl.py` | RL update variants, objective calculations and metric recording |
| `src/ken/diagnostics.py` | Sampled coverage, operand usage and target sensitivity |
| `src/ken/__init__.py` | CLI and plots |
| `src/ken/test.ipynb` | Step-by-step scratch notebook |
| `src/ken/grpo_llm.py` | Separate TRL-based larger-model experiment; not the focus of these results |


Background reading: [DeepSeekMath](https://arxiv.org/abs/2402.03300), [PPO](https://arxiv.org/abs/1707.06347), and [Schulman's KL estimator note](https://joschu.net/blog/kl-approx.html). The custom variants here are experiments, not claims of novel algorithms or faithful implementations of every detail in those sources.

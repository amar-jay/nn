# ken — GRPO, hands-on

Learning repo: Group Relative Policy Optimization, from a from-scratch
implementation on a toy game to a real LLM run with TRL. Built for a 4GB
laptop GPU (RTX 3050), Python 3.11, `uv`-managed venv.

Map:

- `src/ken/countdown.py` — the verifiable task + reward (both stages use it)
- `src/ken/grpo_toy.py` — Stage 1: GRPO written from scratch in pure torch
- `src/ken/grpo_llm.py` — Stage 2: GRPO on Qwen2.5-0.5B-Instruct via TRL
- `tests/` — regression tests for the reward code both stages depend on

## The 2-minute theory (what the code below implements)

GRPO trains a policy from **groups** of completions instead of a learned
value function (that is the whole difference vs PPO):

1. For one prompt, sample G completions from the current ("old") policy.
2. Score each with a reward function — here, checking arithmetic, no humans.
3. Advantage = how much better/worse a completion is **than its own group
   mean**, divided by group std. No critic network anywhere.
4. Update with a PPO-style clipped surrogate: push up probabilities of
   above-average completions, but cap the per-token probability ratio at
   `[1-ε, 1+ε]` so one batch can't wreck the policy.
5. Add a small KL penalty against a frozen reference copy so the policy
   doesn't drift into gibberish that happens to score.
6. Repeat. Reward signal does all the teaching.

Resources: the algorithm comes from the DeepSeekMath paper
([arXiv:2402.03300](https://arxiv.org/abs/2402.03300), Section 4), and its
famous application is DeepSeek-R1's reasoning training
([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).

## Part 1 — `src/ken/countdown.py`: the game and the reward

The countdown game: given N numbers and a target, emit an arithmetic
expression that evaluates to the target using each number at most once.
It is ideal for RL practice because the reward is **verifiable** — a
function checks the answer, so no reward model or human labels are needed.

- `generate_puzzle(rng, n_numbers)` — samples numbers in [1, 20] and builds
  the target by combining them with `+`/`-`/`*`, so every puzzle is
  guaranteed solvable. (If targets were random, half the training signal
  would punish the policy for impossible tasks.)
- `safe_eval(expr)` — evaluates a candidate with `eval` in a locked-down
  namespace (`{"__builtins__": {}}`), rejects anything outside
  `[0-9+\-*/() ]`, and explicitly rejects `**` / `//` (a real crash taught
  us the model discovers exponentiation; `20**20**20` overflows ints).
  Oversized ints (bit_length > 1024) return `None` instead of raising.
- `_uses_allowed_numbers(expr, numbers)` — multisubset check: every number
  in the expression must come from the puzzle, each at most as often as
  given, minimum two numbers (blocks the "just echo the target" hack).
- `reward(expr, numbers, target)` — the ladder both stages optimize:
  `1.0` exact hit, `0.3` valid-but-wrong expression, `0.1` parseable but
  using foreign numbers, `0.0` garbage. Shaping matters: with only
  exact-match reward, a weak policy scores 0 on everything and learns
  nothing (we measured exactly that: mean reward 0.000 for 200
  iterations pre-shaping).

## Part 2 — `src/ken/grpo_toy.py`: GRPO from scratch (Stage 1)

Run: `.venv/bin/python -m ken.grpo_toy` (~10 min on GPU).

- **Vocab / encoding** (`CHARS`, `encode`, `decode`, `prompt_for`) —
  character-level tokens over digits, operators, parens, space, `:`, `=`
  plus BOS/EOS/PAD. A prompt looks like `6 1:7=` and the policy completes
  with e.g. `1+6`. Character level keeps the model tiny and every mechanism
  visible; the price is the policy must also learn digit syntax.
- **`TinyGPT`** — 2-layer, 4-head causal transformer (d=128). ~200k
  params: big enough to learn 2-number countdown, small enough to train in
  minutes. Standard blocks: embedding + learned positions, masked
  multi-head attention, MLP, residuals, final LayerNorm + tied-less head.
- **`sample()`** — rollout collection: autoregressive multinomial sampling
  (temperature 1.0) up to 24 new tokens or EOS. This is step 1 of GRPO;
  sampling from the *old* policy (a frozen deepcopy per iteration) is what
  makes the probability ratios in step 4 meaningful.
- **`completion_logps()`** — the bookkeeping heart: pads a batch of
  prompt+completion sequences, runs one forward pass, and returns
  per-token log-probs plus a mask selecting **completion tokens only**.
  Prompt tokens never get gradient (we train what the policy *says*,
  not what it *reads*).
- **`grpo_step()`** — one update on the batch, run for 2 epochs:
  `ratio = exp(new_logp - old_logp)` per completion token; surrogate =
  `min(ratio·A, clip(ratio)·A)` with the group-normalized advantages `A`;
  plus `beta * KL` with the k3 estimator
  `exp(lr) - lr - 1`, `lr = ref_logp - new_logp`, `beta = 0.02`.
  Loss is averaged over completion tokens, not sequences, so long
  rambling completions don't dominate short ones.
- **`pretrain_format()`** — supervised warm-start: 400 steps of next-token
  cross-entropy on random *valid* expressions (correctness irrelevant).
  Without it the random policy almost never emits anything parseable and
  every group advantage is ~0. This mirrors real pipelines (SFT, then RL)
  and is the single highest-leverage lesson of Stage 1.
- **`train()`** — the loop: 16 prompts × G=12 rollouts per iteration,
  group-normalize rewards → advantages, update, log every 25 iters, finish
  with 3 greedy decodes so you can *see* what was learned.
  Observed: exact-hit rate 0.17 → 0.72 over 300 iterations.

Resources: Andrej Karpathy's
["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) for the
tiny-transformer machinery; the PPO clipped objective it borrows is from
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

## Part 3 — `src/ken/grpo_llm.py`: real LLM GRPO with TRL (Stage 2)

Run: `.venv/bin/python -m ken.grpo_llm --steps 50` (~2 min for 50 steps).

Same algorithm as Stage 1, now with library machinery on
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
(494MB weights, ungated):

- **`make_dataset()`** — 200 countdown prompts with **4** numbers (harder
  than Stage 1's 2), each row carrying `prompt` text plus the raw `numbers`
  / `target` the reward needs. TRL passes those extra columns through to
  the reward function.
- **`countdown_reward(completions, numbers, target)`** — the Stage-1
  `reward()` adapted to chatty models: an instruct model explains instead
  of answering, so we extract every arithmetic-looking fragment with
  `FRAG_RE` and score the **best** one rather than the last line, plus a
  1-shot solved example in the prompt (`(11-3)*(7-4)` = 24) to teach the
  output format. Two bugs lived here: an early version scored only the
  last line (reward stuck at 0), and an empty completion crashed
  `splitlines()[-1]` at step 15 — both covered by tests now.
- **`GRPOConfig` choices for 4GB VRAM** — `beta=0.0` drops the frozen
  reference model entirely (saves ~1GB; also TRL's default), LoRA r=16
  means only adapter weights get gradients/optimizer state, gradient
  checkpointing trades compute for activation memory, bf16 halves weight
  memory, `num_generations=4` × batch 2 keeps rollout KV-cache small, and
  `max_completion_length=64` caps the rambling the reward then has to
  sift through. Observed: reward mean ~0.11–0.20 early → ~0.09–0.25 late
  (avg 0.17, max 0.26), grad norms ~1–1.7 — genuine but early-stage
  signal; 4-number countdown at lr 5e-6 wants more than 50 steps.
- **Reading TRL's log line** — `rewards/countdown_reward/mean` is your
  learning curve; `frac_reward_zero_std` = fraction of groups where every
  completion scored the same (1.0 means no learning possible that step —
  the group-advantage concept made visible); `completions/clipped_ratio`
  = share hitting the 64-token cap (high values mean the model rambles
  instead of answering); `entropy` falling means the policy is
  committing to a strategy.

Resources: the
[TRL GRPO documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
and [TRL repository](https://github.com/huggingface/trl) (this repo pins
`trl>=1.14.2`; defaults quoted above, e.g. `beta=0.0`, were read off the
installed package, so re-check if you upgrade).

## Part 4 — `tests/`: what is covered and why

- `tests/test_countdown.py` — the training signal both stages share:
  the `OverflowError` regression (huge products return `None`, never
  raise), `**`/`//` rejection, the full 1.0/0.3/0.1/0.0 ladder, garbage
  never raising, and puzzle solvability over 50 seeded puzzles.
- `tests/test_grpo_llm.py` — the Stage-2 wrapper: empty/whitespace
  completions score 0.0 without raising (the step-15 crash), and
  best-candidate extraction beats prose (`[1.0, 1.0, 0.1]` case).
- Run: `.venv/bin/python -m pytest tests/ -q` (7 passed) plus
  `.venv/bin/ruff check src/ tests/`.

## Gotchas log (earned, in order)

1. Sparse reward is the default outcome, not bad luck — shaping (0.3/0.1)
   and SFT warm-start fixed Stage 1; 1-shot prompt + candidate extraction
   fixed Stage 2's all-zero start.
2. Models find your reward loopholes: exponentiation (`**`), digit
   concatenation (`262124`), echoing the target. Harden the verifier,
   then add the test.
3. TRL reward functions must never raise on weird completions (empty
   strings happen); a crash at step 15 costs the whole run.
4. Watch output paths: the first Stage-2 run wrote checkpoints to
   `ken/ken/outputs` because the `--out` default already contained `ken/`
   while the command ran from inside `ken/`. Checkpoints now live in
   `outputs/` (gitignored).

## Where to go next

- 200+ Stage-2 steps, or 3-number puzzles, to steepen the curve.
- Re-run with a small `beta` (e.g. 0.01) and a reference model to feel
  what the KL term does to rambling.
- Swap the reward to exact-match-only after warm-up and watch
  `frac_reward_zero_std` — a live demo of why shaping exists.

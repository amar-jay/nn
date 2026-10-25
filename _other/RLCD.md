# RLCD research notes

Research date: 2026-10-25. Scope: TypeSafe's public description, independent implementations, related calibration-reward research, and implications for this lab. This is a broad source review, not a claim to have read every page on the internet or independently reproduced training results.

## What is actually disclosed

TypeSafe introduces **Reinforcement Learning for Calibrated Decisions** as the training direction behind Jev: typed decisions with probabilities that reflect observed outcomes. Its public primer explains calibration across groups of predictions. The launch presents architecture, parallel sampling, and training as separate contributions. The sources inspected do not disclose a reproducible official training objective, optimizer recipe, or training dataset. Independent projects cannot establish what TypeSafe internally implements.

Primary sources:

- [Launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [Documentation index](https://docs.typesafe.ai/llms.txt)
- [Official GitHub organization](https://github.com/typesafe-ai)
- [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python): an LLM-backed comparison interface, not Jev's trainer.
- [WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals): application workflow evaluations, not RLCD training code.

The probability vector and the API's `confidence` field are different. For a Choice with K options, TypeSafe defines confidence as `(max(p)-1/K)/(1-1/K)`. With three options, max probability 0.8 gives confidence 0.7. Calibration against correctness should use the probability whose event is being evaluated, rather than silently interpreting this transformed field as P(correct). [Official confidence definitions](https://docs.typesafe.ai/confidence).

Constrained outputs prevent answers outside the declared schema; they do not prevent incorrect selections. The vendor's own [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) include arithmetic, option-order effects, and contextual interference. These are version-specific observations, not universal properties of all decision models.

## The main mathematical distinction

For a finite categorical event, let `Y ~ q(.|x)` and let the model report `p(.|x)`. A multiclass Brier loss is `sum_k (p_k - 1[k=Y])²`. Its expectation is minimized at `p=q`. Cross-entropy is another proper probability loss. Correctness-only sampling instead maximizes `sum_k p_k q_k`, which favors a mode when q is uncertain.

Calibration, discrimination, accuracy, and operational coverage are separate measurements. A constant base-rate forecast can be calibrated while failing to distinguish individual examples. A useful experiment needs proper scores and risk–coverage behavior, alongside accuracy.

The **random variable matters**: an action-selection distribution, a distribution over possible event labels, and a forecast that a proposed action succeeds are different objects. Several implementations below explicitly depend on that distinction.

## Implementations examined in detail

### 1. Eve-RLCD: selected-action feedback

[Repository](https://github.com/anthony-maio/eve-rlcd) · [reward and loss code](https://github.com/anthony-maio/eve-rlcd/blob/main/rlcd/rewards.py)

The categorical policy samples `a ~ p`; feedback supplies correctness `c`. Its detached reward is `c-p[a]`, with a leave-one-out baseline. The expected REINFORCE gradient equals the gradient of half the multiclass Brier loss when correctness comes from a categorical outcome Y. This is a gradient identity, not a statement that differentiating the numerical expected reward gives that same objective: detaching the reward is essential.

I independently checked the identity by enumerating actions and outcomes for five random soft target distributions: maximum absolute gradient error `2.78e-17` in float64.

The project's write-up reports calibration preservation while accuracy improves, but also documents unequal learning rates in the stable RLVR comparison, shared warmups, mixed hardware, and in-distribution evaluation. It is useful evidence for a concrete recipe, not a controlled demonstration that RLVR generally fails. [Author's experiment and limitations](https://github.com/anthony-maio/eve-rlcd/blob/main/docs/article/2026-09-rlcd-toy.md).

### 2. OpenJev-RLCD: a sampled rationale and differentiable answer distribution

[Paper](https://arxiv.org/abs/2609.38850) · [Repository](https://github.com/ZimmyGao/openjev-rlcd)

This independent reasoning-model study uses a two-stage recipe: calibrate the post-rationale readout, then reinforce rationales with a proper score. It contrasts scoring each rationale's answer distribution with scoring the mixture across rationales. For Brier scores, mixture score equals mean individual score plus between-rationale variance. Good mixture calibration therefore does not establish good single-rationale calibration.

Reported Qwen3-1.7B results include similar GSM8K-Verify accuracy for two-stage RLCD and GRPO (92.2% versus 91.8%), with better confidence ranking for RLCD. Stage 2 helps most on reasoning-essential verification; it adds little on the knowledge-heavy task. These are three-seed experiments on two constructed evaluation settings, not evidence about Jev's internal recipe.

Implementation details checked against [estimators.py](https://github.com/ZimmyGao/openjev-rlcd/blob/main/openjev_rlcd/estimators.py), [train.py](https://github.com/ZimmyGao/openjev-rlcd/blob/main/openjev_rlcd/train.py), and [run scripts](https://github.com/ZimmyGao/openjev-rlcd/blob/main/scripts/two_stage.sh):

- Sample a rationale ending in `Answer:`; read normalized option-token probabilities rather than sample the answer for RLCD.
- Score each readout with `2*p[Y]-sum(p²)`.
- Keep the direct gradient through that readout and a separate score-function gradient through rationale likelihood.
- Stage 1 detaches the rationale likelihood, but still updates the shared model through readout gradients. It does not freeze the backbone.
- Stage 2 retains the direct term, adds a rationale score-function coefficient of 0.3, and anchors to the stage-1 checkpoint using KL reward shaping with beta 0.04.
- The downweighted score-function term is a practical modification of the full unbiased gradient. Its coefficient also scales the KL shaping contribution.
- No PPO ratio/clipping or multi-epoch old-policy batch reuse appears in this RLCD loop. Its GRPO baseline is not identical to this lab's clipped, two-epoch update.
- Stage-1 commands include beta, but the KL shaping enters through the detached rationale likelihood and therefore supplies no gradient in that stage.

Verification: all three upstream exact-enumeration checks passed in the repository's existing virtual environment: interpolated objective, shaping, and mixture estimator. This verifies the mathematical estimator checks, not model-training outcomes.

**Coverage caveat:** [final_tables.py](https://github.com/ZimmyGao/openjev-rlcd/blob/main/openjev_rlcd/final_tables.py) prints `oracle_cov`, selecting coverage using realized errors. [metrics.py](https://github.com/ZimmyGao/openjev-rlcd/blob/main/openjev_rlcd/metrics.py) separately implements predicted-risk-selected coverage and its realized risk. The headline approximately 81% versus 19% coverage should not be treated as a deployment guarantee based solely on the probabilities. Keep the two definitions separate.

### 3. RLCD-lite: sampled decisions and separate stated confidence

[Repository](https://github.com/arnabgho/rlcd-lite) · [reward code](https://github.com/arnabgho/rlcd-lite/blob/main/src/rlcd_lite/rewards.py) · [trainer](https://github.com/arnabgho/rlcd-lite/blob/main/src/rlcd_lite/trainer.py)

This laptop-oriented implementation samples both a choice and a confidence bucket. It offers pure confidence Brier reward, correctness-only reward, and `c-(confidence-c)²`; it also offers decoupled choice/confidence advantages. Its default group baseline centers rewards without standard-deviation whitening. Optional multi-epoch PPO and categorical reference KL are supported.

A particularly useful negative result is the pure stated-confidence Brier trap: predictable wrongness with low stated confidence can score well, so probability honesty alone need not encourage solving. The README also distinguishes stated confidence from normalized choice-logit probabilities; improving one did not reliably calibrate the other. These are findings in its synthetic setup, not an objection to proper scoring of a complete event distribution.

### 4. Gaussian-distribution actions: Laya, Dohnuts, Astro-RLCD

- [Laya training code](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py): perturb detached candidate logits, score their distributions, use Gaussian score-function gradients, and add soft cross-entropy. The current module exposes a soft-CE-only control and notes an experiment finding no gain from the additional term on a typed-decisions split.
- [Dohnuts objective](https://github.com/PsiACE/dohnuts/blob/main/src/dohnuts/rlcd.py) and [recipe](https://github.com/PsiACE/dohnuts/blob/main/docs/rlcd.md): four Gaussian perturbations, log plus spherical reward, ordinal ranked probability score, a group baseline, global advantage scaling, and auxiliary CE. Samples and rewards are detached. This is a distribution-valued action, not sampled arithmetic text.
- [Astro-Han's trainer](https://github.com/Astro-Han/decision-head-rlcd/blob/main/train.py): eight noisy-logit actions and a leave-one-out baseline, with separate CE, accuracy-RL, and exact-accuracy controls. The code explicitly notes that optimizing a proper score of `softmax(z+noise)` need not calibrate the deterministic inference distribution `softmax(z)`.

Do not infer that each project uses the same reward, normalization, supervision, or reference constraint merely because each says RLCD.

### 5. NanoJev: paired-sample Brier estimator

[Experiment specification](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md) · [objective code](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/calibrated_objectives.py)

Its research objective estimates `2*p[Y]-sum(p²)` with independent categorical draws: correctness contributes the first term; pair collisions estimate the squared probability mass. Its local coefficients include all terms involving each sampled action, with a detached conditional baseline. It does not use reward whitening or PPO clipping.

The specification requires the predictive draws to be conditionally independent of the observed outcome. Physical actions that alter their own outcomes need a different event definition. The project compares sampled gradients with direct CE and Brier and explicitly does not establish a general advantage for the sampled estimator. Its game-policy checkpoints and event-calibration experiments are separate evidence.

### 6. Shen's OpenJev / REINFORCE-Analysis

[Repository](https://github.com/ejhshen/OpenJev) · [population objective](https://github.com/ejhshen/OpenJev/blob/main/src/openjev/rlcd/objectives.py) · [analytic estimator](https://github.com/ejhshen/OpenJev/blob/main/src/openjev/rlcd/reinforce.py)

The released objective combines half-Brier and cross-entropy geometry. It computes the known `alpha/2 * sum(p²)` term analytically and samples only outcome-dependent contributions. Behavior-policy probabilities support importance correction; a weighted leave-one-out baseline preserves the intended expectation under exploration. The implementation explicitly rejects reward whitening in this estimator. This is another useful option when full targets are unavailable but the finite policy vector is accessible.

## Additional papers and related work

- [RLCR: Beyond Binary Rewards](https://arxiv.org/abs/2507.16806), [official implementation](https://github.com/damanimehul/RLCR): predates TypeSafe's public RLCD launch. Learns verbalized confidence using correctness plus a calibration reward. It is especially relevant if we keep expression generation and add a correctness-confidence output. Its published large-model runs are not laptop-budget experiments.
- [Rewarding Doubt](https://arxiv.org/abs/2503.02623), [implementation](https://github.com/pasta99/RewardingDoubt): earlier work on confidence expression with RL. It should be included in prior art, separately from TypeSafe.
- [UniJev](https://arxiv.org/abs/2610.25784), [code](https://github.com/wjnwjnwj/UniJev): compares several local RLCD continuations and introduces supervised confidence-bin alignment, CORA. Its results caution against assuming RL automatically improves calibration. The local recipe comparisons use one seed and its CORA/RL training budgets differ. The public CORA code computes detached empirical bin-accuracy targets and differentiates selected-answer probabilities toward them.
- [StartLux-Decision / RLCD²](https://arxiv.org/abs/2610.17309), [release](https://github.com/StartLuxLabs/StartLux-Decision): uses a frozen scorer to reward improvement from pre-reasoning to post-reasoning probability quality, with proper scoring rules and a soft length penalty. The paper describes centered, unwhitened advantages, PPO clipping, and reference KL. Most reported RL gains concern reasoning termination and efficiency; calibration/accuracy effects depend on scale and setting. The inspected public tree exposes serving/evaluation and calibration utilities; I did not locate a complete reproduction trainer for the paper's RLCD² stage there.
- [RLCDAlignBench / Just Ask Jev](https://arxiv.org/abs/2609.29429), [artifacts](https://github.com/sumleo/RLCDAlignBench): evaluates hosted Jev as a detector; it does not recover its training algorithm. Strong ranking can coexist with probability/base-rate mismatch. Reference-field effects and scorer-derived labels matter when interpreting its detector results.
- [Cloudflare Clef announcement](https://blog.cloudflare.com/clef-decision-models/), [open model card](https://huggingface.co/Cloudflare/clef): describes CE plus Brier post-training and a secondary RLCD objective involving ordinal/record rewards and a reference penalty. This is Cloudflare's recipe; it does not disclose TypeSafe's. Open weights do not by themselves supply a fully reproducible training recipe.

## Broader implementation inventory

README and file-tree review also covered the following. Depth varies; this is not a claim to have audited every code path or reproduced their benchmarks.

| Project | What the inspected release establishes |
|---|---|
| [Kev](https://github.com/jaredpalmer/kev) | Trained decision scorer, fitted temperatures, public evaluation tooling; useful supervised control. |
| [OneJev](https://github.com/OmniJev/OneJev) | Multimodal decision model; documented decision training includes CE plus Brier. |
| [Valen](https://github.com/Liuziyu77/Valen) | Multimodal scoring with SFT and an experimental candidate-action RLCD path. |
| [MoJev](https://github.com/MoLeMo-Lab/mojev) | Ranking and Brier calibration training; a distinct recipe. |
| [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) | Public description calls CE+Brier and temperature scaling RLCD; naming does not prove a score-function RL stage. |
| [Vision-Jev](https://github.com/LMDHQ-0420/Vision-Jev) | Static decision-head stage uses supervised probability losses on frozen features; interactive PPO is separate. |
| [Decider](https://github.com/Mapika/decider) | Supervised decision releases plus distinct calibration-aware RL experiments; checkpoint and recipe must be matched. |
| [Jevlike](https://github.com/vinnylarouge/jevlike) | Decision-model training and separate game-policy RL examples. |
| [OmniJev](https://github.com/tinnel123666888/OmniJev) | Multimodal decision heads and holdout temperature calibration; separate from OneJev above. |
| [HeliosLM, now Evastar](https://github.com/tonythetiger168/helioslm) | Broad model/agent research stack and calibration working paper; not a minimal isolated RLCD implementation. |
| [razorback16/OpenJev](https://github.com/razorback16/openjev) | Jev-compatible decision server over multiple open backends; interface compatibility is distinct from training reproduction. |
| [zhihz/OpenJev](https://github.com/zhihz/openjev) | Frozen-model local decision interface; explicitly disclaims proprietary RLCD reproduction. |
| [kw2828/OpenJev](https://github.com/kw2828/OpenJev) | Decision interface and Bayesian/control experiments; explicitly separates them from TypeSafe's undisclosed method. |
| [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | Published card describes parallel constrained inference over existing Qwen weights, not an RL training recipe. |
| [Robot-boundary RLCD study](https://github.com/maskedband1t/RLCD) | Jev-based control/handoff harness and distillation experiments, not a disclosed Jev trainer. |
| [Jev calibration audit](https://github.com/tapesymbolstate/jev-calibration-audit) | API-only empirical audit; evidence about behavior, not internal architecture or reward. |

Also distinguish the unrelated expansion **Reinforcement Learning from Contrastive Distillation**, which shares the RLCD acronym.

## Implications for our Countdown lab — analysis, not an implementation plan

1. **Define the forecast before choosing the loss.** Our policy emits arithmetic expressions with multiple valid solutions. P(generating one exact string) is not P(that string is correct). Per-token softmax probabilities are not correctness confidence either.
2. **Choose a compatible task formulation.** A bounded candidate-selection task, a proposed-expression correctness forecast, or expression generation with an additional confidence report are three different experiments. The first two permit a clearly defined categorical/Boolean event; the last is closer to RLCR than to Jev's no-generation interface.
3. **Do not blindly substitute `c-p(token)` into existing GRPO.** Eve's identity concerns the probability of a complete categorical action and categorical outcome feedback. Token substitution, multiple valid answers, reward whitening, clipping, and old-policy batch reuse change its derivation. A token estimator would need a fresh objective and proof.
4. **Retain direct controls.** With full labels and a small finite option set, direct CE/Brier can avoid sampling variance. If claiming a benefit from RL, match information access, data, initialization, updates, and compute as far as practical.
5. **Keep our frozen reference unless its removal is the tested variable.** Several independent recipes use a fixed anchor; others deliberately do not. Their differing choices do not override the evidence from this lab. A nonzero anchor also changes the optimum of a pure proper-score objective.
6. **Report raw and calibrated probabilities separately.** Use a separate calibration split for temperature fitting; preserve a final test split. Report accuracy, NLL, Brier with its scalar/vector convention, reliability curves/ECE, AURC, predicted-risk-selected coverage with realized risk, and oracle coverage as a separate ranking diagnostic.
7. **Use actual training seeds and save artifacts.** Fix the existing CLI seed-forwarding issue before seed comparisons. Save configuration, coefficients, initialization identity, checkpoint, and per-item predictions. Avoid overwriting the only run checkpoint.

No lab training code was changed and no model training or paid API evaluation was launched.

## Local review provenance

Cloned for inspection under `/tmp/rlcd-research`:

- OpenJev-RLCD: `22db40ad3333f5cf0bc5139097da70e0f8c05018`.
- RLCD-lite: `b0f981fab5e6ce53dd089eae71b141ab1099c31a`.
- HeliosLM/Evastar: `f246a8a9839d17176f1aa2eece45aad45c035848`.

Other repositories were read through public README/tree snapshots and selected source files. Core TypeSafe docs, four Jev-family paper HTML snapshots, model cards, and implementation inventory were cached locally for this review. Those temporary caches are not permanent repository artifacts. Published numerical results remain author-reported unless explicitly identified above as a check performed here.

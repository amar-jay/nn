# Reinforcement Learning

Learning examples imported from [amar-jay/first_rl](https://github.com/amar-jay/first_rl)
at commit `e61f459693551e0818cd4eeb028d2d2bf0443f34`.
Python source and Markdown documentation were imported with the upstream commit
history. Checkpoints and generated NumPy arrays were excluded from every commit.
Imported commit hashes differ because paths were relocated and artifacts removed;
authorship, dates, and commit messages are preserved.

## Contents

- [`ken/`](ken/README.md): a self-contained package for GRPO, RLVR, RLOO,
  reward design, KL regularization, and Countdown experiments. Its own
  `pyproject.toml`, lockfile, and CLI remain together; use its README for setup.
- `simple.py`: random-action environment walkthrough.
- `tabular_methods/`: multi-armed bandit notes, finite Markov decision processes,
  iterative policy evaluation, Q-learning, SARSA, and Expected SARSA.
- `deep_rl_methods/`: deep Q-learning and advantage actor-critic examples.
- `approx_methods/` and `policy_grad_methods/`: placeholders from the source repository.

The finite Markov decision processes directory name corrects the upstream
`decison` spelling. Algorithm implementations otherwise retain the upstream code.

## Running the examples

Dependencies used by these scripts include Gymnasium, NumPy, PyTorch,
Matplotlib, and Seaborn. Some examples open a graphical environment window.
Run scripts from their own directory so generated files stay beside the example.
Training must run before evaluation in examples that load saved weights or Q-tables;
the pretrained artifacts are intentionally excluded from this repository.

These are learning scripts, with overlapping experimental implementations,
rather than a unified training package. No training runs are performed during import.

## Ken history import

`ken/` was imported from the local `ken` repository at source commit
`9cb7d3b40c721ea055cd058f08247d80cad81110`, retaining all 17 commits on its
`main` branch. Paths were prefixed with `modules/reinforcement_learning/ken/`;
model checkpoints, generated arrays, training outputs, and notebook/Python caches
were excluded from history. Commit hashes changed; authors, dates, and messages
were preserved. The original local repository was left untouched.

import matplotlib.pyplot as plt
from ken.grpo_toy import train, infer
from ken.rl import METRIC_NAMES

def show_metrics(metrics):
    if len(metrics) != len(METRIC_NAMES):
        raise ValueError("Old metric format: rerun training to collect corrected metrics.")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    distributions = {"masked_adv_ratio", "surrogate", "masked_surrogate"}
    for ax, name, values in zip(axes.flat, METRIC_NAMES, metrics):
        if name in distributions:
            ax.hist(values, bins=40, edgecolor="white")
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.set_title(f"{name}: distribution")
            ax.set_xlabel("Batch mean value")
            ax.set_ylabel("Number of update epochs")
        else:
            ax.plot(range(1, len(values) + 1), values)
            ax.set_title(name)
            ax.set_xlabel("Update epoch (before optimizer step)")
        ax.grid(alpha=0.3)

    fig.suptitle("Batch metrics: masked = completion tokens; weighted_KL = loss weighting")
    plt.show()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="ken")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--numbers", type=int, default=3)
    training.add_argument("--seed", type=int, default=12345)
    training.add_argument("--rl-step", "--rl_step",
                          choices=["grpo", "ldw", "rlvr", "rlvr_kl", "unclip_kl", "rloo_kl", "rms_kl", "grpo_ref_kl", "rms_ref_kl"], default="grpo")
    evaluation = commands.add_parser("infer", help="Evaluate a saved checkpoint")
    evaluation.add_argument("--count", type=int, default=100)
    evaluation.add_argument("--numbers", type=int, default=3)
    evaluation.add_argument("--show", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=12345)
    diagnostics = commands.add_parser("diagnose", help="Target sensitivity and sampled exploration")
    diagnostics.add_argument("--numbers", type=int, default=3)
    diagnostics.add_argument("--count", type=int, default=100)
    diagnostics.add_argument("--samples", type=int, default=12)
    diagnostics.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.command == "train":
        show_metrics(train(n_numbers=args.numbers, rl_step=args.rl_step,
                           iters=600,
                           sft_steps=2100))
    elif args.command == "diagnose":
        from ken.diagnostics import diagnose
        diagnose(n_numbers=args.numbers, count=args.count, 
                 samples=args.samples, seed=args.seed)
    else:
        infer(count=args.count,
              n_numbers=args.numbers,
              seed=args.seed, 
              show=args.show)

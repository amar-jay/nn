import matplotlib.pyplot as plt
from ken.grpo_toy import train, infer
from ken.rl import METRIC_NAMES

def show_metrics(metrics):
    if len(metrics) != len(METRIC_NAMES):
        raise ValueError("Old metric format: rerun training to collect corrected metrics.")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for ax, name, values in zip(axes.flat, METRIC_NAMES, metrics):
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
    training.add_argument("--numbers", type=int, default=4)
    evaluation = commands.add_parser("infer", help="Evaluate a saved checkpoint")
    evaluation.add_argument("--count", type=int, default=100)
    evaluation.add_argument("--numbers", type=int, default=4)
    evaluation.add_argument("--show", type=int, default=20)
    args = parser.parse_args()
    if args.command == "train":
        show_metrics(train(n_numbers=args.numbers))
    else:
        infer(count=args.count,
              n_numbers=args.numbers,
              show=args.show)

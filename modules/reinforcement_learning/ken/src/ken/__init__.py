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
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else None
    if kind == "train":
        metrics = train()
        show_metrics(metrics)
    elif kind == "infer":
        infer(100)
    else:
        print("Usage: python -m ken [train|infer]")
        sys.exit(1)

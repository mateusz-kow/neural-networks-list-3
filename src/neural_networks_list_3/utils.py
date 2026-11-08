import matplotlib.pyplot as plt


def plot_learning_curves(
    results_dict: dict,
    title: str,
    filename: str,
    metric_name: str = "loss",
    ylabel: str = "Cross-Entropy Loss",
):
    plt.figure(figsize=(10, 6))
    for label, history in results_dict.items():
        plt.plot(history[metric_name], label=label)

    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Zapisano wykres: {filename}")

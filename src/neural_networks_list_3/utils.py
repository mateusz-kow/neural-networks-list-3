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
        if metric_name == "loss":
            color = None

            if metric_name in history:
                train_line, = plt.plot(
                    history[metric_name],
                    label=f"{label} (train)",
                )
                color = train_line.get_color()

            if "test_loss" in history and history["test_loss"]:
                plt.plot(
                    history["test_loss"],
                    color=color,
                    linestyle="--",
                    label=f"{label} (test)",
                )
        elif metric_name in history:
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

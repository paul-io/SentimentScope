import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HISTORY_PATH = os.path.join("model", "history.json")
OUTPUT_PATH  = os.path.join("docs", "training_curve.png")

def load_history() -> dict:
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(
            f"No history.json found at '{HISTORY_PATH}'. "
            f"Run train.py first to generate it."
        )
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)


def plot(history: dict):
    epochs      = list(range(1, len(history["epoch_loss"]) + 1))
    losses      = history["epoch_loss"]
    val_accs    = history["val_accuracy"]
    test_acc    = history.get("test_accuracy")

    # ── Style 
    plt.rcParams.update({
        "figure.facecolor":  "#0e0e11",
        "axes.facecolor":    "#16161a",
        "axes.edgecolor":    "#2a2a2e",
        "axes.labelcolor":   "#9a9aa0",
        "axes.grid":         True,
        "grid.color":        "#2a2a2e",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "xtick.color":       "#6b6b75",
        "ytick.color":       "#6b6b75",
        "text.color":        "#e8e6e0",
        "font.family":       "monospace",
        "legend.facecolor":  "#16161a",
        "legend.edgecolor":  "#2a2a2e",
        "legend.labelcolor": "#9a9aa0",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("IMDB Review Sentiment — Training Results", color="#e8e6e0", fontsize=13, y=1.02)

    # ── Loss plot 
    ax1.plot(epochs, losses, color="#c8b89a", linewidth=2, marker="o",
             markersize=6, markerfacecolor="#0e0e11", markeredgecolor="#c8b89a", markeredgewidth=2)
    ax1.set_title("Training Loss", color="#9a9aa0", fontsize=10, pad=10)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlim(0.7, len(epochs) + 0.3)

    for x, y in zip(epochs, losses):
        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#6b6b75")

    # ── Accuracy plot 
    ax2.plot(epochs, val_accs, color="#7eb8c9", linewidth=2, marker="o",
             markersize=6, markerfacecolor="#0e0e11", markeredgecolor="#7eb8c9", markeredgewidth=2,
             label="Validation")

    if test_acc is not None:
        ax2.axhline(y=test_acc, color="#a8c8a0", linewidth=1.4, linestyle="--",
                    label=f"Test: {test_acc:.2f}%")

    ax2.set_title("Validation Accuracy", color="#9a9aa0", fontsize=10, pad=10)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_xlim(0.7, len(epochs) + 0.3)
    ax2.legend(fontsize=8)

    for x, y in zip(epochs, val_accs):
        ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#6b6b75")

    # ── Save 
    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    plot(load_history())
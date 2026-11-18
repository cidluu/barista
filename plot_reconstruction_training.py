import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_log_to_df(filepath):
    """Reads the training log text file and converts it into a Pandas DataFrame."""
    pattern = re.compile(
        r"Epoch (\d+)/\d+ \(alpha=([\d.]+)\) - Train Loss: ([\d.]+) - Val Target MSE: ([\d.]+)"
    )

    records = []
    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                records.append(
                    {
                        "epoch": int(match.group(1)),
                        "alpha": float(match.group(2)),
                        "train_loss": float(match.group(3)),
                        "val_mse": float(match.group(4)),
                    }
                )

    return pd.DataFrame(records)



def plot_metrics(df, save_path="reconstruction_metrics.png"):
    """Plots Train Loss, Val Target MSE, and Alpha from a structured DataFrame."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel: Losses
    ax1.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss",
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax1.plot(
        df["epoch"],
        df["val_mse"],
        label="Val Target MSE",
        marker="s",
        linewidth=2,
        color="#ff7f0e",
    )

    # Highlight lowest Val MSE
    best_row = df.loc[df["val_mse"].idxmin()]
    ax1.scatter(
        best_row["epoch"],
        best_row["val_mse"],
        color="red",
        zorder=5,
        s=80,
        label=f"Best Val MSE ({best_row['val_mse']:.4f} at Ep {int(best_row['epoch'])})",
    )

    ax1.set_ylabel("Loss / MSE", fontsize=11)
    ax1.set_title(
        "Training Loss, Validation Target MSE, and Alpha Schedule",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Bottom panel: Alpha schedule
    ax2.plot(
        df["epoch"],
        df["alpha"],
        label="Alpha",
        marker="^",
        linewidth=2,
        color="#2ca02c",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Alpha", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.xticks(range(1, df["epoch"].max() + 1, 2))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


# --- Execution ---
if __name__ == "__main__":
    # 1. Convert text file to DataFrame (and optionally save to CSV)
    df = parse_log_to_df("Reconstruction results.txt")
    print(df)
    df.to_csv("reconstruction_results.csv", index=False)

    # 2. Plot using the converted data
    plot_metrics(df)
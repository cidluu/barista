import numpy as np
import matplotlib.pyplot as plt

# 1. Load the pre-saved NumPy array
# Expected shape: (N, 5) -> [Epoch, Train Loss, Train AUC, Val Loss, Val AUC]
data_array = np.load("parsed_results.npy")

# 2. Extract metric columns
epochs = data_array[:, 0]
train_loss = data_array[:, 1]
train_auc = data_array[:, 2]
val_loss = data_array[:, 3]
val_auc = data_array[:, 4]

# 3. Create Figure with Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Train & Val Loss
ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

# Subplot 2: Train & Val AUC
ax2.plot(epochs, train_auc, label="Train AUC", color="#2ca02c", linewidth=2)
ax2.plot(epochs, val_auc, label="Val AUC", color="#d62728", linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC")
ax2.set_title("Training & Validation AUC")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300)
plt.close()
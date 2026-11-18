# import numpy as np
# import matplotlib.pyplot as plt
# import os

# data = np.load("outputs/test_recon_waveforms.npz")
# actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
# reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
# print(reconstructed.shape)
# print(actual.shape)
# print(data["subject_session"])
# seg_idx, ch_idx = 0, 5
# t = np.arange(actual.shape[-1]) / 2048.0  # 2048 Hz sampling rate

# plt.figure(figsize=(12, 4))
# plt.plot(t, actual[seg_idx, ch_idx], label="True", color="black", linewidth=1.2)
# plt.plot(t, reconstructed[seg_idx, ch_idx], label="Reconstructed", color="tab:red", alpha=0.85)
# plt.xlabel("Time (s)")
# plt.ylabel("Z-scored Amplitude")
# plt.title(f"{data['subject_session']} — Segment {seg_idx}, Channel {ch_idx}")
# plt.legend()
# plt.tight_layout()

# out_path = f"outputs/recon_seg{seg_idx}_ch{ch_idx}.png"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Saved plot to {out_path}")


# import os
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # Safe for headless servers without a GUI display
# import matplotlib.pyplot as plt

# data = np.load("outputs/test_recon_waveforms.npz")
# actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
# reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
# subject_session = str(data["subject_session"])

# seg_idx = 0
# y_true = actual[seg_idx]       # (N_channels, 6144)
# y_pred = reconstructed[seg_idx] # (N_channels, 6144)

# num_channels, num_samples = y_true.shape
# t = np.arange(num_samples) / 2048.0  # 2048 Hz sampling rate

# # Vertical spacing between channels (based on signal scale so traces don't collide)
# spacing = np.percentile(np.abs(y_true), 99) * 2.5
# offsets = np.arange(num_channels) * spacing

# # Scale figure height dynamically with number of channels (e.g., ~0.22 inches per channel)
# fig_height = max(8, num_channels * 0.22)
# fig, ax = plt.subplots(figsize=(15, fig_height))

# for ch_idx in range(num_channels):
#     offset = offsets[ch_idx]
#     # Only attach legend labels to the first channel so the legend isn't duplicated 100 times
#     true_label = "True" if ch_idx == 0 else None
#     pred_label = "Reconstructed" if ch_idx == 0 else None

#     ax.plot(t, y_true[ch_idx] + offset, color="black", linewidth=0.7, alpha=0.85, label=true_label)
#     ax.plot(t, y_pred[ch_idx] + offset, color="tab:red", linewidth=0.7, alpha=0.8, label=pred_label)

# # Label the y-axis with channel indices instead of arbitrary offset values
# ax.set_yticks(offsets)
# ax.set_yticklabels([f"Ch {i}" for i in range(num_channels)], fontsize=7)
# ax.set_ylim(-spacing, offsets[-1] + spacing)
# ax.set_xlim(0, t[-1])

# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Channel")
# ax.set_title(f"{subject_session} — All {num_channels} Channels (Segment {seg_idx})")
# ax.legend(loc="upper right")
# fig.tight_layout()

# out_path = f"outputs/plots/all_channels_seg{seg_idx:03d}.png"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# fig.savefig(out_path, dpi=250, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved stacked multi-channel plot to {out_path}")


# import os
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # Safe for headless servers without a GUI display
# import matplotlib.pyplot as plt

# data = np.load("outputs/test_recon_waveforms.npz")
# actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
# reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
# subject_session = str(data["subject_session"])

# seg_idx = 0
# top_k = 5  # Number of best and worst channels to display

# y_true_seg = actual[seg_idx]        # (N_channels, 6144)
# y_pred_seg = reconstructed[seg_idx] # (N_channels, 6144)
# num_channels, num_samples = y_true_seg.shape
# t = np.arange(num_samples) / 2048.0  # 2048 Hz sampling rate

# # 1. Compute Pearson correlation and MSE for every channel in this segment
# corrs = np.zeros(num_channels)
# mses = np.zeros(num_channels)

# for ch in range(num_channels):
#     yt = y_true_seg[ch]
#     yp = y_pred_seg[ch]
#     mses[ch] = np.mean((yt - yp) ** 2)
#     # Guard against flat/zero channels where std == 0
#     if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
#         corrs[ch] = np.corrcoef(yt, yp)[0, 1]
#     else:
#         corrs[ch] = np.nan

# # Sort channels from lowest correlation to highest (ignoring NaNs)
# valid_channels = np.where(~np.isnan(corrs))[0]
# ranked_channels = valid_channels[np.argsort(corrs[valid_channels])]

# worst_channels = ranked_channels[:top_k]         # Lowest r
# best_channels = ranked_channels[-top_k:][::-1]   # Highest r (descending)

# # 2. Plot Worst K (left column) vs. Best K (right column)
# fig, axes = plt.subplots(top_k, 2, figsize=(16, 2.8 * top_k), sharex=True)
# if top_k == 1:
#     axes = axes[np.newaxis, :]

# for row in range(top_k):
#     # --- Left Column: Worst Channels ---
#     ch_w = worst_channels[row]
#     ax_w = axes[row, 0]
#     ax_w.plot(t, y_true_seg[ch_w], color="black", linewidth=1.0, label="True")
#     ax_w.plot(t, y_pred_seg[ch_w], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_w.set_title(f"WORST #{row+1} — Ch {ch_w} (r = {corrs[ch_w]:.3f}, MSE = {mses[ch_w]:.3f})", fontsize=10)
#     ax_w.set_ylabel("Z-score")
#     if row == 0:
#         ax_w.legend(loc="upper right", fontsize=8)

#     # --- Right Column: Best Channels ---
#     ch_b = best_channels[row]
#     ax_b = axes[row, 1]
#     ax_b.plot(t, y_true_seg[ch_b], color="black", linewidth=1.0, label="True")
#     ax_b.plot(t, y_pred_seg[ch_b], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_b.set_title(f"BEST #{row+1} — Ch {ch_b} (r = {corrs[ch_b]:.3f}, MSE = {mses[ch_b]:.3f})", fontsize=10)
#     if row == 0:
#         ax_b.legend(loc="upper right", fontsize=8)

# axes[-1, 0].set_xlabel("Time (s)")
# axes[-1, 1].set_xlabel("Time (s)")
# fig.suptitle(
#     f"{subject_session} — Best vs. Worst Channel Reconstructions (Segment {seg_idx} | "
#     f"Median r = {np.nanmedian(corrs):.3f})",
#     fontsize=13,
#     y=1.01,
# )
# fig.tight_layout()

# out_path = f"outputs/plots/best_vs_worst_seg{seg_idx:03d}.png"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# fig.savefig(out_path, dpi=250, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved best vs. worst comparison to {out_path}")
# print(f"Worst {top_k} channels: {[(int(c), round(float(corrs[c]), 3)) for c in worst_channels]}")
# print(f"Best  {top_k} channels: {[(int(c), round(float(corrs[c]), 3)) for c in best_channels]}")


# import os
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # Safe for headless servers without a GUI display
# import matplotlib.pyplot as plt

# data = np.load("outputs/test_recon_waveforms.npz")
# actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
# reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
# subject_session = str(data["subject_session"])

# ch_idx = 5   # Target channel to inspect across segments
# top_k = 5    # Number of best and worst segments to plot

# y_true_ch = actual[:, ch_idx, :]        # (N_segments, 6144)
# y_pred_ch = reconstructed[:, ch_idx, :] # (N_segments, 6144)
# num_segments, num_samples = y_true_ch.shape
# t = np.arange(num_samples) / 2048.0     # 2048 Hz sampling rate

# # 1. Compute Pearson correlation and MSE across all segments for this channel
# corrs = np.zeros(num_segments)
# mses = np.zeros(num_segments)

# for seg in range(num_segments):
#     yt = y_true_ch[seg]
#     yp = y_pred_ch[seg]
#     mses[seg] = np.mean((yt - yp) ** 2)
#     if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
#         corrs[seg] = np.corrcoef(yt, yp)[0, 1]
#     else:
#         corrs[seg] = np.nan

# # Sort valid segments from lowest correlation to highest
# valid_segments = np.where(~np.isnan(corrs))[0]
# ranked_segments = valid_segments[np.argsort(corrs[valid_segments])]

# worst_segments = ranked_segments[:top_k]         # Lowest r
# best_segments = ranked_segments[-top_k:][::-1]   # Highest r (descending)

# # 2. Plot Worst 5 (left column) vs. Best 5 (right column)
# fig, axes = plt.subplots(top_k, 2, figsize=(16, 2.8 * top_k), sharex=True)
# if top_k == 1:
#     axes = axes[np.newaxis, :]

# for row in range(top_k):
#     # --- Left Column: Worst Segments ---
#     seg_w = worst_segments[row]
#     ax_w = axes[row, 0]
#     ax_w.plot(t, y_true_ch[seg_w], color="black", linewidth=1.0, label="True")
#     ax_w.plot(t, y_pred_ch[seg_w], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_w.set_title(f"WORST #{row+1} — Seg {seg_w} (r = {corrs[seg_w]:.3f}, MSE = {mses[seg_w]:.3f})", fontsize=10)
#     ax_w.set_ylabel("Z-score")
#     if row == 0:
#         ax_w.legend(loc="upper right", fontsize=8)

#     # --- Right Column: Best Segments ---
#     seg_b = best_segments[row]
#     ax_b = axes[row, 1]
#     ax_b.plot(t, y_true_ch[seg_b], color="black", linewidth=1.0, label="True")
#     ax_b.plot(t, y_pred_ch[seg_b], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_b.set_title(f"BEST #{row+1} — Seg {seg_b} (r = {corrs[seg_b]:.3f}, MSE = {mses[seg_b]:.3f})", fontsize=10)
#     if row == 0:
#         ax_b.legend(loc="upper right", fontsize=8)

# axes[-1, 0].set_xlabel("Time (s)")
# axes[-1, 1].set_xlabel("Time (s)")
# fig.suptitle(
#     f"{subject_session} — Channel {ch_idx}: Best vs. Worst {top_k} Segments "
#     f"(Median r = {np.nanmedian(corrs):.3f})",
#     fontsize=13,
#     y=1.01,
# )
# fig.tight_layout()

# out_path = f"outputs/plots/best_vs_worst_ch{ch_idx:03d}.png"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# fig.savefig(out_path, dpi=250, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved channel segment comparison to {out_path}")
# print(f"Worst {top_k} segments: {[(int(s), round(float(corrs[s]), 3)) for s in worst_segments]}")
# print(f"Best  {top_k} segments: {[(int(s), round(float(corrs[s]), 3)) for s in best_segments]}")

# import os
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # Safe for headless servers without a GUI display
# import matplotlib.pyplot as plt

# data = np.load("outputs/test_recon_waveforms.npz", allow_pickle=True)
# actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
# reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
# subject_session = str(data["subject_session"])

# ch_idx = 30   # Target channel to inspect across segments
# top_k = 5    # Number of best and worst segments to plot

# # Automatically grab any saved spatial grouping labels (e.g., ch_destrieux, ch_lobe)
# meta_keys = [k for k in data.keys() if k.startswith("ch_") and not k.endswith("_ids")]
# region_parts = [f"{k.replace('ch_', '')}: {data[k][ch_idx]}" for k in meta_keys]
# ch_label = f"Channel {ch_idx} ({', '.join(region_parts)})" if region_parts else f"Channel {ch_idx}"

# y_true_ch = actual[:, ch_idx, :]        # (N_segments, 6144)
# y_pred_ch = reconstructed[:, ch_idx, :] # (N_segments, 6144)
# num_segments, num_samples = y_true_ch.shape
# t = np.arange(num_samples) / 2048.0     # 2048 Hz sampling rate

# # 1. Compute Pearson correlation and MSE across all segments for this channel
# corrs = np.zeros(num_segments)
# mses = np.zeros(num_segments)

# for seg in range(num_segments):
#     yt = y_true_ch[seg]
#     yp = y_pred_ch[seg]
#     mses[seg] = np.mean((yt - yp) ** 2)
#     if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
#         corrs[seg] = np.corrcoef(yt, yp)[0, 1]
#     else:
#         corrs[seg] = np.nan

# # Sort valid segments from lowest correlation to highest
# valid_segments = np.where(~np.isnan(corrs))[0]
# ranked_segments = valid_segments[np.argsort(corrs[valid_segments])]

# worst_segments = ranked_segments[:top_k]         # Lowest r
# best_segments = ranked_segments[-top_k:][::-1]   # Highest r (descending)

# # 2. Plot Worst 5 (left column) vs. Best 5 (right column)
# fig, axes = plt.subplots(top_k, 2, figsize=(16, 2.8 * top_k), sharex=True)
# if top_k == 1:
#     axes = axes[np.newaxis, :]

# for row in range(top_k):
#     # --- Left Column: Worst Segments ---
#     seg_w = worst_segments[row]
#     ax_w = axes[row, 0]
#     ax_w.plot(t, y_true_ch[seg_w], color="black", linewidth=1.0, label="True")
#     ax_w.plot(t, y_pred_ch[seg_w], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_w.set_title(f"WORST #{row+1} — Seg {seg_w} (r = {corrs[seg_w]:.3f}, MSE = {mses[seg_w]:.3f})", fontsize=10)
#     ax_w.set_ylabel("Z-score")
#     if row == 0:
#         ax_w.legend(loc="upper right", fontsize=8)

#     # --- Right Column: Best Segments ---
#     seg_b = best_segments[row]
#     ax_b = axes[row, 1]
#     ax_b.plot(t, y_true_ch[seg_b], color="black", linewidth=1.0, label="True")
#     ax_b.plot(t, y_pred_ch[seg_b], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
#     ax_b.set_title(f"BEST #{row+1} — Seg {seg_b} (r = {corrs[seg_b]:.3f}, MSE = {mses[seg_b]:.3f})", fontsize=10)
#     if row == 0:
#         ax_b.legend(loc="upper right", fontsize=8)

# axes[-1, 0].set_xlabel("Time (s)")
# axes[-1, 1].set_xlabel("Time (s)")
# fig.suptitle(
#     f"{subject_session} — {ch_label}: Best vs. Worst {top_k} Segments "
#     f"(Median r = {np.nanmedian(corrs):.3f})",
#     fontsize=13,
#     y=1.01,
# )
# fig.tight_layout()

# out_path = f"outputs/plots/best_vs_worst_ch{ch_idx:03d}.png"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# fig.savefig(out_path, dpi=250, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved channel segment comparison to {out_path}")
# print(f"Worst {top_k} segments: {[(int(s), round(float(corrs[s]), 3)) for s in worst_segments]}")
# print(f"Best  {top_k} segments: {[(int(s), round(float(corrs[s]), 3)) for s in best_segments]}")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe for headless servers without a GUI display
import matplotlib.pyplot as plt

data = np.load("outputs/test_recon_waveforms.npz", allow_pickle=True)
actual = data["actual"]               # shape: (N_segments, N_channels, 6144)
reconstructed = data["reconstructed"] # shape: (N_segments, N_channels, 6144)
subject_session = str(data["subject_session"])

channels_to_plot = [0, 15, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]  # List/array of channel indices to plot (or range(actual.shape[1]) for all)
top_k = 5                          # Number of best and worst segments to plot

num_segments, num_channels, num_samples = actual.shape
t = np.arange(num_samples) / 2048.0  # 2048 Hz sampling rate
meta_keys = [k for k in data.keys() if k.startswith("ch_") and not k.endswith("_ids")]

for ch_idx in channels_to_plot:
    if ch_idx < 0 or ch_idx >= num_channels:
        print(f"Skipping ch_idx={ch_idx} (out of bounds for {num_channels} channels)")
        continue

    # Automatically grab any saved spatial grouping labels (e.g., ch_destrieux, ch_lobe)
    region_parts = [f"{k.replace('ch_', '')}: {data[k][ch_idx]}" for k in meta_keys]
    ch_label = f"Channel {ch_idx} ({', '.join(region_parts)})" if region_parts else f"Channel {ch_idx}"

    y_true_ch = actual[:, ch_idx, :]        # (N_segments, 6144)
    y_pred_ch = reconstructed[:, ch_idx, :] # (N_segments, 6144)

    # 1. Compute Pearson correlation and MSE across all segments for this channel
    corrs = np.zeros(num_segments)
    mses = np.zeros(num_segments)

    for seg in range(num_segments):
        yt = y_true_ch[seg]
        yp = y_pred_ch[seg]
        mses[seg] = np.mean((yt - yp) ** 2)
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            corrs[seg] = np.corrcoef(yt, yp)[0, 1]
        else:
            corrs[seg] = np.nan

    # Sort valid segments from lowest correlation to highest
    valid_segments = np.where(~np.isnan(corrs))[0]
    if len(valid_segments) == 0:
        print(f"Skipping ch_idx={ch_idx} (all segments flat/NaN)")
        continue

    ranked_segments = valid_segments[np.argsort(corrs[valid_segments])]
    k_eff = min(top_k, len(ranked_segments))

    worst_segments = ranked_segments[:k_eff]         # Lowest r
    best_segments = ranked_segments[-k_eff:][::-1]   # Highest r (descending)

    # 2. Plot Worst K (left column) vs. Best K (right column)
    fig, axes = plt.subplots(k_eff, 2, figsize=(16, 2.8 * k_eff), sharex=True)
    if k_eff == 1:
        axes = axes[np.newaxis, :]

    for row in range(k_eff):
        # --- Left Column: Worst Segments ---
        seg_w = worst_segments[row]
        ax_w = axes[row, 0]
        ax_w.plot(t, y_true_ch[seg_w], color="black", linewidth=1.0, label="True")
        ax_w.plot(t, y_pred_ch[seg_w], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
        ax_w.set_title(f"WORST #{row+1} — Seg {seg_w} (r = {corrs[seg_w]:.3f}, MSE = {mses[seg_w]:.3f})", fontsize=10)
        ax_w.set_ylabel("Z-score")
        if row == 0:
            ax_w.legend(loc="upper right", fontsize=8)

        # --- Right Column: Best Segments ---
        seg_b = best_segments[row]
        ax_b = axes[row, 1]
        ax_b.plot(t, y_true_ch[seg_b], color="black", linewidth=1.0, label="True")
        ax_b.plot(t, y_pred_ch[seg_b], color="tab:red", linewidth=1.0, alpha=0.85, label="Reconstructed")
        ax_b.set_title(f"BEST #{row+1} — Seg {seg_b} (r = {corrs[seg_b]:.3f}, MSE = {mses[seg_b]:.3f})", fontsize=10)
        if row == 0:
            ax_b.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{subject_session} — {ch_label}: Best vs. Worst {k_eff} Segments "
        f"(Median r = {np.nanmedian(corrs):.3f})",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    out_path = f"outputs/plots/best_vs_worst_ch{ch_idx:03d}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path} | Median r = {np.nanmedian(corrs):.3f}")
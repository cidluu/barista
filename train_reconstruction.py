import copy
import torch
import torch.nn.functional as F
import os

import einops
import numpy as np

from barista.models.model import Barista
from barista.models.utils import seed_everything
from train import (
    apply_overrides,
    freeze_tokenizer,
    get_lr_scheduler,
    get_optimizer,
    load_configs,
    load_pretrained_weights,
    parse_args,
    print_number_of_parmas,
    setup_dataloaders,
)


def compute_recon_loss(recon_patches, target_patches, token_mask, alpha: float):
    """Computes L = L_target + alpha * L_obs."""
    loss_target = F.mse_loss(recon_patches[token_mask], target_patches[token_mask])
    if alpha > 0.0 and (~token_mask).any():
        loss_obs = F.mse_loss(recon_patches[~token_mask], target_patches[~token_mask])
        return loss_target + alpha * loss_obs
    return loss_target


def get_alpha(epoch: int, total_epochs: int = 20) -> float:
    """Alpha = 1.0 for first 10 epochs, then linearly decreases to 0."""
    if epoch < 10:
        return 1.0
    return max(0.0, 1.0 - (epoch - 10) / max(1, (total_epochs - 10)))


def run_recon_epoch(
    model: Barista,
    dataloader,
    device: str,
    alpha: float,
    optimizer=None,
    scheduler=None,
    train: bool = False,
):
    """Run one epoch of masked channel reconstruction training or validation."""
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x = [x_item.to(device) for x_item in batch.x]
        
        batch_size = x[0].shape[0]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            recon_patches, target_patches, token_mask = model.forward_reconstruction(
                x,
                subject_sessions=batch.subject_sessions,
            )
            loss = compute_recon_loss(recon_patches, target_patches, token_mask, alpha)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if train and scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss


# @torch.no_grad()
# def evaluate_leave_one_channel_out(model: Barista, dataloader, device: str):
#     """Masks one channel at a time across all test segments and computes MSE and R^2."""
#     model.eval()
#     all_preds, all_targets = [], []

#     for batch in dataloader:
#         x = [item.to(device) for item in batch.x]
#         num_channels = x[0].shape[-1]

#         for ch_idx in range(num_channels):
#             ch_tensor = torch.tensor([ch_idx], device=device)
#             recon_patches, target_patches, token_mask = model.forward_reconstruction(
#                 x,
#                 subject_sessions=batch.subject_sessions,
#                 masked_channel_indices=ch_tensor,
#             )
#             all_preds.append(recon_patches[token_mask].cpu())
#             all_targets.append(target_patches[token_mask].cpu())

#     preds = torch.cat(all_preds, dim=0)
#     targets = torch.cat(all_targets, dim=0)

#     mse = F.mse_loss(preds, targets).item()
#     ss_res = torch.sum((targets - preds) ** 2)
#     ss_tot = torch.sum((targets - targets.mean()) ** 2)
#     r2 = (1.0 - ss_res / ss_tot).item()
#     return mse, r2

# @torch.no_grad()
# def evaluate_leave_one_channel_out(
#     model: Barista,
#     dataloader,
#     device: str,
#     save_path: str = None,
#     max_segments: int = 100,
# ):
#     """Masks one channel at a time across all test segments, computes MSE and R^2,
#     and optionally saves continuous (N_segments, N_channels, Time) arrays for plotting."""
#     model.eval()
#     all_preds, all_targets = [], []

#     # Buffers for structured waveform saving: list of (B, C, T*L) numpy arrays
#     saved_pred_batches = []
#     saved_target_batches = []
#     segments_collected = 0

#     for batch in dataloader:
#         x = [item.to(device) for item in batch.x]
#         batch_size, _, num_channels = x[0].shape

#         collect_this_batch = save_path is not None and segments_collected < max_segments
#         batch_preds_per_ch = []
#         batch_targets_per_ch = []

#         for ch_idx in range(num_channels):
#             ch_tensor = torch.tensor([ch_idx], device=device)
#             recon_patches, target_patches, token_mask = model.forward_reconstruction(
#                 x,
#                 subject_sessions=batch.subject_sessions,
#                 masked_channel_indices=ch_tensor,
#             )

#             # 1. Keep flat masked patches for overall MSE and R^2 metrics
#             all_preds.append(recon_patches[token_mask].cpu())
#             all_targets.append(target_patches[token_mask].cpu())

#             # 2. Stitch (B, T, L) patches into continuous (B, T*L) waveform for this channel
#             if collect_this_batch:
#                 pred_wave = einops.rearrange(
#                     recon_patches[:, :, ch_idx, :], "b t l -> b (t l)"
#                 ).cpu().numpy()
#                 target_wave = einops.rearrange(
#                     target_patches[:, :, ch_idx, :], "b t l -> b (t l)"
#                 ).cpu().numpy()
#                 batch_preds_per_ch.append(pred_wave)
#                 batch_targets_per_ch.append(target_wave)

#         if collect_this_batch:
#             # Stack across channels -> (B, C, T*L)
#             take = min(batch_size, max_segments - segments_collected)
#             saved_pred_batches.append(np.stack(batch_preds_per_ch, axis=1)[:take])
#             saved_target_batches.append(np.stack(batch_targets_per_ch, axis=1)[:take])
#             segments_collected += take

#     preds = torch.cat(all_preds, dim=0)
#     targets = torch.cat(all_targets, dim=0)

#     mse = F.mse_loss(preds, targets).item()
#     ss_res = torch.sum((targets - preds) ** 2)
#     ss_tot = torch.sum((targets - targets.mean()) ** 2)
#     r2 = (1.0 - ss_res / ss_tot).item()

#     if save_path is not None and saved_target_batches:
#         os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#         actual_arr = np.concatenate(saved_target_batches, axis=0)      # (N_seg, C, 6144)
#         reconstructed_arr = np.concatenate(saved_pred_batches, axis=0) # (N_seg, C, 6144)
#         np.savez_compressed(
#             save_path,
#             actual=actual_arr,
#             reconstructed=reconstructed_arr,
#             subject_session=batch.subject_sessions[0],
#         )
#         print(f"Saved waveforms {actual_arr.shape} to {save_path}")

#     return mse, r2

@torch.no_grad()
def evaluate_leave_one_channel_out(
    model: Barista,
    dataloader,
    device: str,
    save_path: str = None,
    max_segments: int = 100,
):
    """Masks one channel at a time across all test segments, computes MSE and R^2,
    and optionally saves continuous (N_segments, N_channels, Time) arrays and channel metadata."""
    model.eval()
    all_preds, all_targets = [], []

    # Buffers for structured waveform saving: list of (B, C, T*L) numpy arrays
    saved_pred_batches = []
    saved_target_batches = []
    segments_collected = 0
    saved_sub_sesh = None

    for batch in dataloader:
        x = [item.to(device) for item in batch.x]
        batch_size, _, num_channels = x[0].shape

        collect_this_batch = save_path is not None and segments_collected < max_segments
        if collect_this_batch and saved_sub_sesh is None:
            saved_sub_sesh = str(batch.subject_sessions[0])

        batch_preds_per_ch = []
        batch_targets_per_ch = []

        for ch_idx in range(num_channels):
            ch_tensor = torch.tensor([ch_idx], device=device)
            recon_patches, target_patches, token_mask = model.forward_reconstruction(
                x,
                subject_sessions=batch.subject_sessions,
                masked_channel_indices=ch_tensor,
            )

            # 1. Keep flat masked patches for overall MSE and R^2 metrics
            all_preds.append(recon_patches[token_mask].cpu())
            all_targets.append(target_patches[token_mask].cpu())

            # 2. Stitch (B, T, L) patches into continuous (B, T*L) waveform for this channel
            if collect_this_batch:
                pred_wave = einops.rearrange(
                    recon_patches[:, :, ch_idx, :], "b t l -> b (t l)"
                ).cpu().numpy()
                target_wave = einops.rearrange(
                    target_patches[:, :, ch_idx, :], "b t l -> b (t l)"
                ).cpu().numpy()
                batch_preds_per_ch.append(pred_wave)
                batch_targets_per_ch.append(target_wave)

        if collect_this_batch:
            # Stack across channels -> (B, C, T*L)
            take = min(batch_size, max_segments - segments_collected)
            saved_pred_batches.append(np.stack(batch_preds_per_ch, axis=1)[:take])
            saved_target_batches.append(np.stack(batch_targets_per_ch, axis=1)[:take])
            segments_collected += take

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = F.mse_loss(preds, targets).item()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item()

    if save_path is not None and saved_target_batches:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        actual_arr = np.concatenate(saved_target_batches, axis=0)      # (N_seg, C, 6144)
        reconstructed_arr = np.concatenate(saved_pred_batches, axis=0) # (N_seg, C, 6144)

        save_kwargs = {
            "actual": actual_arr,
            "reconstructed": reconstructed_arr,
            "subject_session": saved_sub_sesh,
        }

        meta = getattr(dataloader.dataset, "metadata", None)
        if meta is not None and getattr(meta, "_spatial_groups", None) is not None:
            # Pull all available spatial groupings (e.g., destrieux, lobe, channel, etc.)
            group_names = meta._spatial_groups.get_unique_values_in_col("name")
            for g_name in group_names:
                sg = meta.get_spatial_grouping(saved_sub_sesh, g_name)
                if sg is not None:
                    # Format each channel's group_components tuple into a clean string
                    labels = [
                        " / ".join(str(x) for x in comp) if isinstance(comp, (tuple, list)) else str(comp)
                        for comp in sg.group_components
                    ]
                    save_kwargs[f"ch_{g_name}"] = np.asarray(labels)
                    save_kwargs[f"ch_{g_name}_ids"] = np.asarray(sg.group_ids)

        np.savez_compressed(save_path, **save_kwargs)
        saved_meta_keys = [k for k in save_kwargs if k.startswith("ch_")]
        print(f"Saved waveforms {actual_arr.shape} to {save_path} (metadata keys: {saved_meta_keys})")

    return mse, r2


def main():
    
    args = parse_args()
    dataset_config, train_config, model_config = load_configs(args)

    configs = {"dataset": dataset_config, "train": train_config, "model": model_config}
    configs = apply_overrides(configs, args.override)
    dataset_config = configs["dataset"]
    train_config = configs["train"]
    model_config = configs["model"]

    save_dir = getattr(train_config, "save_dir", "outputs")
    waveforms_file = os.path.join(save_dir, "test_recon_waveforms.npz")

    seed_everything(train_config.seed)

    # Setup dataloaders (reuses existing BrainTreebankDataset pipeline)
    _, train_dataloader, val_dataloader, test_dataloader = setup_dataloaders(
        dataset_config, train_config
    )

    device = train_config.device
    model = Barista(model_config, train_dataloader.dataset.metadata)

    if train_config.checkpoint_path:
        print("Running pretrained model")
        model = load_pretrained_weights(model, train_config.checkpoint_path, device)
        if train_config.optimization.freeze_tokenizer:
            freeze_tokenizer(model)
    else:
        print("Running non-pretrained model")

    # Initialize reconstruction modules (M, H, linear decoder) instead of classification head
    model.create_reconstruction_head()
    model.to(device)

    print_number_of_parmas(model)

    optimizer = get_optimizer(
        model,
        finetune_lr=train_config.optimization.finetune_lr,
        new_param_lr=train_config.optimization.new_param_lr,
    )
    scheduler = get_lr_scheduler(optimizer)

    num_epochs = train_config.epochs
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        alpha = get_alpha(epoch, total_epochs=num_epochs)
        train_loss = run_recon_epoch(
            model, train_dataloader, device, alpha, optimizer, scheduler, train=True
        )
        # Validate strictly on target reconstruction loss (alpha=0.0)
        val_loss = run_recon_epoch(
            model, val_dataloader, device, alpha=0.0, train=False
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} (alpha={alpha:.2f}) "
            f"- Train Loss: {train_loss:.4f} - Val Target MSE: {val_loss:.4f}"
        )

        if best_state is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch + 1,
                "model": copy.deepcopy(model.state_dict()),
                "val_loss": val_loss,
            }

    # # Load best checkpoint and run leave-one-channel-out evaluation
    # model.load_state_dict(best_state["model"])
    # test_mse, test_r2 = evaluate_leave_one_channel_out(model, test_dataloader, device)
    # print(f"\nBEST TEST LEAVE-ONE-CHANNEL-OUT MSE: {test_mse:.4f} | R^2: {test_r2:.4f}")

    model.load_state_dict(best_state["model"])
    save_dir = getattr(train_config, "save_dir", "outputs")
    waveforms_file = os.path.join(save_dir, "test_recon_waveforms.npz")

    test_mse, test_r2 = evaluate_leave_one_channel_out(
        model,
        test_dataloader,
        device,
        save_path=waveforms_file,
        max_segments=1000,
    )
    print(f"\nBEST TEST LEAVE-ONE-CHANNEL-OUT MSE: {test_mse:.4f} | R^2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
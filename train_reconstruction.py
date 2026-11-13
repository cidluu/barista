import copy
import torch
import torch.nn.functional as F

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


@torch.no_grad()
def evaluate_leave_one_channel_out(model: Barista, dataloader, device: str):
    """Masks one channel at a time across all test segments and computes MSE and R^2."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in dataloader:
        x = [item.to(device) for item in batch.x]
        num_channels = x[0].shape[-1]

        for ch_idx in range(num_channels):
            ch_tensor = torch.tensor([ch_idx], device=device)
            recon_patches, target_patches, token_mask = model.forward_reconstruction(
                x,
                subject_sessions=batch.subject_sessions,
                masked_channel_indices=ch_tensor,
            )
            all_preds.append(recon_patches[token_mask].cpu())
            all_targets.append(target_patches[token_mask].cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = F.mse_loss(preds, targets).item()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item()
    return mse, r2


def main():
    args = parse_args()
    dataset_config, train_config, model_config = load_configs(args)

    configs = {"dataset": dataset_config, "train": train_config, "model": model_config}
    configs = apply_overrides(configs, args.override)
    dataset_config = configs["dataset"]
    train_config = configs["train"]
    model_config = configs["model"]

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

    # Load best checkpoint and run leave-one-channel-out evaluation
    model.load_state_dict(best_state["model"])
    test_mse, test_r2 = evaluate_leave_one_channel_out(model, test_dataloader, device)
    print(f"\nBEST TEST LEAVE-ONE-CHANNEL-OUT MSE: {test_mse:.4f} | R^2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
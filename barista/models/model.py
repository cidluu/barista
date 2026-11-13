import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn
# from typing import List
import torch.nn.functional as F
from typing import List, Optional

from barista.data.metadata import Metadata
from barista.models.tokenizer import Tokenizer
from barista.models.transformer import Transformer

#Masked predictor implemented here, this is new code!
class MaskedPredictorH(nn.Module):
    """5-layer MLP predictor network H described in Appendix B."""

    def __init__(self, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # 5 linear layers total: input -> 3 hidden -> output, with GELU + 10% dropout
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Barista(nn.Module):
    def __init__(self, model_config: DictConfig, metadata: Metadata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        
        self.tokenizer = Tokenizer(
            config=model_config.tokenizer,
            metadata=self.metadata,
        )
        
        self.backbone = Transformer(
            **model_config.backbone,
        )
        
        self.d_hidden = model_config.backbone.d_hidden
        
        self.head = None
        
    def create_downstream_head(self, n_chans, output_dim):
        self.channel_weights = nn.Linear(
            n_chans * self.tokenizer.num_subsegments,
            1,
            bias=False,
        )
        self.binary_classifier = nn.Linear(
            self.d_hidden, output_dim
        )

    #code added here, the reconstruction head!
    def create_reconstruction_head(self):
        """Initializes the learnable mask token M, predictor H, and linear patch decoder."""
        patch_len = self.tokenizer.config.temporal_subsegment_len  # L = 512
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_hidden) * 0.02)
        self.predictor_h = MaskedPredictorH(d_hidden=self.d_hidden, dropout=0.1)
        self.recon_linear = nn.Linear(self.d_hidden, patch_len)


    def get_latent_embeddings(self, x: torch.Tensor, subject_sessions: List):
        #  Get tokens
        tokenized_x = self.tokenizer(x, subject_sessions, output_as_list=False)
        
        # Pass through transformer
        latents = self.backbone(
            x=tokenized_x.tokens, 
            seq_lens=tokenized_x.seq_lens, 
            position_ids=tokenized_x.position_ids,
            )
        
        return latents

    def forward(self, x: torch.Tensor, subject_sessions: List):
        
        latents = self.get_latent_embeddings(x, subject_sessions)
        
        # Pass through Task head
        batch_size = x[0].shape[0]
        latents_reshaped = latents.reshape(batch_size, -1, latents.shape[-1])
        x = self.channel_weights(latents_reshaped.permute(0, 2, 1)).squeeze(dim=-1)
        x = self.binary_classifier(x)
    
        return x

    #added this module!!!
    def forward_reconstruction(
        self,
        x: List[torch.Tensor],
        subject_sessions: List[str],
        masked_channel_indices: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.3,
    ):
        """
        Performs spatial masking in latent space and reconstructs raw time-series patches.

        Args:
            x: List of tensors of shape (B, N_timepoints, N_channels).
            subject_sessions: List of session IDs for each sample in the batch.
            masked_channel_indices: Optional 1D tensor of channel indices to mask
                (used during leave-one-channel-out evaluation). If None, randomly masks
                spatial groups up to `mask_ratio` (used during training).
        """
        # 1. Tokenize without adding spatial encodings yet so we can replace target tokens with M
        tokenized_x = self.tokenizer(
            x,
            subject_sessions,
            output_as_list=False,
            add_spatial_encoding_to_tokens=False,
        )

        # Assuming single-session batch (B, T * C, D)
        batch_size = x[0].shape[0]
        num_channels = x[0].shape[-1]
        num_patches = self.tokenizer.num_subsegments  # T = 12
        subseg_len = self.tokenizer.config.temporal_subsegment_len  # L = 512

        tokens = tokenized_x.tokens.view(batch_size, num_patches, num_channels, self.d_hidden).clone()
        spatial_emb = tokenized_x.spatial_embeddings.view(
            batch_size, num_patches, num_channels, self.d_hidden
        )

        # 2. Build boolean channel mask of shape (batch_size, num_channels)
        chan_mask = torch.zeros((batch_size, num_channels), dtype=torch.bool, device=tokens.device)

        if masked_channel_indices is not None:
            # Evaluation mode: mask specific target channel(s) across all time patches
            chan_mask[:, masked_channel_indices] = True
        else:
            # Training mode: mask spatial categories (channel, parcel, or lobe) up to mask_ratio
            sub_sesh = subject_sessions[0]
            sg_row = self.metadata.get_spatial_grouping(
                subject_session=sub_sesh,
                name=self.tokenizer.config.spatial_grouping,
            )
            group_ids = torch.tensor(sg_row.group_ids, device=tokens.device)
            unique_groups = torch.unique(group_ids)

            for b in range(batch_size):
                perm = unique_groups[torch.randperm(len(unique_groups), device=tokens.device)]
                selected_count = 0
                target_count = max(1, int(round(num_channels * mask_ratio)))
                for gid in perm:
                    g_mask = group_ids == gid
                    chan_mask[b] |= g_mask
                    selected_count += g_mask.sum().item()
                    if selected_count >= target_count:
                        break

        # Expand channel mask across all temporal patches: (B, T, C)
        token_mask = einops.repeat(chan_mask, "b c -> b t c", t=num_patches)

        # 3. Replace masked target tokens with shared learnable mask token M, then add E_j (M + E_j)
        tokens[token_mask] = self.mask_token.to(tokens.dtype)
        s_masked = (tokens + spatial_emb).reshape(1, batch_size * num_patches * num_channels, self.d_hidden)

        # 4. Run Transformer backbone G -> MLP predictor H -> Linear patch decoder
        latents = self.backbone(
            x=s_masked,
            seq_lens=tokenized_x.seq_lens,
            position_ids=tokenized_x.position_ids,
        )
        predicted_tokens = self.predictor_h(latents)
        reconstructed_patches = self.recon_linear(predicted_tokens)  # (1, B*T*C, L)

        # Reshape back to (B, T, C, L)
        reconstructed_patches = reconstructed_patches.view(
            batch_size, num_patches, num_channels, subseg_len
        )

        # 5. Extract ground-truth patches P_ij of shape (B, T, C, L)
        target_patches = einops.rearrange(x[0], "b n c -> b c n").unfold(
            dimension=-1,
            size=subseg_len,
            step=self.tokenizer.config.temporal_subsegment_step,
        )
        target_patches = einops.rearrange(target_patches, "b c t l -> b t c l")

        return reconstructed_patches, target_patches, token_mask

    # def get_task_params(self):
    #     return [*self.channel_weights.named_parameters(), *self.binary_classifier.named_parameters()]
    
    def get_task_params(self):
        if hasattr(self, "channel_weights"):
            return [
                *self.channel_weights.named_parameters(),
                *self.binary_classifier.named_parameters(),
            ]
        elif hasattr(self, "recon_linear"):
            return [
                ("mask_token", self.mask_token),
                *self.predictor_h.named_parameters(),
                *self.recon_linear.named_parameters(),
            ]
        raise AttributeError(
            "No task head initialized. Call create_downstream_head() or create_reconstruction_head() first."
        )

    def get_upstream_params(self):
        return [*self.tokenizer.named_parameters(), *self.backbone.named_parameters()]

    
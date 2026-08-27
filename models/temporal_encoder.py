"""Multimodal temporal fusion and latent-token compression for VLA."""

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Fuse image, state, demo-video, and text features into compact latents.

    Each modality is expected as ``[B, T, feature_dim]``. Robot state may also
    be provided as ``[B, feature_dim]`` and is treated as one token.
    """

    def __init__(self, image_dim=1024, state_dim=14, video_dim=1024,
                 text_dim=1024, d_model=256, num_latent_tokens=8,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.modality = nn.Parameter(torch.zeros(4, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers)
        self.queries = nn.Parameter(torch.randn(1, num_latent_tokens, d_model) * 0.02)
        self.compress = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image, state, demo_video, text):
        if state.ndim == 2:
            state = state[:, None, :]
        tokens = torch.cat([
            self.image_proj(image) + self.modality[0],
            self.state_proj(state) + self.modality[1],
            self.video_proj(demo_video) + self.modality[2],
            self.text_proj(text) + self.modality[3],
        ], dim=1)
        encoded = self.temporal(tokens)
        queries = self.queries.expand(encoded.shape[0], -1, -1)
        latents, _ = self.compress(queries, encoded, encoded, need_weights=False)
        return self.norm(latents)


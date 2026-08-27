"""Temporal encoder dedicated to demonstration-video features."""

import torch
from torch import nn


class VideoTemporalEncoder(nn.Module):
    """Encode a video feature sequence and compress it into a few tokens."""

    def __init__(self, video_dim=1024, d_model=256, num_tokens=4,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(video_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.compress = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, video_features):
        video = self.temporal(self.input_proj(video_features))
        queries = self.queries.expand(video.shape[0], -1, -1)
        tokens, _ = self.compress(queries, video, video, need_weights=False)
        return self.norm(tokens)


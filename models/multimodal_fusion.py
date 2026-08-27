"""Fusion of features produced by the existing multimodal backbone."""

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """Fuse image, text, robot-state, and compressed video features."""

    def __init__(self, image_dim=1024, state_dim=14, text_dim=1024,
                 video_dim=256, d_model=256, num_latent_tokens=8,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.modality = nn.Parameter(torch.zeros(4, 1, d_model))
        self.observation_query = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.progress_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.video_retrieval = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers)
        self.queries = nn.Parameter(torch.randn(1, num_latent_tokens, d_model) * 0.02)
        self.compress = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_features, state, text_features, video_tokens):
        if state.ndim == 2:
            state = state[:, None, :]
        image = self.image_proj(image_features) + self.modality[0]
        state_tokens = self.state_proj(state) + self.modality[1]
        text = self.text_proj(text_features) + self.modality[2]
        video = self.video_proj(video_tokens) + self.modality[3]

        # Current observation retrieves the relevant parts of the demonstration.
        observation = torch.cat([
            image.mean(dim=1), state_tokens.mean(dim=1), text.mean(dim=1)
        ], dim=-1)
        observation_query = self.observation_query(observation)[:, None, :]
        progress_query = self.progress_token.expand(video.shape[0], -1, -1)
        queries = torch.cat([observation_query, progress_query], dim=1)
        retrieved, _ = self.video_retrieval(
            queries, video, video, need_weights=False
        )

        tokens = torch.cat([
            image, state_tokens, text, video, retrieved
        ], dim=1)
        encoded = self.fusion(tokens)
        queries = self.queries.expand(encoded.shape[0], -1, -1)
        latents, _ = self.compress(queries, encoded, encoded, need_weights=False)
        return self.norm(latents)
